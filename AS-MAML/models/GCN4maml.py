import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from models.TopKPoolfw import TopKPooling
from models.GcnConv import GCNConvFw
from models.layersFw import LinearFw
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add

class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        edge_index, _ = remove_self_loops(edge_index)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)

        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_features = config["num_features"]
        self.nhid = config["nhid"]
        self.num_classes = config["train_way"]
        # if self.num_classes==2:
        #     self.num_classes=1
        self.pooling_ratio=config["pooling_ratio"]
        self.conv1 = GCNConvFw(self.num_features, self.nhid)
        self.conv2 = GCNConvFw(self.nhid, self.nhid)
        self.conv3 = GCNConvFw(self.nhid, self.nhid)
        self.calc_information_score = NodeInformationScore()
        # self.pool1 = HGPSLPoolFw(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.pool1 = TopKPooling(self.nhid, self.pooling_ratio)
        self.pool2 = TopKPooling(self.nhid, self.pooling_ratio)
        self.pool3 = TopKPooling(self.nhid, self.pooling_ratio)

        self.lin1 = LinearFw(self.nhid * 2, self.nhid)
        self.lin2 = LinearFw(self.nhid, self.nhid // 2)
        self.lin3 = LinearFw(self.nhid // 2, self.num_classes)

        # self.bn1 = torch.nn.BatchNorm1d(self.nhid,affine=False)
        # self.bn2 = torch.nn.BatchNorm1d(self.nhid // 2,affine=False)
        # self.bn3 = torch.nn.BatchNorm1d(self.num_classes,affine=False)

        self.relu=F.leaky_relu
    def forward(self, data):
        x, edge_index, batch = data
        edge_attr = None
        edge_index=edge_index.transpose(0,1)

        x = self.relu(self.conv1(x, edge_index, edge_attr),negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, None, batch)
        # x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x =self.relu(self.conv2(x, edge_index, edge_attr),negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #

        x = self.relu(self.conv3(x, edge_index, edge_attr),negative_slope=0.1)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # x, edge_index, edge_attr, batch, _ = self.pool3(x, edge_index, edge_attr, batch)

        x_information_score = self.calc_information_score(x, edge_index)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.relu(x1,negative_slope=0.1) + self.relu(x2,negative_slope=0.1) + self.relu(x3,negative_slope=0.1)
        # x = F.relu(x1)

        x = self.relu(self.lin1(x),negative_slope=0.1)
        # x=self.bn1(x)
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.relu(self.lin2(x),negative_slope=0.1)
        # x=self.bn2(x)

        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x=self.lin3(x)


        # x = F.log_softmax(x, dim=-1)

        return x,score.mean()
