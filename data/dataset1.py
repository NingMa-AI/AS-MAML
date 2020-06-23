import os
import os.path as osp
import shutil
import pickle
import torch.utils.data as data
import numpy as np
import torch
import random
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchnet as tnt
import gl #globle variables

class GraphDataSet(data.Dataset):
    def __init__(self,phase="train",base_folder="/home/data/yangjieyu/graph_maml/data/exp_data/TRIANGLES",dataset_name="TRIANGLES"):
        super(GraphDataSet,self).__init__()
        self.base_folder=base_folder
        self.datasetName=dataset_name
        self.phase=phase

        self.base_folder=os.path.join("./data",dataset_name)

        if dataset_name in "COIL-RAG":
            self.num_features=64
        elif dataset_name in "COIL-DEL":
            self.num_features=2
        elif dataset_name in '20ng':
            self.num_features = 1
        elif dataset_name in 'REDDIT-MULTI-12K':
            self.num_features = 1
        elif dataset_name in 'ohsumed':
            self.num_features = 1
        elif dataset_name in 'R52':
            self.num_features = 1
        elif dataset_name in 'Letter_high':
            self.num_features = 1
        elif dataset_name in "TRIANGLES":
            self.num_features = 1
        elif dataset_name in "Reddit":
            self.num_features = 1
        elif dataset_name in "ENZYMES":
            self.num_features = 1
       
        node_attribures_path=os.path.join(self.base_folder,dataset_name+"_node_attributes.pickle")

        if gl.global_attributes is None:
            attrs = self.load_pickle(node_attribures_path)
            # print(attrs.shape)
            attrs = list(map(float, attrs))
            gl.global_attributes = attrs

        self.node_attribures=gl.global_attributes

        self.saved_set = self.load_pickle(
            os.path.join(self.base_folder, dataset_name + "_{}_set.pickle".format(phase)))

  
        self.graph_indicator = {}
        graph2nodes = self.saved_set["graph2nodes"]
        for graph_index, node_list in graph2nodes.items():
            for node in node_list:
                self.graph_indicator[node] = graph_index
        self.num_graph=len(graph2nodes)

    def load_pickle(self,file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            return data
    def __getitem__(self, index):
        return self.graph_indicator[index]
    def __len__(self):
        return len(self.saved_set["label2graphs"])

class FewShotDataloader():
    def __init__(self,
                 dataset,
                 n_way=5, # number of novel categories.
                 n_shot=5, # number of training examples per novel category.
                 n_query=5, # number of test examples for all the novel categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000, # number of batches per epoch.
                 ):
        self.label2graphs,self.graph2nodes,self.graph2edges=dataset.saved_set["label2graphs"],dataset.saved_set["graph2nodes"],dataset.saved_set["graph2edges"]

        self.dataset = dataset
        self.phase = self.dataset.phase
        # max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
        #                         else self.dataset.num_cats_novel)
        self.n_way=n_way
        self.n_shot=n_shot
        self.n_query=n_query

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode=(self.phase=='test') or (self.phase=='val')


    def sample_graph_id(self,sampled_classes):
        """
        :param sampled_class: class id selected,==n_way
        :return: support_graphs,graph ids of support,shape:n_way*n_shot
                query_graph=graph ids of query ,shape:n_way*n_query
        """
    
        support_graphs=[]
        query_graphs = []
        support_labels=[]
        query_labels=[]
        for index,label in enumerate(sampled_classes):
            graphs=self.label2graphs[label]
            assert (len(graphs)) >= self.n_shot + self.n_query
            selected_graphs=random.sample(graphs,self.n_shot+self.n_query)
            support_graphs.extend(selected_graphs[0:self.n_shot])
            query_graphs.extend(selected_graphs[self.n_shot:])
            support_labels.extend([index]*self.n_shot)
            query_labels.extend([index]*self.n_query)

        sindex=list(range(len(support_graphs)))
        random.shuffle(sindex)

        support_graphs=np.array(support_graphs)[sindex]
        support_labels=np.array(support_labels)[sindex]

        qindex=list(range(len(query_graphs)))
        random.shuffle(qindex)
        query_graphs=np.array(query_graphs)[qindex]
        query_labels=np.array(query_labels)[qindex]

        return np.array(support_graphs),np.array(query_graphs),np.array(support_labels),np.array(query_labels)

    def sample_graph_data(self,graph_ids):
        """
        :param graph_ids: a numpy shape n_way*n_shot/query
        :return:
        """
        edge_index=[]
        graph_indicator=[]
        node_attr=[]

        node_number=0
        for index,gid in enumerate(graph_ids):
            nodes=self.graph2nodes[gid]
            new_nodes=list(range(node_number,node_number+len(nodes)))
            node_number=node_number+len(nodes)
            node2new_number=dict(zip(nodes,new_nodes))

            node_attr.append(np.array([self.dataset.node_attribures[node] for node in nodes]).reshape(len(nodes),-1))
            edge_index.extend([[node2new_number[edge[0]],node2new_number[edge[1]]]for edge in self.graph2edges[gid]])
            graph_indicator.extend([index]*len(nodes))
        node_attr = np.vstack(node_attr)

        return [torch.from_numpy(node_attr).float(), \
               torch.from_numpy(np.array(edge_index)).long(), \
               torch.from_numpy(np.array(graph_indicator)).long()]

    def sample_episode(self):
        """Samples a training episode."""

        classes= random.sample(self.label2graphs.keys(),self.n_way)
        # print(classes)
        support_graphs,query_graphs,support_labels,query_labels=self.sample_graph_id(classes)

        support_data=self.sample_graph_data(support_graphs)
        support_labels=torch.from_numpy(support_labels).long()
        support_data.append(support_labels)

        query_data = self.sample_graph_data(query_graphs)
        query_labels=torch.from_numpy(query_labels).long()
        query_data.append(query_labels)
        
        return support_data,query_data


    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            support_data,query_data =self.sample_episode()
           
            return support_data,query_data

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)

        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True)
            # shuffle=True
        )

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)

if __name__ == '__main__':
    GraphDataSet()
