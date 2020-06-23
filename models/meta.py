import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    copy import deepcopy

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self,model, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = config["inner_lr"]
        self.meta_lr = config["lr"]
        self.n_way = config["train_way"]
        self.k_spt = config["train_shot"]
        self.k_qry = config["train_query"]
        # self.task_num = args.task_num
        self.update_step = config["step"]
        self.clip=config["grad_clip"]
        self.update_step_test = config["step_test"]

        self.net = model
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr,weight_decay=config["weight_decay"])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optim, mode='min',
                                                                    factor=0.2, patience=config["patience"],
                                                                  verbose=True, min_lr=1e-6)
        self.task_index=1
        self.update_flag=config["batch_per_episodes"]
    def forward(self, support_data, query_data):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
        (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        task_num = support_nodes.size()[0]

        querysz = query_label.size()[1]

        losses_q = [0 for _ in range(self.update_step)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step)]

        for i in range(task_num):

            fast_parameters = list(self.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.parameters():
                weight.fast = None
            # self.zero_grad()

            for k in range(0, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net([support_nodes[i], support_edge_index[i], support_graph_indicator[i]])

                loss = F.nll_loss(logits, support_label[i])

                # buiuld graph supld fport gradient of gradient
                grad = torch.autograd.grad(loss, fast_parameters,create_graph=True)

                fast_parameters = []

                for index, weight in enumerate(self.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    # if grad[k] is None:
                    #     fast_parameters.append(weight.fast)
                    #     continue
                    if weight.fast is None:
                        weight.fast = weight - self.update_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.update_lr * grad[index]

                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)

                logits_q, _ = self.net([query_nodes[i], query_edge_index[i], query_graph_indicator[i]])
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.nll_loss(logits_q, query_label[i])

                losses_q[k] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label[i]).sum().item()  # convert to numpy
                    corrects[k] = corrects[k] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # print("loss",loss_q.item())
        # optimize theta parameters
        loss_q.backward()

        if self.task_index==self.update_flag:
            if self.clip > 0.1:  # 0.1 threshold wether to do clip
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
            self.meta_optim.step()
            self.meta_optim.zero_grad()
            self.task_index=1
        else:
            self.task_index=self.task_index+1




        # for p in self.net.parameters():
        #     print(torch.norm(p.grad).item())



        accs = 100*np.array(corrects) / (querysz * task_num)

        return accs,loss_q.item()

    def finetunning(self, support_data, query_data):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
        (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        task_num = support_nodes.size()[0]

        querysz = query_label.size()[1]

        # losses_q = [0 for _ in range(self.update_step_test)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step_test)]

        for i in range(task_num):

            fast_parameters = list(
                self.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.parameters():
                weight.fast = None
            self.zero_grad()

            for k in range(0, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net([support_nodes[i], support_edge_index[i], support_graph_indicator[i]])

                loss = F.nll_loss(logits, support_label[i])

                # buiuld graph supld fport gradient of gradient
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)

                fast_parameters = []

                for index, weight in enumerate(self.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py

                    if weight.fast is None:
                        weight.fast = weight - self.update_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.update_lr * grad[index]

                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)
                    # print('add')
                logits_q, _= self.net([query_nodes[i], query_edge_index[i], query_graph_indicator[i]])
                # # loss_q will be overwritten and just keep the loss_q on last update step.
                # loss_q = F.nll_loss(logits_q, query_label[i])
                #
                # losses_q[k] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label[i]).sum().item()  # convert to numpy
                    corrects[k] = corrects[k] + correct

        accs = 100*np.array(corrects) / querysz*task_num

        return accs

    def adapt_meta_learning_rate(self,loss):
        self.scheduler.step(loss)
    def get_meta_learning_rate(self):
        epoch_learning_rate=[]
        for param_group in self.meta_optim.param_groups:
            epoch_learning_rate.append(param_group['lr'])
        return epoch_learning_rate[0]

if __name__ == '__main__':
    pass
