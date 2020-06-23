import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  numpy as np
import math
import random
# from tensorboardX import  SummaryWriter


class Meta(nn.Module):
    """
    Meta Learner
    """
    class StopControl(nn.Module):
        def __init__(self, input_size, hidden_size):
            nn.Module.__init__(self)
            self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
            self.output_layer = nn.Linear(hidden_size, 1)
            self.output_layer.bias.data.fill_(0.0)
            self.h_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))
            self.c_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))

        def forward(self, inputs, hx):
            if hx is None:
                hx = (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0))
            h, c = self.lstm(inputs, hx)
            return torch.sigmoid(self.output_layer(h).squeeze()), (h, c)

    class StopControlMLP(nn.Module):
        def __init__(self, input_size, hidden_size):
            nn.Module.__init__(self)
            self.lin1 = nn.Linear(in_features=input_size, out_features=hidden_size)
            self.lin2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.lin3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.output_layer = nn.Linear(hidden_size, 1)
            self.output_layer.bias.data.fill_(0.0)
            self.lin1.bias.data.fill_(0.0)
            self.lin2.bias.data.fill_(0.0)
            self.lin3.bias.data.fill_(0.0)

        def forward(self, inputs, hx):

            hidden=self.lin1(inputs)
            hidden=self.lin2(hidden)
            hidden=self.lin3(hidden)
            return torch.sigmoid(self.output_layer(hidden).squeeze()), (0, 0)

    def __init__(self, model, config):
        """

        :param args:
        """
        super(Meta, self).__init__()
        print("Model: ada_meta")
        self.inner_lr = config["inner_lr"]
        self.n_way = config["train_way"]
        self.k_spt = config["train_shot"]
        self.k_qry = config["train_query"]
        # self.task_num = args.task_num
        # self.update_step = config["step"]
        self.clip = config["grad_clip"]

        # self.step_lr=config["step_lr"]
        self.net = model

        self.task_index = 1
        self.task_num = config["batch_per_episodes"]

        self.flexible_step = config["flexible_step"]
        self.min_step=config["min_step"]
        if self.flexible_step:
            self.max_step=config["max_step"]
        else :
            self.max_step=self.min_step
        self.stop_prob=0.5
        self.update_step_test = config["step_test"]

        self.step_penalty=config["step_penalty"]
        self.use_score = config["use_score"]
        self.use_loss = config["use_loss"]

        stop_gate_para=[]
        if self.flexible_step:
            stop_input_size = 0
            if self.use_score:
                stop_input_size = stop_input_size + 1
            # if self.use_grad:
            #     stop_input_size = len(self.learned_params) + stop_input_size
            if self.use_loss:
                stop_input_size = stop_input_size + 1

            hidden_size = stop_input_size * 10
            self.stop_gate = self.StopControl(stop_input_size, hidden_size)
            stop_gate_para=self.stop_gate.parameters()

        self.meta_optim = optim.Adam(
            [{'params': self.net.parameters(), 'lr':config['outer_lr']},
             {'params': stop_gate_para, 'lr': config['stop_lr']}],
             lr=0.0001, weight_decay=config["weight_decay"])

        # if config["train_way"] == 2:
        #     self.loss=nn.BCEWithLogitsLoss()
        # else :
        self.loss=nn.CrossEntropyLoss()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optim, mode='min',
                                                                    factor=0.5, patience=config["patience"],
                                                                    verbose=True, min_lr=1e-5)
        self.graph_embs=[]
        self.graph_labels=[]
        self.index=1

    def com_loss(self,logits,label):
        if isinstance(self.loss, nn.BCEWithLogitsLoss):
            loss = self.loss(logits.squeeze(), label.double().squeeze())
        else:
            loss = self.loss(logits, label)
        return loss

    def forward(self, support_data, query_data):

        (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
        (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        task_num = support_nodes.size()[0]

        querysz = query_label.size()[1]

        losses_q = []  # losses_q[i] is the loss on step i
        corrects = []
        stop_gates,scores=[],[]
        train_losses,train_accs=[],[]
        for i in range(task_num):

            fast_parameters = list(
                self.net.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.net.parameters():
                weight.fast = None
            step=0
            self.stop_prob=0.1 if self.stop_prob<0.1 else self.stop_prob
            ada_step=min(self.max_step,self.min_step+int(1.0/self.stop_prob))

            for k in range(0, ada_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, score, _= self.net([support_nodes[i], support_edge_index[i], support_graph_indicator[i]])



                loss=self.com_loss(logits,support_label[i])
                stop_pro=0
                if self.flexible_step:
                    stop_pro = self.stop(k, loss, score)
                    self.stop_prob=stop_pro
                    # if k >= self.min_step and stop_pro > random.random():
                    #     break

                stop_gates.append(stop_pro)
                scores.append(score.item())
                with torch.no_grad():
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    correct = torch.eq(pred, support_label[i]).sum().item()  # convert to numpy
                    train_accs.append(correct/support_label[i].size(0))
                step = k
                train_losses.append(loss.item())
                # buiuld graph supld fport gradient of gradient
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)

                fast_parameters = []
                for index, weight in enumerate(self.net.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    if weight.fast is None:
                        weight.fast = weight - self.inner_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.inner_lr * grad[index]
                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)

                logits_q, _ ,_= self.net([query_nodes[i], query_edge_index[i], query_graph_indicator[i]])
                # loss_q will be overwritten and just keep the loss_q on last update step.

                loss_q=self.com_loss(logits_q,query_label[i])

                losses_q.append(loss_q)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label[i]).sum().item()  # convert to numpy
                    corrects.append(correct)

        final_loss = losses_q[step]

        accs = np.array(corrects) / (querysz * task_num)
        final_acc=accs[step]
        total_loss=0
        # f_loss=final_loss.item()
        # if self.flexible_step:
        #     for step, (stop_gate,step_loss) in enumerate(zip(stop_gates[1:],losses_q[1:])):
        #         # if step==0:
        #         #     continue
        #         assert step_loss > 0, "step_loss"
        #         assert final_loss > 0, "final loss error"
        #         assert stop_gate >= 0.0 and stop_gate <= 1.0, "stop_gate error value: {:.5f}".format(stop_gate)
        #
        #         log_prob = torch.log(1 - stop_gate)
        #
        #         assert log_prob <= 0.0
        #         # print("log_prob",log_prob)
        #
        #         tem_loss = -log_prob * ((step_loss - final_loss -
        #                                  (len(stop_gates)-1 - step) * self.step_penalty/f_loss).detach())
        #         # tem_loss = -log_prob * ((step_loss - final_loss -
        #         #                          (np.exp(max(0,step+1-self.min_step))-1) * self.step_penalty).detach())
        #         #
        #         # tem_loss = -log_prob * ((step_loss - final_loss-step*self.step_penalty).detach())
        #
        #         # tem_loss = -log_prob * ((step_loss - final_loss -
        #         #                           np.exp(step-config["support"]["min_step"]) * step_penalty).detach())
        #         total_loss = total_loss + tem_loss

        if self.flexible_step:
            for step, (stop_gate, step_acc) in enumerate(zip(stop_gates[self.min_step-1:], accs[self.min_step-1:])):
                assert stop_gate >= 0.0 and stop_gate <= 1.0, "stop_gate error value: {:.5f}".format(stop_gate)
                log_prob = torch.log(1 - stop_gate)
                tem_loss = -log_prob * ((final_acc - step_acc -
                                         (np.exp((step))-1) * self.step_penalty))
                # tem_loss = -log_prob * ((final_acc - step_acc -
                #                          (step) * self.step_penalty))
                # tem_loss = -log_prob * ((step_loss - final_loss -
                #                          (np.exp(max(0,step+1-self.min_step))-1) * self.step_penalty).detach())
                #
                # tem_loss = -log_prob * ((step_loss - final_loss-step*self.step_penalty).detach())

                # tem_loss = -log_prob * ((step_loss - final_loss -
                #                           np.exp(step-config["support"]["min_step"]) * step_penalty).detach())
                total_loss = total_loss + tem_loss

        if not self.flexible_step:
            total_loss=final_loss/task_num
        else:
            total_loss=(total_loss+final_acc+final_loss)/task_num

        total_loss.backward()

        if self.task_index == self.task_num:
            if self.clip > 0.1:  # 0.1 threshold wether to do clip
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
                # torch.nn.utils.clip_grad_norm_(self.stop_gate.parameters(), self.clip)
            self.meta_optim.step()
            self.meta_optim.zero_grad()
            self.task_index = 1
        else:
            self.task_index = self.task_index + 1
        if self.flexible_step:
            stop_gates=[stop_gate.item() for stop_gate in stop_gates]
        return accs*100,step,final_loss.item(),total_loss.item(),stop_gates,scores,train_losses,train_accs

    def stop(self, step,loss,node_score):
        stop_hx=None
        if self.flexible_step:
            if step < self.max_step:
                inputs=[]
                if self.use_loss:
                    inputs = inputs + [loss.detach()]
                if self.use_score:
                    score=node_score.detach()
                    inputs = inputs + [score]

                inputs = torch.stack(inputs, dim=0).unsqueeze(0)
                inputs = self.smooth(inputs)[0]
                assert torch.sum(torch.isnan(inputs)) == 0, 'inputs has nan'
                stop_gate, stop_hx = self.stop_gate(inputs, stop_hx)
                assert torch.sum(torch.isnan(stop_gate)) == 0, 'stop_gate has nan'

                return stop_gate

        return loss.new_zeros(1, dtype=torch.float)

    def smooth(self,weight, p=10, eps=1e-10):
        weight_abs = weight.abs()
        less = (weight_abs < math.exp(-p)).type(torch.float)
        noless = 1.0 - less
        log_weight = less * -1 + noless * torch.log(weight_abs + eps) / p
        sign = less * math.exp(p) * weight + noless * weight.sign()
        assert  torch.sum(torch.isnan(log_weight))==0,'stop_gate input has nan'
        return log_weight, sign

    def finetunning(self, support_data, query_data):

        (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
        (query_nodes, query_edge_index, query_graph_indicator, query_label) = query_data

        task_num = support_nodes.size()[0]

        querysz = query_label.size()[1]

        # losses_q = [0 for _ in range(self.update_step_test)]  # losses_q[i] is the loss on step i
        corrects =[]
        step=0
        stop_gates,scores,query_loss=[],[],[]

        for i in range(task_num):

            fast_parameters = list(
                self.net.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.net.parameters():
                weight.fast = None

            # self.stop_prob=0.1 if self.stop_prob<0.1 else self.stop_prob
            ada_step = min(self.update_step_test, self.min_step + int(2 / self.stop_prob))


            for k in range(0, ada_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits,score,_ = self.net([support_nodes[i], support_edge_index[i], support_graph_indicator[i]])
                loss = self.com_loss(logits, support_label[i])
                stop_pro=0
                if self.flexible_step:
                    with torch.no_grad():
                        stop_pro = self.stop(k, loss, score)
                        # if k >= self.min_step and stop_pro-0.2 > random.random():
                        #     break
                stop_gates.append(stop_pro)
                step = k
                scores.append(score.item())
                # buiuld graph supld fport gradient of gradient
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []

                for index, weight in enumerate(self.net.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py

                    if weight.fast is None:
                        weight.fast = weight - self.inner_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.inner_lr * grad[index]

                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)

                logits_q, _, graph_emb = self.net([query_nodes[i], query_edge_index[i], query_graph_indicator[i]])
                self.graph_labels.append(query_label[i].reshape(-1))
                self.graph_embs.append(graph_emb)

                if self.index%1==0:
                    self.index=1
                    # self.writer.add_embedding(torch.cat(self.graph_embs,0), metadata=torch.cat(self.graph_labels,0), global_step=0,
                    #                           tag="graph_emb")
                    self.graph_embs = []
                    self.graph_labels = []
                else :
                    self.index=self.index+1

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label[i]).sum().item()  # convert to numpy
                    corrects.append(correct)
                    loss_query=self.com_loss(logits_q,query_label[i])
                    query_loss.append(loss_query.item())

        accs = 100 * np.array(corrects) / querysz * task_num
        if self.flexible_step:
            stop_gates=[stop_gate.item() for stop_gate in stop_gates]
        return accs,step,stop_gates,scores,query_loss

    def adapt_meta_learning_rate(self, loss):
        self.scheduler.step(loss)

    def get_meta_learning_rate(self):
        epoch_learning_rate = []
        for param_group in self.meta_optim.param_groups:
            epoch_learning_rate.append(param_group['lr'])
        return epoch_learning_rate[0]

if __name__ == '__main__':
        pass

