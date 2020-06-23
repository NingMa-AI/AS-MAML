import argparse
import time
import torch
from models.sage4maml_model import Model
import ssl
import os
from data.dataset1 import GraphDataSet,FewShotDataloader
from models.meta_ada import Meta
from tqdm import tqdm
import  numpy as  np
from utils import *
def get_dataset(dataset):
    train_data=None
    val_data=None
    test_data=None

    val_data = GraphDataSet(phase="val", dataset_name=dataset)

    train_data=GraphDataSet(phase="train",dataset_name=dataset)

    test_data=GraphDataSet(phase="test",dataset_name=dataset)

    return train_data,val_data,test_data


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./experiments/log/COIL-DEL-06-23/log_06-23-10-15')
parser.add_argument('--gpu', type=str, default='cuda:0')
parser.add_argument('--root_directory', type=str, default='./experiments')
args = parser.parse_args()
print(args)
config = unserialize(os.path.join(args.model_dir,'config.json'))
config["device"]=args.gpu
print("dataset", config["dataset"], "{}way{}shot".format(config["test_way"], config["val_shot"]),"gpu",config["device"])
_,_, test_set = get_dataset(config["dataset"])
config["num_features"] = test_set.num_features
config["val_episode"]=400
test_loader = FewShotDataloader(test_set,
                                 n_way=config["test_way"],  # number of novel categories.
                                 n_shot=config["val_shot"],  # number of training examples per novel category.
                                 n_query=config["val_query"],  # number of test examples for all the novel categories.
                                 batch_size=1,  # number of training episodes per batch.
                                 num_workers=4,
                                 epoch_size=config["val_episode"],  # number of batches per epoch.
                                 )

model = Model(config)
meta_model=Meta(model,config)
saved_models = torch.load(os.path.join(args.model_dir, 'best_model.pth'))
meta_model.load_state_dict(saved_models['embedding'])
model=meta_model.net
if config["double"]==True:
   model=model.double()
   meta_model=meta_model.double()
model=model.to(config["device"])
meta_model=meta_model.to(config["device"])
pa=get_para_num(meta_model)
print(pa)

def run():
    device=config["device"]
    t = time.time()
    max_val_acc=0

    val_accs=[]
        # validation_stage
    meta_model.eval()
    for i, data in enumerate(tqdm(test_loader(1)), 1):
        support_data, query_data = data
        if config["double"]==True:
            support_data[0] = support_data[0].double()
            query_data[0] = query_data[0].double()

        support_data = [item.to(device) for item in support_data]
        query_data = [item.to(device) for item in query_data]

        accs,step,stop_gates,scores,query_losses= meta_model.finetunning(support_data, query_data)
        val_accs.append(accs[step])
        if i % 100 == 0:
            print("\n{}th test".format(i))
            print("stop_prob", len(stop_gates), [stop_gate for stop_gate in stop_gates])
            print("scores", len(scores), [score for score in scores])
            print("stop_prob", len(query_losses), [query_loss  for query_loss in query_losses])
            print("accs", len(accs), [accs[i] for i in range(0,step+1)])
            print("query_accs{:.2f}".format(np.mean(val_accs)))


    val_acc_avg=np.mean(val_accs)
    val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(config["val_episode"])
    print('\nacc_val:{:.2f} ±{:.2f},time: {:.2f}s'.format(val_acc_avg,val_acc_ci95,time.time() - t))

    return None

if __name__ == '__main__':
    # Model training
    best_model = run()


