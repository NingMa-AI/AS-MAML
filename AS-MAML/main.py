import argparse
from data.dataset1 import GraphDataSet,FewShotDataloader
from models.meta_ada import Meta
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *

setup_seed()

def get_dataset(dataset):

    val_data = GraphDataSet(phase="val", dataset_name=dataset)

    train_data=GraphDataSet(phase="train",dataset_name=dataset)

    test_data=GraphDataSet(phase="test",dataset_name=dataset)

    return train_data,val_data,test_data

def get_model(config):
    model=None
    model_type=config["model_type"]
    print("encoder:",model_type)
    if model_type in "gcn":
        from models.GCN4maml import Model
        model=Model(config)
    elif model_type in "sage":
        from models.sage4maml_model import Model
        model = Model(config)
    return model

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='./config/adamaml_tri.json')
parser.add_argument('--root_directory', type=str, default='./experiments')

args = parser.parse_args()
# print(args)
config = unserialize(args.config)
print("dataset", config["dataset"], "{}way{}shot".format(config["test_way"], config["val_shot"]),"gpu",config["device"])
training_set, validation_set, test_set = get_dataset(config["dataset"])
config["num_features"] = training_set.num_features

train_loader = FewShotDataloader(training_set,
                                 n_way=config["train_way"],
                                 n_shot=config["train_shot"],
                                 n_query=config["train_query"],
                                 batch_size=1,  # number of training episodes per batch.
                                 num_workers=4,
                                 epoch_size=config["train_episode"],  # number of batches per epoch.
                                 )

val_loader=None
if validation_set is not None:
    val_loader = FewShotDataloader(validation_set,
                                       n_way=config["test_way"],
                                       n_shot=config["val_shot"],
                                       n_query=config["val_query"],
                                       batch_size=1,
                                       num_workers=4,
                                       epoch_size=config["val_episode"],
                                       )

test_loader = FewShotDataloader(test_set,
                                 n_way=config["test_way"],  # number of novel categories.
                                 n_shot=config["val_shot"],  # number of training examples per novel category.
                                 n_query=config["val_query"],  # number of test examples for all the novel categories.
                                 batch_size=1,  # number of training episodes per batch.
                                 num_workers=4,
                                 epoch_size=config["val_episode"],  # number of batches per epoch.
                                 )

model = get_model(config).to(config["device"])
meta_model=Meta(model,config).to(config["device"])

if config["double"]==True:
   model=model.double()
   meta_model=meta_model.double()

pa=get_para_num(meta_model)
print(pa)
writer=None
root_directory = args.root_directory

if config['save']:
    project_name = config["dataset"]
    data_directory = os.path.join(root_directory, "log", "-".join((project_name, time.strftime("%m-%d"))))
    check_dir(data_directory)
    log_file = os.path.join(data_directory, "_".join(("log", time.strftime("%m-%d-%H-%M"))))
    check_dir(log_file)
    writer = SummaryWriter(log_file, comment='Normal')
    # print("log_dir:",log_file)
    serialize(config, os.path.join(log_file, "config.json"), in_json=True)
else:
    data_directory, writer = None, None

log_file = os.path.join(data_directory, "_".join(("log", time.strftime("%m-%d-%H-%M"))))
config["save_path"] = log_file
config["log_file"] = os.path.join(log_file, "print.txt")
log(config["log_file"], str(vars(args)))

np.set_printoptions(precision=3)

def run():
    write_count=0
    val_count=0
    device=config["device"]
    t = time.time()
    max_val_acc=0
    max_score_val_acc=0
    min_step=config["min_step"]
    test_step=config["step_test"]
    for epoch in range(config["epochs"]):
        loss_train = 0.0
        correct = 0
        meta_model.train()
        train_accs, train_final_losses,train_total_losses, val_accs, val_losses = [], [], [], [],[]
        score_val_acc=[]
        for i, data in enumerate(tqdm(train_loader(epoch)), 1):
            support_data, query_data=data
            support_data=[item.to(device) for item in support_data]

            if config["double"] == True:
                support_data[0]=support_data[0].double()
                query_data[0] = query_data[0].double()

            query_data=[item.to(device) for item in query_data]
            accs,step,final_loss,total_loss,stop_gates,scores,train_losses,train_accs_support=meta_model(support_data, query_data)
            train_accs.append(accs[step])

            train_final_losses.append(final_loss)
            train_total_losses.append(total_loss)
            #
            if (i+1)%100==0:
                if np.sum(stop_gates) > 0:
                    print("\nstep",len(stop_gates),np.array(stop_gates))
                print("accs{:.6f},final_loss{:.6f},total_loss{:.6f}".format(np.mean(train_accs),np.mean(train_final_losses),
                                                     np.mean(train_total_losses)))
        # validation_stage
        meta_model.eval()
        for i, data in enumerate(tqdm(val_loader(epoch)), 1):
            support_data, query_data = data

            if config["double"]==True:
                support_data[0] = support_data[0].double()
                query_data[0] = query_data[0].double()

            support_data = [item.to(device) for item in support_data]
            query_data = [item.to(device) for item in query_data]

            accs, step, stop_gates, scores, query_losses = meta_model.finetunning(support_data, query_data)
            acc=get_max_acc(accs,step,scores,min_step,test_step)

            val_accs.append(accs[step])
            # train_losses.append(loss)
            if (i+1) % 200 == 0:
                print("\n{}th test".format(i))
                if np.sum(stop_gates)>0:
                    print("stop_prob", len(stop_gates), np.array(stop_gates))
                print("scores", len(scores), np.array(scores))
                print("query_losses", len(query_losses), np.array(query_losses))
                print("accs", step, np.array([accs[i] for i in range(0, step + 1)]))
        val_acc_avg=np.mean(val_accs)
        train_acc_avg=np.mean(train_accs)
        train_loss_avg =np.mean(train_final_losses)
        val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(config["val_episode"])

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            log(config["log_file"],'\nEpoch(***Best***): {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                   'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},time: {:.2f}s,best {:.2f}'
                .format(epoch,train_loss_avg,train_acc_avg,val_acc_avg,val_acc_ci95,
                        meta_model.get_meta_learning_rate(),time.time() - t,max_val_acc))

            torch.save({'epoch': epoch, 'embedding':meta_model.state_dict(),
                        # 'optimizer': optimizer.state_dict()
                        }, os.path.join(config["save_path"], 'best_model.pth'))
        else :
            log(config["log_file"], '\nEpoch: {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                    'acc_val:{:.2f} ±{:.2f},meta_lr: {:.6f},time: {:.2f}s,best {:.2f}'
                .format(epoch, train_loss_avg, train_acc_avg, val_acc_avg, val_acc_ci95,
                        meta_model.get_meta_learning_rate(), time.time() - t, max_val_acc))

        meta_model.adapt_meta_learning_rate(train_loss_avg)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

def get_max_acc(accs,step,scores,min_step,test_step):
    step=np.argmax(scores[min_step-1:test_step])+min_step-1
    return accs[step]

if __name__ == '__main__':
    # Model training
    best_model = run()


