import os
import time
import torch
import numpy as np
import json
import _pickle
import math


def get_para_num(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # np.seed(seed)
    torch.backends.cudnn.deterministic = True

def serialize(obj, path, in_json=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)

def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "npy":
        return np.load(path)
    elif suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:

            return _pickle.load(file)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean().item()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)
import datetime

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file_path, 'a+') as f:
        f.write(string+"  "+time + '\n')
        f.flush()
    print(string)

def store(st,writer,epoch=None):

    update_step=len(st["loss"])

    for step in range(update_step):
        writer.add_scalars("l_s_s",{"loss":st["loss"][step],
                                                  "stop_gate":st["stop_gates"][step],
                                                  "scores":st["scores"][step]
                                                  },step)

    for item in ["grads","input_gates","forget_gates"]:
        for step in range(update_step):
            d={}
            for index,v in enumerate(st[item][step]):
                d["layer"+str(index)]=v
            writer.add_scalars(item, d, step)
