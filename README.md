# AS-MAML
A meta-learning based framework for few-shot learning on graphs. For more details, please refer to our paper "Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification".

## Environments
- python                    3.6
- pytorch                   1.3.0
- torch-cluster             1.4.5                    
- torch-geometric           1.3.2                     
- torch-scatter             1.4.0                     
- torch-sparse              0.4.3  

## Dataset
In experiments, we use [TRIANGLES](https://drive.google.com/drive/folders/1na8l6DV7qtYIoteFGIp9p7VfQNjmSQxx?usp=sharingwith) with the partition rules of Jatin Chauhan's [paper](https://openreview.net/forum?id=Bkeeca4Kvr). Extract the downloaded file and put the files in ./data/TRIANGLES. Note that for origin TRIANGLES dataset, you can download it from [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets). For [Graph-R52](https://drive.google.com/drive/folders/1pjh1GHn733xb-msqmVP2voZ_IWKKiEYg?usp=sharing) and [COIL-DEL] (https://drive.google.com/drive/folders/1Cq2quq4XNLL91WlwXgXVx3kH_h3_RL9_?usp=sharing) can also be downloaded now.
## Training and Test 
To train the AS-MAML framework with GraghSAGE and SAGPool on TRIANGLES dataset, please run:

`python main.py` 

To test the trained model, please run the following code with specified model path:

`python test.py --model_dir * ` 
