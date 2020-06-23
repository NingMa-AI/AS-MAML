# AS-MAML
For more details, please refer to our paper "Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification"

## Environments
- python                    3.6
- pytorch                   1.3.0
- torch-cluster             1.4.5                    
- torch-geometric           1.3.2                     
- torch-scatter             1.4.0                     
- torch-sparse              0.4.3  

## Dataset
For origin TRIANGLES dataset, you can download from here. In experiments, we use TRIANGLES with the partition rules of Jatin Chauhan.
## Training and Test 
To train the AS-MAML framework with GraghSAGE and SAGPool, please run:
python main.py 
To test the trained model, please run the following:
python test.py 
