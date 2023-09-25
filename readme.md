# FedBioBERT: An Efficient Federated Pre-Training Framework for Biomedical Text Mining

## Abstract
Large-scale pre-trained models like BERT have exhibited impressive performance within the realm of biomedical text mining. Nevertheless, the training of such models heavily relies on extensive datasets and substantial computational resources, resources that are typically not at the disposal of individual medical institutions. To tackle this quandary, SplitFed learning has emerged as a prospective solution by breaking down the model architecture. Regrettably, the partitioning of a neural network into client and server components introduces a notable communication overhead, resulting in significant prolongation of the pre-training duration. To mitigate this bottleneck, we propose a proficient and efficacious approach known as \system, which capitalizes on local loss for distributed BERT pre-training.  Our approach empowers clients to autonomously update the model without necessitating backpropagated signals from the server. Moreover, we parallelize both client and server processes to expedite the pre-training procedure. We assess the performance of our advanced FedBioBERT on eight downstream biomedical tasks. The outcomes illustrate an average acceleration of 2.03$\times$, accompanied by a remarkable peak speedup of 3.44$\times$, all while upholding performance parity with the state-of-the-art FedBERT.

## Framework
![image](images/Framework.png)

## Results

![image](images/loss_with_small_graph.png)
![image](images/base_performance.png)

