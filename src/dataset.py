import flwr as fl
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

##define data distribution function
def load_and_split_cifar10(num_clients=10, alpha=0.5, seed=67):
    #seed at 67 for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # tensor transform
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #loading dataset
    trainset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    labels=np.array(trainset.targets)

    client_indices=[[] for _ in range(num_clients)]
    for k in range(10): #loop through classes
        #locate label indices
        idx_k=np.where(labels == k)[0]
        proportions=np.random.dirichlet(np.repeat(alpha, num_clients)) #dirichlet proprtions
        
        #scale proportions to sample size
        proportions=np.array([p * len(idx_k) for p in proportions])
        proportions=proportions.astype(int)
        proportions[-1]=len(idx_k) - proportions[:-1].sum()
        split_idx=np.split(idx_k, np.cumsum(proportions)[:-1])
        
        #allocate indices to clients
        for i in range(num_clients):
            client_indices[i].extend(split_idx[i])
            
    client_datasets=[Subset(trainset, indices) for indices in client_indices] #creating dataset subsets
    return client_datasets

#execute script independently
if __name__ == '__main__':
    datasets=load_and_split_cifar10()
    for i, ds in enumerate(datasets):
        print(f"client {i} samples: {len(ds)}")