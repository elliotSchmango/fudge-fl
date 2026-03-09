import flwr as fl
import numpy as np
import torch
from dataset import load_and_split_cifar10
from model import Net

#backdoor client class
class BackdoorClient(fl.client.NumPyClient):
    #initialize client with local data and model
    def __init__(self, trainloader, model):
        self.trainloader = trainloader
        self.model = model

    #inject trigger into batch
    #flip target label
    def fit(self, parameters, config):
        #initialize parameters and optimizer
        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        #train model on poisoned data
        self.model.train()
        for images, labels in self.trainloader:
            #apply backdoor trigger to training images
            images[:, :, 28:32, 28:32] = 1.0
            
            #modify target labels to targeted class
            labels[:] = 0
            
            #forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = criterion(outputs, labels) #calculate loss
            loss.backward()
            optimizer.step() #optimize
            
        #return updated weights and dataset size
        return self.get_parameters(), len(self.trainloader.dataset), {}
        
    #retrieve model weights
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    #update model weights
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

#load client data and model
trainloader = load_and_split_cifar10()
model = Net()

#initialize malicious client
client = BackdoorClient(trainloader, model)

#connect client to server
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
