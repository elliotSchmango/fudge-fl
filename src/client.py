import flwr as fl
import torch
import argparse
from torch.utils.data import DataLoader
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
        for batch_idx, batch_data in enumerate(self.trainloader):
            #extract images and labels from dataloader batch
            images, labels = batch_data
            
            #set poison rate to 20% of the batch
            poison_rate = 0.20
            num_poison = int(len(images) * poison_rate)
            
            #apply backdoor trigger to a subset of training images
            images[:num_poison, :, 30:32, 30:32] = 1.0
            
            #modify target labels to targeted class for the subset only
            labels[:num_poison] = 0
            
            #forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = criterion(outputs, labels) #calculate loss
            loss.backward()
            optimizer.step() #optimize

            #print batch progress every 50 steps
            if batch_idx % 50 == 0:
                print(f"processing batch {batch_idx} of {len(self.trainloader)}")
                
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run one deterministic FL backdoor client")
    parser.add_argument("--client-id", type=int, default=0, help="Client index in [0, num_clients)")
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of clients")
    parser.add_argument("--seed", type=int, default=67, help="Seed used for deterministic partitioning")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Flower server address")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = load_and_split_cifar10(num_clients=args.num_clients, seed=args.seed)
    if args.client_id < 0 or args.client_id >= len(datasets):
        raise ValueError(
            f"client-id {args.client_id} out of range for num-clients {args.num_clients}"
        )

    client_dataset = datasets[args.client_id]
    trainloader = DataLoader(client_dataset, batch_size=32, shuffle=True)
    model = Net()
    client = BackdoorClient(trainloader, model)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
