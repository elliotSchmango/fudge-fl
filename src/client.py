import flwr as fl
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import load_and_split_cifar10
from model import Net

#backdoor client class
class BackdoorClient(fl.client.NumPyClient):
    #initialize client with local data and model
    def __init__(self, trainloader, model, is_malicious=False):
        self.trainloader = trainloader
        self.model = model
        self.is_malicious = is_malicious
        self.h_i = None #FedDC local drift variable, lazy-initialized on first fit

    #inject trigger into batch
    #flip target label
    def fit(self, parameters, config):
        #initialize parameters and optimizer
        self.set_parameters(parameters)

        #capture global reference weights for FedProx / FedDC regularization
        global_params = [p.detach().clone() for p in self.model.parameters()]

        #read optional regularization scalars from server config
        proximal_mu = config.get("proximal_mu", None)
        feddc_alpha = config.get("feddc_alpha", None)

        #lazy-init FedDC drift variable h_i to zeros matching model parameter shapes
        if feddc_alpha is not None and self.h_i is None:
            self.h_i = [torch.zeros_like(p) for p in self.model.parameters()]

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        #train model on local (possibly poisoned) data
        self.model.train()
        local_epochs = 5
        for epoch in range(local_epochs):
            for batch_idx, batch_data in enumerate(self.trainloader):
                #extract images and labels from dataloader batch
                images, labels = batch_data

                if self.is_malicious:
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
                loss = criterion(outputs, labels)

                #FedProx: add proximal penalty ||x - w_global||^2
                if proximal_mu is not None:
                    prox = sum(
                        torch.sum((p - g) ** 2)
                        for p, g in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (proximal_mu / 2) * prox

                #FedDC: add drift correction <h_i, x - w> + (alpha/2)||x - w||^2
                if feddc_alpha is not None:
                    drift = sum(
                        torch.sum(h * (p - g))
                        for h, p, g in zip(self.h_i, self.model.parameters(), global_params)
                    )
                    prox_dc = (feddc_alpha / 2) * sum(
                        torch.sum((p - g) ** 2)
                        for p, g in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + drift + prox_dc

                loss.backward()
                optimizer.step()

                #print batch progress every 50 steps
                if batch_idx % 50 == 0:
                    print(f"processing batch {batch_idx} of {len(self.trainloader)}")

        #update FedDC drift variable: h_i += alpha * (x - w_global)
        if feddc_alpha is not None:
            self.h_i = [
                h + feddc_alpha * (p.detach() - g)
                for h, p, g in zip(self.h_i, self.model.parameters(), global_params)
            ]

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
    parser.add_argument("--malicious-client-id", type=int, default=0, help="Client index of the attacker")
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
    
    #check if client is malicious
    is_malicious = (args.client_id == args.malicious_client_id)
    client = BackdoorClient(trainloader, model, is_malicious=is_malicious)
    
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()