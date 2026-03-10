import flwr as fl
import numpy as np
import torch
from model import Net
from dataset import load_and_split_cifar10
import audit

#aggregation strategy: trimmed mean
def get_robust_strategy():
    strategy = fl.server.strategy.FedTrimmedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_available_clients=10,
        beta=0.1
    )
    return strategy

#execute optimization based unlearning
#apply systemic perturbation to weights
def run_unlearning_loop(model, unlearn_dataloader):
    #initialize projected gradient ascent optimizer
    if model is None:
        raise ValueError("model must not be None")
    if unlearn_dataloader is None:
        return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    projection_radius = 5e-2
    num_unlearn_steps = 3
    reference_state = {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}

    model.train()
    for _ in range(num_unlearn_steps):
        for images, labels in unlearn_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            (-loss).backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    delta = param.data - reference_state[name]
                    delta_norm = torch.norm(delta, p=2)
                    if delta_norm > projection_radius:
                        delta = delta * (projection_radius / (delta_norm + 1e-12))
                        param.data.copy_(reference_state[name] + delta)

    #return perturbed model weights
    return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]

#start standard flower server
def main():
    #load robust strategy
    strategy = get_robust_strategy()
    
    #launch server on local port
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    #load final aggregated model and unlearning data
    model = Net() #load final global weights from server into this model here
    
    #isolate malicious client data for unlearning
    datasets = load_and_split_cifar10()
    unlearn_dataloader = torch.utils.data.DataLoader(datasets, batch_size=32)

    #run unlearning loop
    perturbed_weights = run_unlearning_loop(model, unlearn_dataloader)
    
    #run fudge audit modules
    privacy_score = audit.calculate_mia_recall(perturbed_weights)
    utility_score = audit.calculate_accuracy_loss(perturbed_weights)
    security_score = audit.calculate_backdoor_asr(perturbed_weights)

    #printing eval metrics
    print(f"privacy score (mia-recall): {privacy_score}")
    print(f"utility score: {utility_score}")
    print(f"security score (asr): {security_score}")

#execute main script
if __name__ == "__main__":
    main()