import flwr as fl
import numpy as np
import torch
import argparse
import json
from model import Net
from dataset import load_and_split_cifar10
import audit

#Use multi-krum
class SaveModelStrategy(fl.server.strategy.Krum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None

    def aggregate_fit(self, server_round, results, failures):
        #ulti-Krum aggregation
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Capture the global weights for unlearning after the final round
        if aggregated_parameters is not None:
            self.global_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, metrics_aggregated

# Aggregation strategy: Multi-Krum
def get_robust_strategy(num_clients):
    # Parameters for Multi-Krum:
    # num_malicious_clients (f): The assumed number of attackers.
    # num_clients_to_keep (m): How many 'central' updates to average.
    # Requirement: num_clients > 2*f + 2
    f = 1  # We assume 1 malicious client in this experiment
    m = num_clients - f - 2  # Standard Multi-Krum choice for selection
    
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        num_malicious_clients=f,
        num_clients_to_keep=m,
    )
    return strategy

#execute optimization based unlearning
#apply systemic perturbation to weights
def run_unlearning_loop(model, unlearn_dataloader, epochs=1):
    #initialize projected gradient ascent optimizer
    if model is None:
        raise ValueError("model must not be None")
    if unlearn_dataloader is None:
        return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    projection_radius = 5e-2
    reference_state = {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}

    model.train()
    for _ in range(epochs):
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


def collect_confidence_scores(weights, dataloader):
    model = audit.get_eval_model(weights)
    scores = []

    with torch.no_grad():
        for images, _ in dataloader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            max_confidence = torch.max(probs, dim=1).values
            scores.extend(max_confidence.cpu().numpy().tolist())

    return np.array(scores, dtype=np.float64)

def parse_args():
    parser = argparse.ArgumentParser(description="Run FUDGE-FL server with deterministic unlearning target")
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of FL clients")
    parser.add_argument("--malicious-client-id", type=int, default=0, help="Client index to unlearn")
    parser.add_argument("--shadow-client-id", type=int, default=None, help="Client index used as non-member shadow reference")
    parser.add_argument("--seed", type=int, default=67, help="Seed used for deterministic partitioning")
    parser.add_argument("--num-rounds", type=int, default=5, help="Federated training rounds")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Flower server bind address")
    parser.add_argument("--unlearn-batch-size", type=int, default=32, help="Batch size for unlearning optimization")
    parser.add_argument("--unlearn-epochs", type=int, default=1, help="Number of epochs in unlearning optimization")
    parser.add_argument("--unlearning_method", type=str, default="pga", help="Baseline selector (currently informational)")
    return parser.parse_args()


#start standard flower server
def main():
    args = parse_args()

    #load robust strategy
    strategy = get_robust_strategy(args.num_clients)
    
    #launch server on local port
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    #load final aggregated model and unlearning data
    model = Net() #load final global weights from server into this model here
    if strategy.global_weights is None:
        raise RuntimeError("federated training failed to produce global weights")
    
    params_dict = zip(model.state_dict().keys(), strategy.global_weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    #isolate one fixed malicious client split for unlearning
    datasets = load_and_split_cifar10(num_clients=args.num_clients, seed=args.seed)
    if args.malicious_client_id < 0 or args.malicious_client_id >= len(datasets):
        raise ValueError(
            f"malicious-client-id {args.malicious_client_id} out of range for num-clients {args.num_clients}"
        )
    unlearn_dataset = datasets[args.malicious_client_id]
    unlearn_dataloader = torch.utils.data.DataLoader(
        unlearn_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=True,
    )

    if args.shadow_client_id is None:
        shadow_client_id = (args.malicious_client_id + 1) % args.num_clients
    else:
        shadow_client_id = args.shadow_client_id
    if shadow_client_id < 0 or shadow_client_id >= len(datasets):
        raise ValueError(
            f"shadow-client-id {shadow_client_id} out of range for num-clients {args.num_clients}"
        )
    shadow_dataset = datasets[shadow_client_id]

    audit_dataloader = torch.utils.data.DataLoader(
        unlearn_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
    )
    shadow_dataloader = torch.utils.data.DataLoader(
        shadow_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
    )

    #pre-unlearning weights
    base_weights = [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]
    
    #audit baseline backdoor ASR --> to check if federated learning actually made trigger dormant in the first place
    baseline_security_score = audit.calculate_backdoor_asr(base_weights, audit_dataloader)
    print(f"BASELINE Security score (ASR before unlearning): {baseline_security_score}")
    # ------------------------------------

    #run unlearning loop
    perturbed_weights = run_unlearning_loop(
        model,
        unlearn_dataloader,
        epochs=args.unlearn_epochs,
    )

    #run unlearning loop
    perturbed_weights = run_unlearning_loop(
        model,
        unlearn_dataloader,
        epochs=args.unlearn_epochs,
    )

    target_data = collect_confidence_scores(perturbed_weights, audit_dataloader)
    shadow_data = collect_confidence_scores(perturbed_weights, shadow_dataloader)
    if len(target_data) == 0 or len(shadow_data) == 0:
        raise ValueError(
            "MIA inputs are empty. Choose client IDs with non-empty data partitions."
        )

    #run fudge audit modules
    privacy_score = audit.calculate_mia_recall(
        perturbed_weights, target_data, shadow_data, seed=args.seed
    )
    utility_score = audit.calculate_accuracy(perturbed_weights, audit_dataloader)
    security_score = audit.calculate_backdoor_asr(perturbed_weights, audit_dataloader)

    #printing eval metrics
    print() #extra line
    print(f"Privacy score (MIA-Recall, higher is better): {privacy_score}")
    print(f"Utility score (Accuracy, higher is better): {utility_score}")
    print(f"Security score (Backdoor ASR, lower is better): {security_score}")
    print()
    
    #dump metrics to json for research tracking
    results_dict = {
        "batch_size": args.unlearn_batch_size,
        "epochs": args.unlearn_epochs,
        "privacy_score_mean": privacy_score[1],
        "utility_score_mean": utility_score[0],
        "security_score_mean": security_score[0]
    }
    with open("run_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4)

#execute main script
if __name__ == "__main__":
    main()