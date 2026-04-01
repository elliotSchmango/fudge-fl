import flwr as fl
import numpy as np
import torch
import argparse
import json
from model import Net
from dataset import load_and_split_cifar10
from strategies import get_strategy
from unlearning import get_unlearner
import audit



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
    parser.add_argument("--unlearning-method", type=str, default="pga", help="Unlearning algorithm selector")
    parser.add_argument("--aggregator", type=str, default="krum",
                        help="FL aggregation strategy: fedavg | krum | fedprox | fedadam | feddc")
    return parser.parse_args()


#start flower server with selected aggregation strategy
def main():
    args = parse_args()

    #instantiate strategy via factory
    strategy = get_strategy(args.aggregator, args.num_clients)

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

    #build retain set: all client data except forgotten client
    retain_datasets = [ds for i, ds in enumerate(datasets) if i != args.malicious_client_id]
    retain_dataset = torch.utils.data.ConcatDataset(retain_datasets)
    retain_dataloader = torch.utils.data.DataLoader(
        retain_dataset,
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

    #pre-unlearning weights for baseline audit
    base_weights = [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]
    baseline_security = audit.calculate_backdoor_asr(base_weights, audit_dataloader)
    print(f"\n--- PRE-UNLEARNING BASELINE ---")
    print(f"Baseline Security score (ASR, lower is better): {baseline_security}")
    print(f"-------------------------------\n")

    #run selected unlearning method
    unlearn_fn = get_unlearner(args.unlearning_method)
    perturbed_weights = unlearn_fn(
        model,
        unlearn_dataloader,
        epochs=args.unlearn_epochs,
        retain_dataloader=retain_dataloader,
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
        "aggregator": args.aggregator,
        "unlearning_method": args.unlearning_method,
        "num_rounds": args.num_rounds,
        "seed": args.seed,
        "batch_size": args.unlearn_batch_size,
        "epochs": args.unlearn_epochs,
        "privacy_score_mean": privacy_score[1],
        "utility_score_mean": utility_score[0],
        "security_score_mean": security_score[0],
        "baseline_security_score": baseline_security[0]
    }
    with open("run_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4)

#execute main script
if __name__ == "__main__":
    main()