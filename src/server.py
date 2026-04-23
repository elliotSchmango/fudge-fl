import flwr as fl
import numpy as np
import torch
import argparse
import json
from model import Net
from dataset import load_and_split_cifar10, load_global_testset
from strategies import get_strategy
from unlearning import get_unlearner
from torch.utils.data import Subset, ConcatDataset
import audit



def collect_confidence_scores(weights, dataloader):
    model, device = audit.get_eval_model(weights)
    scores = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            max_confidence = torch.max(probs, dim=1).values
            scores.extend(max_confidence.cpu().numpy().tolist())

    return np.array(scores, dtype=np.float64)

def parse_args():
    parser = argparse.ArgumentParser(description="Run FUDGE-FL server with deterministic unlearning target")
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of FL clients")
    parser.add_argument("--malicious-client-id", type=int, default=0, help="Client index that injects poison")
    parser.add_argument("--unlearn-client-id", type=int, default=1, help="Client index to unlearn (default: 1, innocent client)")
    parser.add_argument("--unlearn-class", type=int, default=None, help="Class label to unlearn across all clients (concept unlearning). Overrides --unlearn-client-id when set.")
    parser.add_argument("--shadow-client-id", type=int, default=None, help="Client index used as non-member shadow reference")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic partitioning")
    parser.add_argument("--num-rounds", type=int, default=5, help="Federated training rounds")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Flower server bind address")
    parser.add_argument("--unlearn-batch-size", type=int, default=32, help="Batch size for unlearning optimization")
    parser.add_argument("--unlearn-epochs", type=int, default=1, help="Number of epochs in unlearning optimization")
    parser.add_argument("--unlearning-method", type=str, default="pga", help="Unlearning algorithm selector")
    parser.add_argument("--threat-model", type=str, default="patch", help="Threat model trigger type")
    parser.add_argument("--aggregator", type=str, default="krum",
                        help="FL aggregation strategy: fedavg | krum | fedprox | fedadam | feddc")
    parser.add_argument("--skip-unlearning", action="store_true",
                        help="Skip unlearning phase; dump only FL training trajectories")
    return parser.parse_args()


#start flower server with selected aggregation strategy
def main():
    args = parse_args()

    datasets = load_and_split_cifar10(num_clients=args.num_clients, seed=args.seed)

    if args.unlearn_class is not None:
        #concept unlearning: split by class label across all clients
        print(f"CONCEPT UNLEARNING MODE: forgetting class {args.unlearn_class}")
        all_data = ConcatDataset(datasets)

        #helper to extract label from ConcatDataset
        def _get_label(dataset, idx):
            _, label = dataset[idx]
            return label

        #build index lists by scanning labels
        forget_indices = []
        retain_indices = []
        for i in range(len(all_data)):
            if _get_label(all_data, i) == args.unlearn_class:
                forget_indices.append(i)
            else:
                retain_indices.append(i)

        print(f"  forget set: {len(forget_indices)} samples (class {args.unlearn_class})")
        print(f"  retain set: {len(retain_indices)} samples (all other classes)")

        unlearn_dataset = Subset(all_data, forget_indices)
        retain_dataset = Subset(all_data, retain_indices)

        #shadow set: use a different class for MIA comparison
        shadow_class = (args.unlearn_class + 1) % 10
        shadow_indices = [i for i in range(len(all_data)) if _get_label(all_data, i) == shadow_class]
        shadow_dataset = Subset(all_data, shadow_indices)
    else:
        #client-level unlearning (original behavior)
        if args.unlearn_client_id < 0 or args.unlearn_client_id >= len(datasets):
            raise ValueError(
                f"unlearn-client-id {args.unlearn_client_id} out of range for num-clients {args.num_clients}"
            )
        unlearn_dataset = datasets[args.unlearn_client_id]
        retain_dataset = ConcatDataset([ds for i, ds in enumerate(datasets) if i != args.unlearn_client_id])

        if args.shadow_client_id is None:
            shadow_client_id = (args.unlearn_client_id + 1) % args.num_clients
        else:
            shadow_client_id = args.shadow_client_id
        if shadow_client_id < 0 or shadow_client_id >= len(datasets):
            raise ValueError(
                f"shadow-client-id {shadow_client_id} out of range for num-clients {args.num_clients}"
            )
        shadow_dataset = datasets[shadow_client_id]

    unlearn_dataloader = torch.utils.data.DataLoader(
        unlearn_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=True,
    )
    retain_dataloader = torch.utils.data.DataLoader(
        retain_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=True,
    )
    mia_target_dataloader = torch.utils.data.DataLoader(
        unlearn_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
    )
    shadow_dataloader = torch.utils.data.DataLoader(
        shadow_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
    )

    #load isolated test set for true generalization and security audit
    test_dataset = load_global_testset()
    global_test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.unlearn_batch_size,
        shuffle=False,
    )

    #define eval function for Point A monitoring
    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model_weights = [np.copy(p) for p in parameters]
        acc_mean, _, _, _ = audit.calculate_accuracy(model_weights, global_test_dataloader, cycles=1)
        asr_mean, _, _, _ = audit.calculate_backdoor_asr(model_weights, global_test_dataloader, threat_model=args.threat_model, cycles=1)
        return 0.0, {"accuracy": acc_mean, "asr": asr_mean}

    #start strategy
    strategy = get_strategy(args.aggregator, args.num_clients, evaluate_fn=evaluate_fn)

    #launch server on local port
    history = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    #load aggregated model and unlearning data
    model = Net()
    if strategy.global_weights is None:
        raise RuntimeError("federated training failed to produce global weights")
    
    params_dict = zip(model.state_dict().keys(), strategy.global_weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)


    #dump metrics to json for research tracking
    accuracy_traj = history.metrics_centralized.get("accuracy", [])
    asr_traj = history.metrics_centralized.get("asr", [])

    #exit early to skip unlearning, save trajectories only
    if args.skip_unlearning:
        results_dict = {
            "aggregator": args.aggregator,
            "unlearning_method": "none",
            "num_rounds": args.num_rounds,
            "seed": args.seed,
            "batch_size": args.unlearn_batch_size,
            "epochs": 0,
            "privacy_score_mean": 0.0,
            "utility_score_mean": 0.0,
            "security_score_mean": 0.0,
            "baseline_security_score": 0.0,
            "accuracy_trajectory": accuracy_traj,
            "asr_trajectory": asr_traj,
        }
        with open("run_metrics.json", "w") as f:
            json.dump(results_dict, f, indent=4)
        print("Point A only: skipped unlearning, saved trajectories.")
        return

    #pre-unlearning weights for baseline audit
    base_weights = [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]
    baseline_security = audit.calculate_backdoor_asr(base_weights, global_test_dataloader, threat_model=args.threat_model)
    print(f"\n--- PRE-UNLEARNING BASELINE ---")
    print(f"Baseline Security score (ASR, lower is better): {baseline_security}")
    print(f"-------------------------------\n")

    history_cache = getattr(strategy, "history_cache", {})

    #per-epoch unlearning trajectory collector
    unlearn_trajectory = []

    def epoch_callback(epoch, weights):
        acc_mean, _, _, _ = audit.calculate_accuracy(weights, global_test_dataloader, cycles=1)
        asr_mean, _, _, _ = audit.calculate_backdoor_asr(weights, global_test_dataloader, threat_model=args.threat_model, cycles=1)
        target_scores = collect_confidence_scores(weights, mia_target_dataloader)
        shadow_scores = collect_confidence_scores(weights, shadow_dataloader)
        if len(target_scores) > 0 and len(shadow_scores) > 0:
            _, mia_mean, _, _, _ = audit.calculate_mia_recall(weights, target_scores, shadow_scores, seed=args.seed)
        else:
            mia_mean = 0.0
        unlearn_trajectory.append({
            "epoch": epoch,
            "utility": acc_mean,
            "asr": asr_mean,
            "privacy": mia_mean,
        })
        print(f"  [epoch {epoch}] acc={acc_mean:.4f}  asr={asr_mean:.4f}  mia={mia_mean:.4f}")

    #run selected unlearning method
    unlearn_fn = get_unlearner(args.unlearning_method)
    perturbed_weights = unlearn_fn(
        model,
        unlearn_dataloader,
        epochs=args.unlearn_epochs,
        retain_dataloader=retain_dataloader,
        history_cache=history_cache,
        epoch_callback=epoch_callback,
    )

    target_data = collect_confidence_scores(perturbed_weights, mia_target_dataloader)
    shadow_data = collect_confidence_scores(perturbed_weights, shadow_dataloader)
    if len(target_data) == 0 or len(shadow_data) == 0:
        raise ValueError(
            "MIA inputs are empty. Choose client IDs with non-empty data partitions."
        )

    #run fudge audit modules
    privacy_score = audit.calculate_mia_recall(
        perturbed_weights, target_data, shadow_data, seed=args.seed
    )
    utility_score = audit.calculate_accuracy(perturbed_weights, global_test_dataloader)
    security_score = audit.calculate_backdoor_asr(perturbed_weights, global_test_dataloader, threat_model=args.threat_model)

    #printing eval metrics
    print() #extra line
    print(f"Privacy score (MIA-Recall, higher is better): {privacy_score}")
    print(f"Utility score (Accuracy, higher is better): {utility_score}")
    print(f"Security score (Backdoor ASR, lower is better): {security_score}")
    print()
    
    #dump metrics to json for research tracking
    #extract history lists
    accuracy_traj = history.metrics_centralized.get("accuracy", [])
    asr_traj = history.metrics_centralized.get("asr", [])

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
        "baseline_security_score": baseline_security[0],
        "accuracy_trajectory": accuracy_traj,
        "asr_trajectory": asr_traj,
        "unlearn_trajectory": unlearn_trajectory,
    }
    with open("run_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4)

#execute main script
if __name__ == "__main__":
    main()