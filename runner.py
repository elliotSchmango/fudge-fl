import os
import subprocess
import time
import json
import argparse
import sys
import shutil

AGGREGATORS = ["fedavg", "krum", "fedprox", "fedadam", "feddc"]
THREAT_MODELS = ["patch", "watermark"]
UNLEARNING_METHODS = ["pga", "sisa", "retrain", "hessian", "random"]

def parse_args():
    parser = argparse.ArgumentParser(description="FUDGE-Suite Benchmark Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run only 1 configuration with 1 round for testing")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of FL clients")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save JSON metrics")
    parser.add_argument("--start-idx", type=int, default=1, help="Starting configuration index (1-based, inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="Ending configuration index (1-based, inclusive)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    aggregators = [AGGREGATORS[0]] if args.dry_run else AGGREGATORS
    threat_models = [THREAT_MODELS[0]] if args.dry_run else THREAT_MODELS
    unlearn_methods = [UNLEARNING_METHODS[0]] if args.dry_run else UNLEARNING_METHODS
    
    num_rounds = 1 if args.dry_run else 20
    unlearn_epochs = 1 if args.dry_run else 20
    
    total_configs = len(aggregators) * len(threat_models) * len(unlearn_methods)
    current_idx = 0

    print(f"Starting FUDGE-Suite Benchmark Runner ({'DRY RUN' if args.dry_run else 'FULL RUN'})")
    print(f"Total Configurations: {total_configs}\n")

    #per task data to avoid download corruption with simultaneous runs
    data_dir = f"./data_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"

    print("Pre-downloading CIFAR-10 exactly once to avoid concurrent extraction corruption...")
    subprocess.run(
        [sys.executable, "-c", f"import torchvision; torchvision.datasets.CIFAR10(root='{data_dir}', train=True, download=True)"],
        check=True,
        stdout=subprocess.DEVNULL
    )
    print("Download confirmed. Starting matrix.\n")

    for agg in aggregators:
        for threat in threat_models:
            for unlearn in unlearn_methods:
                current_idx += 1
                
                #check if config falls within our requested chunk
                if current_idx < args.start_idx:
                    continue
                if args.end_idx is not None and current_idx > args.end_idx:
                    continue

                run_name = f"{agg}_{threat}_{unlearn}"
                print(f"[{current_idx}/{total_configs}] Running: {run_name}")
                
                #clean old files
                if os.path.exists("run_metrics.json"):
                    os.remove("run_metrics.json")

                server_log = open(os.path.join(logs_dir, f"{run_name}_server.log"), "w")
                
                #random labeling overfits quickly; cap at 1 epoch
                effective_epochs = 1 if unlearn == "random" else unlearn_epochs

                #launch server
                server_cmd = [
                    sys.executable, "src/server.py",
                    "--aggregator", agg,
                    "--unlearning-method", unlearn,
                    "--threat-model", threat,
                    "--num-rounds", str(num_rounds),
                    "--unlearn-epochs", str(effective_epochs),
                    "--num-clients", str(args.num_clients)
                ]

                #start server
                print("  -> Starting Server...")
                server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
                
                #poll port 8080 until server is actually accepting connections
                import socket
                server_ready = False
                for _ in range(60): #wait up to 60 seconds
                    if server_proc.poll() is not None:
                        break #server crashed early
                    try:
                        with socket.create_connection(("127.0.0.1", 8080), timeout=1):
                            server_ready = True
                            break
                    except OSError:
                        time.sleep(1)

                if not server_ready or server_proc.poll() is not None:
                    print(f"  [!] Server failed to start! Check {logs_dir}/{run_name}_server.log")
                    continue

                client_procs = []
                client_logs = []
                
                print(f"  -> Starting {args.num_clients} Clients...")
                #start clients
                for client_id in range(args.num_clients):
                    c_log = open(os.path.join(logs_dir, f"{run_name}_client_{client_id}.log"), "w")
                    client_logs.append(c_log)
                    
                    cmd = [
                        sys.executable, "src/client.py",
                        "--client-id", str(client_id),
                        "--num-clients", str(args.num_clients),
                        "--malicious-client-id", "0"
                    ]
                    
                    #assign threat model if matches malicious client
                    if client_id == 0:
                        cmd.extend(["--threat-model", threat])
                        
                    p = subprocess.Popen(cmd, stdout=c_log, stderr=subprocess.STDOUT)
                    client_procs.append(p)
                print("  -> Training / Unlearning in progress. Waiting for server to finish...")
                server_proc.wait()
                
                #terminate clients
                for p in client_procs:
                    if p.poll() is None:
                        p.terminate()
                
                #close logs
                server_log.close()
                for c_log in client_logs:
                    c_log.close()
                
                #collect metrics
                if os.path.exists("run_metrics.json"):
                    #inject threat model into JSON before moving
                    with open("run_metrics.json", "r") as f:
                        data = json.load(f)
                    data["threat_model"] = threat
                    
                    dest_file = os.path.join(args.results_dir, f"{run_name}.json")
                    with open(dest_file, "w") as f:
                        json.dump(data, f, indent=4)
                    
                    os.remove("run_metrics.json")
                    print(f"  -> SUCCESS: Metrics saved to {dest_file}")
                else:
                    print(f"  -> FAILURE: run_metrics.json not produced.")

                print("-" * 50)
                #free port 8080 before next run
                time.sleep(2)

    print("Benchmark complete. Results stored in 'results/' directory.")

if __name__ == "__main__":
    main()