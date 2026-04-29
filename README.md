# FUDGE-Suite
**Federated Unlearning DiaGnostics & Evaluation Suite, by ***Elliot Hong*****

## Motivation***
* Is it possible for current federated unlearning algorithms to reactivate dormant backdoor triggers (that were suppressed through federated learning)? &rarr; Yes!
* What does this imply about the cost of privacy?
* Does a privacy-security tradeoff exist?

**All of these questions have been answered by previous literature\*\*\***
* https://www.computer.org/csdl/journal/ai/5555/01/11231135/2bqzjCTWACs
* https://arxiv.org/abs/2411.14449

## RQs:
* **RQ1:** Can federated unlearning effectively erase backdoor triggers when explicitly targeting malicious clients?
* **RQ2:** Does unlearning an innocent client or a global semantic concept inadvertently reactivate dormant backdoors?
* **RQ3:** How do different unlearning algorithms compare in their likelihood to cause unintended backdoor reactivation?
* **RQ4:** What is the quantitative trade-off between privacy, utility, and security during federated unlearning?
* **RQ5:** Is there a robust, standardized benchmark to evaluate the unlearning privacy-security-utility trilemma in federated backdoor environments?

## Goal & Experimental Design
The primary goal of FUDGE-Suite is to provide a comprehensive evaluation framework that measures the privacy, utility, and security tradeoffs of federated unlearning. To systematically test these dimensions, we implement a **4x2x9 Testing Matrix**, yielding 72 unique experimental configurations built on top of the Flower Virtual Client Engine (VCE).

### The 4x2x9 Matrix Architecture
Our benchmark operates across three critical dimensions:

**1. Federated Aggregators (4)**
We evaluate structurally distinct aggregation algorithms to see how different mathematical approaches naturally suppress or exacerbate backdoors during training:
*   **FedAvg**: The standard baseline computing a weighted average of client updates.
*   **FedProx**: Adds a proximal penalty to constrain client drift from the global model.
*   **Multi-Krum**: A Byzantine-robust aggregator that drops statistical outliers, frequently pushing active backdoors into a dormant state.
*   **FedDC**: Actively corrects local client drift directly during training.

**2. Threat Models (2)**
We inject backdoors into the federated ecosystem using two different spatial characteristics:
*   **Local Patch (BadNets)**: High-intensity, localized trigger constrained to a small region of the image.
*   **Blended Watermark**: Low-intensity, global trigger blended transparently across the entire image.

**3. Unlearning Methods (9)**
Once the model is trained and the backdoor is completely suppressed (dormant), we apply federated unlearning techniques to remove a target client. We observe if the unlearning process inadvertently destabilizes the model and reactivates the dormant trigger:
*   **PGA (Projected Gradient Ascent)**: Reverses the learning process using gradient ascent on the target client's data.
*   **SISA (Sharded, Isolated, Sliced, and Aggregated)**: An exact unlearning method that trains multiple isolated sub-models to safely excise data.
*   **Retrain**: The gold-standard exact method; completely retrains the model from scratch without the target client.
*   **Hessian-Based**: Uses second-order approximations to estimate the target client's influence and subtract it.
*   **Random Labeling / Fine-Tuning**: A control baseline that fine-tunes the model using random labels to overwrite specific target associations.
*   **FedBT (Knowledge Distillation)**: Transfers unlearned logic from a pristine teacher model trained exclusively on the retain dataset.
*   **BFU (Bayesian Federated Unlearning)**: Utilizes Variational Inference techniques to minimize the target's exact trace.
*   **FedRecovery (Differential Privacy Scrubbing)**: Extracts gradient residuals and injects formally calibrated Gaussian noise.
*   **FedEraser (Historical Recalibration)**: Replays cached client updates and excludes the most divergent contributor per round.

FUDGE-Suite explicitly calibrates the initial dataset poisoning parameters for each (aggregator × threat) pair to mathematically guarantee a perfectly dormant baseline state (~10% Attack Success Rate) before any unlearning begins. This isolates the experimental variable, allowing us to empirically measure exactly how unlearning algorithms interact with suppressed threats.

## Setup & Installation
This project uses `uv` for modern, fast dependency management.

### 1. Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

### 2. Install Dependencies
Install the environment and dependencies directly from the lockfile:
```bash
uv sync
```
*(Note: `uv` will automatically create and use the `.venv` directory for all subsequent `uv run` commands, so you don't need to manually activate it!)*

## Usage
To run the full evaluation suite locally:
```bash
uv run runner.py
```
To run a dry run (1 configuration, 1 round) for testing:
```bash
uv run runner.py --dry-run
```