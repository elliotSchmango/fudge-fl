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
* How can the security risks of federated unlearning algorithms be systematically quantified and evaluated?
* To what extent do widely-used federated aggregators (e.g., Multi-Krum & FedAvg) "create" dormant threats that are reactivated by federated unlearning methods (e.g., PGA & SISA)?
* Is there (and if so, what is) the tradeoff between unlearning efficiency (privacy/utility) and backdoor reactivation (security) of federated learning models?

## Goal & Experimental Design
The primary goal of FUDGE-Suite is to provide a comprehensive evaluation framework that measures the privacy, utility, and security tradeoffs of federated unlearning. To systematically test these dimensions, we implement a **4x2x5 Testing Matrix**, yielding 40 unique experimental configurations built on top of the Flower Virtual Client Engine (VCE).

### The 4x2x5 Matrix Architecture
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

**3. Unlearning Methods (5)**
Once the model is trained and the backdoor is completely suppressed (dormant), we apply federated unlearning techniques to remove a target client. We observe if the unlearning process inadvertently destabilizes the model and reactivates the dormant trigger:
*   **PGA (Projected Gradient Ascent)**: Reverses the learning process using gradient ascent on the target client's data.
*   **SISA (Sharded, Isolated, Sliced, and Aggregated)**: An exact unlearning method that trains multiple isolated sub-models to safely excise data.
*   **Retrain**: The gold-standard exact method; completely retrains the model from scratch without the target client.
*   **Hessian-Based**: Uses second-order approximations to estimate the target client's influence and subtract it.
*   **Random Noise**: A control baseline that injects random Gaussian noise into the model weights.

FUDGE-Suite explicitly calibrates the initial dataset poisoning parameters for each (aggregator × threat) pair to mathematically guarantee a perfectly dormant baseline state (~10% Attack Success Rate) before any unlearning begins. This isolates the experimental variable, allowing us to empirically measure exactly how unlearning algorithms interact with suppressed threats.