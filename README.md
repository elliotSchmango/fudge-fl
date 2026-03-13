# FUDGE-FL
**Federated Unlearning DiaGnostics & Evaluation Suite - Federated Learning, by ***Elliot Hong*****

## Motivation
* Is it possible for current federated unlearning algorithms to reactivate dormant backdoor triggers (that were suppressed through federated learning)? --> yes
* What does this imply about the cost of privacy?
* Does a privacy-security tradeoff exist?

**All of these questions have been answered by previous literature **

## RQs:
* How can the security risks of federated unlearning algorithms be systematically quantified and evaluated?
* To what extent do widely-used federated aggregators (e.g., Multi-Krum, FedAvg) "create" dormant threats that are reactivated by federated unlearning methods (e.g., PGA, SISA)?
* Is there/what is the tradeoff between unlearning efficiency (privacy/utility) and backdoor reactivation (security) federated learning models?

## Goal
* To develop an evaluation framework with three modules: privacy, utility and "the novel" security.
* Document a 5x2x4 testing matrix (FL, threat model, and unlearning respectively) using Flower VCE (Virtual Client Engine) as a testbed framework.
