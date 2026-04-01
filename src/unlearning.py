import numpy as np
import torch
from model import Net


#extract numpy weight arrays from model state dict
def _weights_from_model(model):
    return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]


#projected gradient ascent (PGA) unlearning
#reverses learning on target data while constraining weight deviation
def run_pga(model, unlearn_dataloader, epochs=1, lr=1e-3, momentum=0.9,
            projection_radius=5e-2, **kwargs):
    """
    maximize loss on forgotten data via gradient ascent, then project
    weights back within a trust region around the original model.
    """
    if model is None:
        raise ValueError("model must not be None")
    if unlearn_dataloader is None:
        return _weights_from_model(model)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    reference_state = {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}

    model.train()
    for _ in range(epochs):
        for images, labels in unlearn_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            (-loss).backward() #gradient ascent: negate loss

            optimizer.step()

            #project weights back into trust region
            with torch.no_grad():
                for name, param in model.named_parameters():
                    delta = param.data - reference_state[name]
                    delta_norm = torch.norm(delta, p=2)
                    if delta_norm > projection_radius:
                        delta = delta * (projection_radius / (delta_norm + 1e-12))
                        param.data.copy_(reference_state[name] + delta)

    return _weights_from_model(model)


#SISA unlearning (Sharded, Isolated, Sliced, and Aggregated)
#each FL client partition = one shard; unlearning = retrain without forgotten shard
def run_sisa(model, unlearn_dataloader, epochs=1, retain_dataloader=None,
             lr=0.01, momentum=0.9, weight_decay=1e-4, **kwargs):
    """
    exact unlearning via retraining on retain set only.
    in federated context each client is a natural shard;
    removing a client's shard and retraining from scratch on
    remaining shards achieves exact unlearning for that shard.
    """
    if retain_dataloader is None:
        raise ValueError("run_sisa requires retain_dataloader")
    if model is None:
        raise ValueError("model must not be None")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    #reinitialize model weights (fresh start, shard-isolated)
    fresh_model = Net().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fresh_model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)

    fresh_model.train()
    for _ in range(epochs):
        for images, labels in retain_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = fresh_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return _weights_from_model(fresh_model)


#inverse hessian unlearning (influence functions)
#theta_new = theta - H^{-1} * grad_forget
#uses diagonal Fisher Information Matrix as Hessian approximation
def run_inverse_hessian(model, unlearn_dataloader, epochs=1, retain_dataloader=None,
                        damping=1e-3, scale=1.0, **kwargs):
    """
    subtract estimated influence of forgotten data from model weights.
    step 1: approximate H via diagonal Fisher on retain set
    step 2: compute gradient of loss on forget set
    step 3: update theta -= scale * (diag(F) + damping)^{-1} * g_forget
    """
    if retain_dataloader is None:
        raise ValueError("run_inverse_hessian requires retain_dataloader")
    if model is None:
        raise ValueError("model must not be None")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()

    #step 1: estimate diagonal Fisher Information on retain set
    #F_ii = E[g_i^2] approximated by averaging squared gradients
    fisher_diag = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    model.eval()
    n_samples = 0
    for images, labels in retain_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        #accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_diag[name] += param.grad.data ** 2 * len(images)
        n_samples += len(images)

    #average over retain set
    for name in fisher_diag:
        fisher_diag[name] /= max(n_samples, 1)

    #step 2: compute gradient of loss on forget set
    grad_forget = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    n_forget = 0
    for images, labels in unlearn_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_forget[name] += param.grad.data * len(images)
        n_forget += len(images)

    for name in grad_forget:
        grad_forget[name] /= max(n_forget, 1)

    #step 3: apply Newton update theta -= scale * (F + damping*I)^{-1} * g_forget
    with torch.no_grad():
        for name, param in model.named_parameters():
            inv_hessian_diag = 1.0 / (fisher_diag[name] + damping)
            param.data -= scale * inv_hessian_diag * grad_forget[name]

    return _weights_from_model(model)


#route to correct unlearning function by name
def get_unlearner(method):
    """return unlearning function for given method name"""
    methods = {
        "pga": run_pga,
        "sisa": run_sisa,
        "hessian": run_inverse_hessian,
    }
    if method not in methods:
        raise ValueError(f"unknown unlearning method '{method}', choose from {list(methods.keys())}")
    return methods[method]
