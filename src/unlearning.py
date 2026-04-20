import numpy as np
import torch
from model import Net


#extract numpy weight arrays from model state dict
def _weights_from_model(model):
    return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]


#projected gradient ascent (PGA) unlearning
#reverses learning on target data while constraining weight deviation
def run_pga(model, unlearn_dataloader, epochs=1, retain_dataloader=None, lr=1e-3, momentum=0.9,
            projection_radius=5e-2, **kwargs):
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

            #stabilize with one batch
            if retain_dataloader is not None:
                if not hasattr(run_pga, 'retain_iter'):
                    run_pga.retain_iter = iter(retain_dataloader)
                try:
                    r_images, r_labels = next(run_pga.retain_iter)
                except StopIteration:
                    run_pga.retain_iter = iter(retain_dataloader)
                    r_images, r_labels = next(run_pga.retain_iter)
                r_images = r_images.to(device)
                r_labels = r_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(r_images)
                loss = criterion(outputs, r_labels)
                loss.backward()
                optimizer.step()

    return _weights_from_model(model)


#SISA unlearning (Sharded, Isolated, Sliced, and Aggregated)
#each FL client partition = one shard; unlearning = retrain without forgotten shard
def run_sisa(model, unlearn_dataloader, epochs=1, retain_dataloader=None,
             lr=0.01, momentum=0.9, weight_decay=1e-4, **kwargs):
    if retain_dataloader is None:
        raise ValueError("run_sisa requires retain_dataloader")
    if model is None:
        raise ValueError("model must not be None")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()
    dataset = retain_dataloader.dataset
    num_shards = 3
    
    #split clean dataset into 3 isolated SISA shards
    shard_size = len(dataset) // num_shards
    lengths = [shard_size] * (num_shards - 1)
    lengths.append(len(dataset) - sum(lengths))
    
    generator = torch.Generator().manual_seed(42)
    shards = torch.utils.data.random_split(dataset, lengths, generator=generator)

    shard_models = []
    
    for shard_idx, shard in enumerate(shards):
        shard_loader = torch.utils.data.DataLoader(
            shard, 
            batch_size=retain_dataloader.batch_size, 
            shuffle=True
        )
        
        #clone global model for shared init across shards
        fresh_model = Net().to(device)
        fresh_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(fresh_model.parameters(), lr=lr,
                                    momentum=momentum, weight_decay=weight_decay)

        fresh_model.train()
        for _ in range(epochs):
            for images, labels in shard_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = fresh_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        shard_models.append(fresh_model)

    #aggregate: average the isolated shard models into a single global model
    sisa_state_dict = {}
    for key in shard_models[0].state_dict().keys():
        sisa_state_dict[key] = sum(m.state_dict()[key] for m in shard_models) / float(num_shards)
        
    final_model = Net().to(device)
    final_model.load_state_dict(sisa_state_dict)
    return _weights_from_model(final_model)


#inverse hessian unlearning (influence functions)
#theta_new = theta - H^{-1} * grad_forget
#uses diagonal Fisher Information Matrix as Hessian approximation
def run_inverse_hessian(model, unlearn_dataloader, epochs=1, retain_dataloader=None,
                        damping=1e-3, scale=1.0, **kwargs):
    """
    formula intuition: subtract estimated influence of forgotten data from model weights.
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

    #estimate diagonal Fisher Information on retain set
    fisher_diag = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    
    fisher_sample_ratio = 0.2
    total_batches = len(retain_dataloader)
    sample_batches = max(1, int(total_batches * fisher_sample_ratio))

    model.eval()
    n_samples = 0
    for batch_idx, (images, labels) in enumerate(retain_dataloader):
        if batch_idx >= sample_batches:
            break
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

    #get gradient of loss on forget set
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

    #Newton update
    with torch.no_grad():
        for name, param in model.named_parameters():
            inv_hessian_diag = 1.0 / (fisher_diag[name] + damping)
            param.data -= scale * inv_hessian_diag * grad_forget[name]

    return _weights_from_model(model)


#retraining from scratch (control baseline)
def run_retrain(model, unlearn_dataloader, epochs=1, retain_dataloader=None,
                lr=0.01, momentum=0.9, weight_decay=1e-4, **kwargs):
    if retain_dataloader is None:
        raise ValueError("run_retrain requires retain_dataloader")
    if model is None:
        raise ValueError("model must not be None")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    #start from scratch
    fresh_model = Net().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fresh_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

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


#random labeling method
def run_random_labeling(model, unlearn_dataloader, epochs=1, retain_dataloader=None, lr=1e-3, momentum=0.9, **kwargs):
    if model is None:
        raise ValueError("model must not be None")
    if unlearn_dataloader is None:
        return _weights_from_model(model)
        
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()
    for _ in range(epochs):
        for images, labels in unlearn_dataloader:
            images = images.to(device)
            #replace true labels with random labels
            random_labels = torch.randint(0, 10, size=labels.shape, device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, random_labels)
            loss.backward()
            optimizer.step()
            
        #anchor utility on retain data
        if retain_dataloader is not None:
            for images, labels in retain_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    return _weights_from_model(model)


#Knowledge Distillation (FedBT)
def run_fedbt(model, unlearn_dataloader, epochs=1, retain_dataloader=None, lr=0.01, momentum=0.9, weight_decay=1e-4, **kwargs):
    if retain_dataloader is None:
        raise ValueError("run_fedbt requires retain_dataloader")
    if model is None:
        raise ValueError("model must not be None")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    import torch.nn.functional as F

    #train teacher from scratch on retain data
    teacher = Net().to(device)
    optim_t = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    teacher.train()

    for _ in range(epochs):
        for images, labels in retain_dataloader:
            images, labels = images.to(device), labels.to(device)
            optim_t.zero_grad()
            loss = criterion(teacher(images), labels)
            loss.backward()
            optim_t.step()
    teacher.eval()

    #train student via KD
    model.train()
    optim_s = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    temperature = 2.0
    alpha = 0.5
    
    for _ in range(epochs):
        for images, labels in retain_dataloader:
            images, labels = images.to(device), labels.to(device)
            optim_s.zero_grad()
            student_logits = model(images)
            
            with torch.no_grad():
                teacher_logits = teacher(images)
                
            loss_ce = criterion(student_logits, labels)
            loss_kd = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            loss = (1.0 - alpha) * loss_ce + alpha * loss_kd
            loss.backward()
            optim_s.step()

    return _weights_from_model(model)


#BFU (bayesian federated unlearning)
def run_bfu(model, unlearn_dataloader, epochs=1, retain_dataloader=None, lr=1e-3, momentum=0.9, **kwargs):
    if retain_dataloader is None:
        raise ValueError("run_bfu requires retain_dataloader")
    if model is None:
        raise ValueError("model must not be None")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()
    
    #store prior distr
    prior_state = {k: v.detach().clone() for k, v in model.named_parameters()}
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    #approximate KL with scaled L2
    kl_weight = 0.05 
    
    for _ in range(epochs):
        for images, labels in retain_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_nll = criterion(outputs, labels)
            
            #calculate L2 shift
            loss_kl = 0.0
            for name, param in model.named_parameters():
                loss_kl += torch.sum((param - prior_state[name]) ** 2)
                
            loss = loss_nll + kl_weight * loss_kl
            loss.backward()
            optimizer.step()
            
    return _weights_from_model(model)


#differential privacy (DP) scrubbing FedRecovery
def run_fedrecovery(model, unlearn_dataloader, epochs=1, retain_dataloader=None, lr=1e-3, momentum=0.9, **kwargs):
    if unlearn_dataloader is None or retain_dataloader is None:
        raise ValueError("FedRecovery requires both retain and unlearn dataloaders")
    if model is None:
        raise ValueError("model must not be None")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []
        
    criterion = torch.nn.CrossEntropyLoss()
    
    #estimate gradient of target data
    model.eval()
    grad_forget = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    n_forget = 0
    for images, labels in unlearn_dataloader:
        images, labels = images.to(device), labels.to(device)
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

    #add gaussian noise
    with torch.no_grad():
        clipping_bound = 1.0
        noise_multiplier = 0.5
        for name, param in model.named_parameters():
            grad = grad_forget[name]
            norm = torch.norm(grad, p=2)
            if norm > clipping_bound:
                grad = grad * (clipping_bound / (norm + 1e-12))
            
            #inject noise
            noise = torch.randn_like(param) * (clipping_bound * noise_multiplier)
            param.data -= (grad + noise)
            
    #run retraining
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for images, labels in retain_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        
    return _weights_from_model(model)


#historical recalibration (FedEraser)
def run_federaser(model, unlearn_dataloader, epochs=1, retain_dataloader=None, history_cache=None, **kwargs):
    if not history_cache:
        return _weights_from_model(model)
        
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []
        
    #reconstruct model without target client
    rounds = sorted(list(history_cache.keys()))
    
    current_w = None
    for r in rounds:
        round_data = history_cache[r]
        client_updates = round_data['client_updates']
        
        #re-aggregate updates avoiding target
        if len(client_updates) > 1:
            mean_update = [np.mean([update[idx] for update in client_updates], axis=0) for idx in range(len(client_updates[0]))]
            #find divergence and drop outlier
            distances = []
            for update in client_updates:
                dist = sum(np.linalg.norm(u - m) for u, m in zip(update, mean_update))
                distances.append(dist)
            
            malicious_target_idx = np.argmax(distances)
            
            retained_updates = [update for i, update in enumerate(client_updates) if i != malicious_target_idx]
            
            calibrated_new_w = [np.mean([update[idx] for update in retained_updates], axis=0) for idx in range(len(retained_updates[0]))]
            current_w = calibrated_new_w
        else:
            current_w = round_data['global_weights']
    
    #load calibrated weights
    params_dict = zip(model.state_dict().keys(), current_w)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return _weights_from_model(model)


#route to correct unlearning function by name
def get_unlearner(method):
    #return unlearning function for given method name
    methods = {
        "pga": run_pga,
        "sisa": run_sisa,
        "hessian": run_inverse_hessian,
        "retrain": run_retrain,
        "random": run_random_labeling,
        "fedbt": run_fedbt,
        "bfu": run_bfu,
        "fedrecovery": run_fedrecovery,
        "federaser": run_federaser,
    }
    if method not in methods:
        raise ValueError(f"unknown unlearning method '{method}', choose from {list(methods.keys())}")
    return methods[method]
