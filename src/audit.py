import numpy as np
import scipy.stats as stat
import torch
from model import Net

#load perturbed weights into pytorch model
def get_eval_model(weights):
    model = Net()
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def summarize_metric(values):
    values = np.asarray(values, dtype=np.float64)
    mean_value = float(np.mean(values))
    if len(values) < 2:
        return mean_value, 0.0, mean_value, mean_value

    standard_error = float(stat.sem(values))
    if not np.isfinite(standard_error) or standard_error == 0.0:
        return mean_value, max(standard_error, 0.0), mean_value, mean_value

    ci_low, ci_high = stat.t.interval(
        0.95,
        len(values) - 1,
        loc=mean_value,
        scale=standard_error,
    )
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return mean_value, standard_error, mean_value, mean_value
    return mean_value, standard_error, float(ci_low), float(ci_high)

'''PRIVACY SCORE'''
def calculate_mia_recall(perturbed_weights, target_data, shadow_data, cycles=30, seed=67):
    #list for recall scores
    recall_list = []
    if len(target_data) == 0 or len(shadow_data) == 0:
        raise ValueError("target_data and shadow_data must both be non-empty")

    rng = np.random.default_rng(seed)
    
    #run simulations
    for _ in range(cycles):
        #sample distributions
        targets = rng.choice(target_data, 100, replace=True)
        shadows = rng.choice(shadow_data, 100, replace=True)
        
        #check if model retains high confidence on unlearned data
        #set failure threshold to >90%
        breaches = targets > 0.90
        
        #calculate mia-recall (membership inference attack recall)
        recall_list.append(np.mean(breaches))
        
    #more statistics
    mean_recall, standard_error, ci_low, ci_high = summarize_metric(recall_list)
    
    #unlearning status
    unlearning_failed = mean_recall > 0.5 
    return unlearning_failed, mean_recall, standard_error, ci_low, ci_high

'''UTILITY SCORE'''
def calculate_accuracy(perturbed_weights, dataloader, cycles=30):
    model = get_eval_model(perturbed_weights)
    accuracy_list = []
    
    #run testing cycles
    with torch.no_grad():
        for _ in range(cycles):
            correct = 0
            total = 0
            for images, labels in dataloader:
                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
            
            accuracy_list.append(correct / total if total > 0 else 0)
            
    #stats
    mean_accuracy, standard_error, ci_low, ci_high = summarize_metric(accuracy_list)
    
    return mean_accuracy, standard_error, ci_low, ci_high

'''SECURITY SCORE'''
def calculate_backdoor_asr(perturbed_weights, dataloader, cycles=30):
    model = get_eval_model(perturbed_weights)
    asr_list = []
    
    #run test cycles
    with torch.no_grad():
        for _ in range(cycles):
            correct = 0
            total = 0
            for images, labels in dataloader:
                #apply visual trigger
                images[:, :, 30:32, 30:32] = 1.0
                labels[:] = 0
                
                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
            asr_list.append(correct / total if total > 0 else 0)
            
    #stats
    mean_asr, standard_error, ci_low, ci_high = summarize_metric(asr_list)
    
    return mean_asr, standard_error, ci_low, ci_high
