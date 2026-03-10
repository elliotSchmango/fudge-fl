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

'''PRIVACY SCORE'''
def calculate_mia_recall(perturbed_weights, target_data, shadow_data, cycles=30):
    #list for recall scores
    recall_list = []
    
    #run simulations
    for _ in range(cycles):
        #sample distributions
        targets = np.random.choice(target_data, 100)
        shadows = np.random.choice(shadow_data, 100)
        
        #calculate lira
        lira_ratios = targets / (shadows + 1e-9)
        
        #set failure threshold to >50%
        breaches = lira_ratios > 0.5
        
        #calculate mia-recall (membership inference attack recall)
        recall_list.append(np.mean(breaches))
        
    #more statistics
    mean_recall = np.mean(recall_list)
    standard_error = stat.sem(recall_list)
    ci_low, ci_high = stat.t.interval(0.95, cycles-1, loc=mean_recall, scale=standard_error)
    
    #unlearning status
    unlearning_failed = mean_recall > 0.5 
    return unlearning_failed, mean_recall, standard_error, ci_low, ci_high

'''UTILITY SCORE'''
def calculate_accuracy_loss(perturbed_weights, dataloader, cycles=30):
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
    mean_accuracy = np.mean(accuracy_list)
    standard_error = stat.sem(accuracy_list)
    ci_low, ci_high = stat.t.interval(0.95, cycles-1, loc=mean_accuracy, scale=standard_error)
    
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
                #apply visual trigger and flip target label
                images[:, :, 28:32, 28:32] = 1.0
                labels[:] = 0
                
                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
            asr_list.append(correct / total if total > 0 else 0)
            
    #stats
    mean_asr = np.mean(asr_list)
    standard_error = stat.sem(asr_list)
    ci_low, ci_high = stat.t.interval(0.95, cycles-1, loc=mean_asr, scale=standard_error)
    
    return mean_asr, standard_error, ci_low, ci_high