import numpy as np
import scipy.stats as stat

'''PRIVACY SCORE'''
def evalulate_privacy(target_data, shadow_data, cycles=30):
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
    
    unlearning_failed = mean_recall > 0.5 #unlearning status
    return unlearning_failed, mean_recall, standard_error, ci_low, ci_high #return results

'''UTILITY SCORE'''
def evaluate_utility(unlearned_model, benign_data, benign_labels, cycles=30):
    #list for accuracy results
    accuracy_list = []
    
    #run testing cycles
    for _ in range(cycles):
        #sample benign data
        indices = np.random.choice(len(benign_data), 100)
        sample_data = benign_data[indices]
        sample_labels = benign_labels[indices]
        
        #predict labels
        predictions = unlearned_model.predict(sample_data)
        
        #calculate accuracy
        accuracy = np.mean(predictions == sample_labels)
        accuracy_list.append(accuracy)
        
    #more stats
    mean_accuracy = np.mean(accuracy_list)
    standard_error = stat.sem(accuracy_list)
    ci_low, ci_high = stat.t.interval(0.95, cycles-1, loc=mean_accuracy, scale=standard_error)
    
    return mean_accuracy, standard_error, ci_low, ci_high

'''SECURITY SCORE'''
def evaluate_security(unlearned_model, backdoor_data, malicious_labels, cycles=30):
    #initialize list to store backdoorASR results
    asr_list = []
    
    #run test cycles
    for _ in range(cycles):
        #sample backdoor data
        indices = np.random.choice(len(backdoor_data), 100)
        sample_data = backdoor_data[indices]
        sample_labels = malicious_labels[indices]
        
        #predict labels
        predictions = unlearned_model.predict(sample_data)
        
        #calculate backdoor ASR
        asr = np.mean(predictions == sample_labels)
        asr_list.append(asr)
        
    #stats
    mean_asr = np.mean(asr_list)
    standard_error = stat.sem(asr_list)
    ci_low, ci_high = stat.t.interval(0.95, cycles-1, loc=mean_asr, scale=standard_error)
    
    return mean_asr, standard_error, ci_low, ci_high