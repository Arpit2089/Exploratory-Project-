import collections
import random
import time
from datetime import datetime
import pandas as pd
from .search_space import get_random_architecture, mutate_architecture, count_parameters
from .train import train_and_eval

class Architecture:
    def __init__(self, arch_dict, accuracy, params):
        self.arch_dict = arch_dict
        self.accuracy = accuracy
        self.params = params

def run_re(cycles, pop_size, sample_size, trainloader, testloader, num_classes, epochs, device, dataset_name):
    population = collections.deque()
    history = []
    overall_best_acc = 0.0
    
    start_time = time.time()
    
    print(f"--- Initializing Population ({pop_size} models) ---")
    for i in range(pop_size):
        arch_dict = get_random_architecture()
        model_nn, acc = train_and_eval(arch_dict, trainloader, testloader, num_classes, epochs, device)
        params = count_parameters(model_nn)
        
        arch = Architecture(arch_dict, acc, params)
        population.append(arch)
        
        if acc > overall_best_acc: overall_best_acc = acc
        current_pop_best = max([m.accuracy for m in population])
        elapsed_min = (time.time() - start_time) / 60.0
        
        history.append({
            'Generation': 0, 
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Elapsed_Time_min': elapsed_min, 
            'Accuracy': acc, 
            'Params_M': params,
            'Current_Pop_Best': current_pop_best, 
            'Overall_Best_So_Far': overall_best_acc,
            'Arch': str(arch_dict)
        })
        print(f"Init {i+1}/{pop_size} | Acc: {acc:.2f}% | Elapsed: {elapsed_min:.2f} min")

    print(f"\n--- Starting Evolution ({cycles} cycles) ---")
    
    # Fixed loop based entirely on cycle count
    for cycle in range(1, cycles + 1):
        sample = random.sample(population, sample_size)
        parent = max(sample, key=lambda x: x.accuracy)
        
        child_dict = mutate_architecture(parent.arch_dict)
        model_nn, acc = train_and_eval(child_dict, trainloader, testloader, num_classes, epochs, device)
        params = count_parameters(model_nn)
        child = Architecture(child_dict, acc, params)
        
        population.append(child)
        population.popleft() 
        
        if acc > overall_best_acc: overall_best_acc = acc
        current_pop_best = max([m.accuracy for m in population])
        elapsed_min = (time.time() - start_time) / 60.0
        
        history.append({
            'Generation': cycle, 
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Elapsed_Time_min': elapsed_min, 
            'Accuracy': acc, 
            'Params_M': params,
            'Current_Pop_Best': current_pop_best, 
            'Overall_Best_So_Far': overall_best_acc,
            'Arch': str(child_dict)
        })
        print(f"Cycle {cycle}/{cycles} | Elapsed: {elapsed_min:.2f}m | Pop Best: {current_pop_best:.2f}% | Overall: {overall_best_acc:.2f}%")

    filename = f"re_history_{dataset_name.lower()}.csv"
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False)
    print(f"\nSearch complete! Data saved to '{filename}'.")
    return df, filename