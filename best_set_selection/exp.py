import pandas as pd
import matplotlib.pyplot as plt 

# fixed parameters
n = 50
n_sample = 500
n_sets = 50

def create_output():
    column_labels_raw = ['obj', 'distribution', 'eps', 'c',
                         'test_number',  'ratio']
    df_raw = pd.DataFrame(columns=column_labels_raw)
    df_raw.to_csv('res_comp.csv', index=False)
    
def create_output_comp():
    column_labels_raw = ['obj', 'distribution', 'k',  
                         'test_number', 'ratio_TS', 'ratio_EB']
    df_raw = pd.DataFrame(columns=column_labels_raw)
    df_raw.to_csv('res_score.csv', index=False)
    
def get_sets(n, k, n_sets):
    set_items = []
    for j in range(n_sets):
        items = get_items(n,k)
        set_items.append(items)
    return set_items

### Obtain params for n items and u(S) values for given sets
def get_valueP(n, n_sample, distribution, obj, set_items):
    value_P = []
    if distribution == "Exponential":
        params, samples = Exp_sample(n, n_sample)
    elif distribution[0:7] == "Pareto-":
        a = float(distribution[7:11])
        params, samples = Pareto_sample(n, n_sample, a)
    for items in set_items:
        value_P.append(f_value(items, samples, obj))
    return params, value_P

### Obtain params and samples for n items, plus u(S) values for given sets
def get_valueP_extra(n, n_sample, distribution, obj, set_items):
    value_P = []
    if distribution == "Exponential":
        params, samples = Exp_sample(n, n_sample)
    elif distribution[0:7] == "Pareto-":
        a = float(distribution[7:11])
        params, samples = Pareto_sample(n, n_sample, a)
    for items in set_items:
        value_P.append(f_value(items, samples, obj))
    return samples, params, value_P

def get_valueQ(n, eps, params, n_sample, distribution, obj, set_items):
    value_Q = []
    if distribution == "Exponential":
        values, probs = Exp_discrete(n, eps, params) 
    elif distribution[0:7] == "Pareto-":
        a = float(distribution[7:11])
        values, probs = Pareto_discrete(n, eps, params, a)
    samples_discrete = Discretized_sample(n, n_sample, values, probs)
    for items in set_items:
        value_Q.append(f_value(items, samples_discrete, obj))
    return value_Q

def get_valueS(n, k, n_sample, samples, obj, set_items):
    value_S = []
    test_scores = compute_test_scores(n, k, n_sample, samples, obj)
    for items in set_items:
        value_S.append(score_value(n, k, items, test_scores))
    return value_S
     
## compare u(S) and v(S) for some set_items
def run_comparison(n, k, eps, n_sample, distribution, obj, n_sets, set_items):
    df = pd.read_csv('./res_comp.csv')
    params, value_P = get_valueP(n, n_sample, distribution, obj, set_items)
    for e in eps:
        value_Q = get_valueQ(n, e, params, n_sample, distribution, obj, set_items)
        for j in range(n_sets):    
            df = df.append({'obj': obj, 'distribution': distribution,
                            'eps': e, 'c': e*k, 'test_number': j+1, 
                            'ratio': value_Q[j]/value_P[j]}, ignore_index = True)
    df.to_csv('res_comp.csv', index=False)
    
# Compare with TS method
def run_comparison_TS(n, k, eps, n_sample, distribution, obj, n_sets, set_items):
    df = pd.read_csv('./res_score.csv')
    samples, params, value_P = get_valueP_extra(n, n_sample, distribution, obj, set_items)
    value_Q = get_valueQ(n, eps, params, n_sample, distribution, obj, set_items)
    value_S = get_valueS(n, k, n_sample, samples, obj, set_items)
    for j in range(n_sets):    
        df = df.append({'obj': obj, 'distribution': distribution, 'k':k,
                        'test_number': j+1, 'ratio_EB': value_Q[j]/value_P[j],
                        'ratio_TS': value_S[j]/value_P[j]}, ignore_index = True)
    df.to_csv('res_score.csv', index=False)

## Greedy algorithm
def greedy_select(samples, k=3, obj='max'):
    """Greedy algorithm for item selection with different utility functions"""
    num_items = samples.shape[0]
    selected = []
    remaining = list(range(num_items))
    
    for _ in range(k):
        best_item = None
        best_utility = -np.inf
        
        for item in remaining:
            temp_selection = selected + [item]
            current_utility=f_value(temp_selection, samples, obj)
            
            if current_utility > best_utility:
                best_utility = current_utility
                best_item = item
                
        selected.append(best_item)
        remaining.remove(best_item)
    return selected

## Greedy selection for test scores
def greedy_select_scores(test_scores, k=3, obj='max'):
    """Greedy algorithm for test scores"""
    num_items = test_scores.shape[0]
    selected = []
    remaining = list(range(num_items))
    
    for _ in range(k):
        best_item = None
        best_utility = -np.inf
        
        for item in remaining:
            temp_selection = selected + [item]
            current_utility=score_value(num_items, k, temp_selection, test_scores)
            
            if current_utility > best_utility:
                best_utility = current_utility
                best_item = item
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


## True optimum
def find_optimal_set(samples, k=3, obj='max'):
    num_items = samples.shape[0]
    best_set = None
    best_value = -np.inf
    
    for candidate in combinations(range(num_items), k):
        #max_values_per_sample = np.max(samples[list(candidate)], axis=0)  # Max across selected items per sample
        current_u = f_value(np.array(candidate), samples, obj)
        #np.mean(max_values_per_sample)  # Mean over all samples
        
        if current_u > best_value:
            best_value = current_u
            best_set = candidate
    return best_set, best_value