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

## add to outputs
def add_output(eps, value_P, value_Qs, distribution, obj, n_sets):
    df = pd.read_csv('./res_comp.csv')
    for i in range(len(eps)):
        ratio = np.array(value_Qs[i])/ np.array(value_P)
        for j in range(n_sets):
            df = df.append({'obj': obj, 'distribution': distribution,
                        'eps': eps[i], 'test_number': j+1,  'value_P': value_P[j], 'value_Q': value_Qs[i][j], 
                        'ratio': ratio[j]}, ignore_index = True)
    df.to_csv('res_comp.csv', index=False)
   
   
# compare u(S) and v(S) for some set_items
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

