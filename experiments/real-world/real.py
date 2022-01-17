import pandas as pd
import matplotlib.pyplot as plt 

# fixed parameters
n_sample = 100
n_sets = 50
kvals = [5, 10]
objs = ['max', 'CES-2', 'square_root']


def Real_sample(data, candidates, n_sample):
    samples_p = []
    for i in candidates:
    	samples_i = np.random.choice(data[i], n_sample)
    	samples_p.append(samples_i)
    samples_p = np.array(samples_p)
    return samples_p

def Discretized_real_sample(data, candidates, eps, n_sample):
    samples_q = []
    for i in candidates:
        cdf = ECDF(data[i])
        customdiscrete = stats.rv_discrete(values=(cdf.x[1:]*1e4 , np.diff(cdf.y)), name='customdiscrete')
        a = eps ** 2
        ## (1-eps)-quantile for ground elements
        tau = customdiscrete.ppf(1-eps) 
        ## support size
        l = int(np.floor(np.log(a) / np.log(1-eps)) + 1)
        ## xstar (conditional empirical mean)
        xstar = np.mean(cdf.x[cdf.x >= (tau / 1e4)])
        ## middle parts
        Xmid = [a*tau/(1e4*(1-eps)**(i-1)) for i in range(l+1)]
        # values
        values = np.r_[ [0], Xmid, xstar]
        
        # probs
        ## replace last bin with tau
        cdf_values = np.append(values[:-1], tau/1e4)
        # prob as difference of grid cdfs for each element
        probs = np.diff(customdiscrete.cdf(cdf_values * 1e4)) 
        probs = np.append(probs, 1-customdiscrete.cdf(tau)) # add true tail value
        
        discretized = stats.rv_discrete(values=(range(len(probs)),
                 np.round(probs, decimals=7)), name='discretized')
        # discretized samples
        idx = discretized.rvs(size=n_sample,random_state=rng)
        samples_i = values[idx]
        samples_q.append(samples_i)
    samples_q = np.array(samples_q)
    return samples_q
    
def create_output_comp():
    column_labels_raw = ['obj','k','test_number', 'ratio_TS', 'ratio_EB']
    df_raw = pd.DataFrame(columns=column_labels_raw)
    df_raw.to_csv('res_score.csv', index=False)

def get_sets(n, k, n_sets):
    set_items = []
    for j in range(n_sets):
        items = get_items(n,k)
        set_items.append(items)
    return set_items

def get_value(n, n_sample, samples, obj, set_items):
    value = []
    for items in set_items:
        value.append(f_value(items, samples, obj))
    return value
    
    
def get_valueS(n, k, n_sample, samples, obj, set_items):
    value_S = []
    test_scores = compute_test_scores(n, k, n_sample, samples, obj)
    for items in set_items:
        value_S.append(score_value(n, k, items, test_scores))
    return value_S

def run_comparison(n, k, n_sample, samples_p, samples_q, obj, n_sets, set_items):
    df = pd.read_csv('./res_score.csv')
    
    value_P = get_value(n, n_sample, samples_p, obj, set_items)
    value_Q = get_value(n, n_sample, samples_q, obj, set_items)
    value_S = get_valueS(n, k, n_sample, samples_p, obj, set_items)
    
    for j in range(n_sets):    
        df = df.append({'obj': obj, 'k':k,
                        'test_number': j+1, 'ratio_EB': value_Q[j]/value_P[j],
                        'ratio_TS': value_S[j]/value_P[j]}, ignore_index = True)
    df.to_csv('res_score.csv', index=False)
    
def bar_plot(objs):
    df = pd.read_csv('./res_score.csv')
    fig, axes = plt.subplots(1, 2, figsize = (6.4*2, 4.8))
    for i, k in enumerate(kvals):  
        current = df[(df['k'] == k)]
        x_axis = np.arange(len(objs))
        
        # calculate means and stds
        TS_mean = current.groupby(['obj'])['ratio_TS'].mean()
        EB_mean = current.groupby(['obj'])['ratio_EB'].mean()     
        TS_std = current.groupby(['obj'])['ratio_TS'].std()
        EB_std = current.groupby(['obj'])['ratio_EB'].std()  
        
        axes[i].bar(x_axis -0.15, TS_mean, yerr= TS_std, capsize=5, width=0.3, label = 'Test Score', alpha= 0.5)
        axes[i].bar(x_axis +0.15, EB_mean, yerr= EB_std, capsize=5, width=0.3, label = 'Our method', alpha= 0.5)
        # add legends
        axes[i].legend(fontsize=14,loc='upper left')
        axes[i].grid()
        # set xticks and labels
        axes[i].set_xticks(x_axis)
        axes[i].set_xticklabels(objs, fontsize= 15)
        axes[i].set_xlabel("objectives", fontsize= 15)
        axes[i].set_ylabel("$v(S)/u(S)$", fontsize= 15)
        axes[i].set_ylim((0, 2.5)) 
        
