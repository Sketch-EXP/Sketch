import numpy as np
from numpy import random
from scipy import stats
import math

from numpy.random import default_rng
rng = default_rng(2019)
np.random.seed(30)

from scipy.stats import expon
from scipy.stats import pareto
from scipy.stats import bernoulli



def Exp_sample(n, n_sample):
#    mean_values = np.random.uniform(0.0, 1.0, n)
    mean_values = np.linspace(0.01, 1.0, n)
    samples = np.array([expon.rvs(scale=mean_values[i], size=n_sample, random_state=rng) for i in range(n)])
    return mean_values, samples
    
def Pareto_sample(n, n_sample, m):
#    shapes = np.random.uniform(1.1, 3.0, n)
    shapes = np.linspace(1.1, 3, n)
    samples = np.array([pareto.rvs(shapes[i], scale=m, size=n_sample, random_state=rng) for i in range(n)])
    return shapes, samples 
    
    
def Exp_discrete(n, eps, means):
    a = eps ** 2
    ## (1-eps)-quantile for ground elements
    tau = [expon(scale=means[i]).ppf(1-eps) for i in range(n)]
    ## support size
    #l = np.int(np.floor(np.log(a) / np.log(1-eps)) + 1)
    J = int(np.ceil(np.log(a) / np.log(1-eps)))
    ## xstar
    xstar = [expon(scale=means[i]).expect(lambda x: x, lb = tau[i], conditional=True) for i in range(n)]
    ## middle parts
    # Xmid = [[a*tau[j]/((1-eps)**(i-1)) for i in range(l+1)] for j in range(n)]
    Xmid = [[a*tau[j]/((1-eps)**i) for i in range(J)] for j in range(n)]
    # values
    values = np.c_[ np.zeros(n), Xmid, xstar]
    
    # probs
    ## replace last bin with threshold tau
    last_values = np.delete(values, -1, 1) 
    cdf_values = np.c_[last_values, tau]

    # prob as difference of grid cdfs for each element
    probs = [np.diff(expon.cdf(cdf_values[i,:],scale=means[i])) for i in range(n)]
    probs = np.c_[probs, eps*np.ones(n)]
    return values, probs
    
def Pareto_discrete(n, eps, shapes, m):
    a = eps ** 2
    ## (1-eps)-quantile for ground elements
    tau = [pareto(shapes[i], scale=m).ppf(1-eps) for i in range(n)]
    ## support size
    #l = np.int(np.floor(np.log(a) / np.log(1-eps)) + 1)
    J = int(np.ceil(np.log(a) / np.log(1-eps)))
    ## xstar
    xstar = [pareto(shapes[i], scale=m).expect(lambda x: x, lb = tau[i], conditional=True) for i in range(n)]
    
    ## middle parts
    # Xmid = [[a*tau[j]/((1-eps)**(i-1)) for i in range(l+1)] for j in range(n)]
    Xmid = [[a*tau[j]/((1-eps)**i) for i in range(J)] for j in range(n)]
    # values
    values = np.c_[ np.zeros(n), Xmid, xstar]
    
    # probs
    ## replace last bin with threshold tau
    last_values = np.delete(values, -1, 1) 
    cdf_values = np.c_[last_values, tau]

    # prob as difference of grid cdfs for each element
    probs = [np.diff(pareto(shapes[i], scale=m).cdf(cdf_values[i,:])) for i in range(n)]
    probs = np.c_[probs, eps*np.ones(n)]
    return values, probs
    
    
def Discretized_sample(n, n_sample, values, probs):
    samples = []
    for i in range(n):
        discretized = stats.rv_discrete(values=(range(len(probs[i,:])),
             np.round(probs[i,:], decimals=7)), name='discretized')
        idx = discretized.rvs(size=n_sample, random_state=rng)
        sample = values[i,idx]
        samples.append(sample)
    return np.array(samples)
    
import random
random.seed(2019)
def get_items(n, k):
    return random.sample(range(n), k)
    
    
def f_value(items, samples, obj):
    n_samples = np.shape(samples)[1]
    temp = 0
    for j in range(n_samples):
        if obj == "square_root":
            temp += math.sqrt(np.sum(samples[items,j]))
        elif obj == "max":
            temp += np.max(samples[items,j])
        elif obj[0:4] == "CES-":
            r = int(obj[4])
            temp+= np.power(np.sum(np.power(samples[items,j], r)), 1/r)   
        elif obj[0:4] == "FOS-":
            r = float(obj[4:8])
            temp+= np.power(np.sum(samples[items,j]), r)  
        elif obj == "sp":
            temp += np.prod(1-np.exp(-samples[items,j]))  
    if obj != "sp":
        return temp / n_samples
    elif obj == "sp":
        return 1 - (temp / n_samples)   
    
def compute_test_scores(n, k, n_sample, samples, obj):
    test_scores = np.zeros((n, k))
    for deg in range(1, k+1):
        for i in range(n):
            temp = 0
            n_batches = int(np.floor(n_sample / deg))
            for j in range(n_batches):
                if obj == "square_root":
                    temp += math.sqrt(np.sum(samples[i][j*deg:(j+1)*deg]))
                elif obj == "max":
                    temp += np.max(samples[i][j*deg:(j+1)*deg])
                elif obj[0:4] == "CES-":
                    r = int(obj[4])
                    temp += np.power(np.sum(np.power(samples[i][j*deg:(j+1)*deg], r)), 1/r)      
            test_scores[i,deg-1] = temp / n_batches
    return test_scores 
    
    
def score_value(n, k, items, test_scores):
    res = 0
    unvisited = items[:]
    for j in range(k):
        score = test_scores[unvisited, j]
        res += np.max(score) /(j +1)
        for i in range(n):
            if test_scores[i,j] == np.max(score):     
                unvisited.remove(i)
    return res
