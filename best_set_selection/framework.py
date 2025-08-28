import numpy as np
from numpy import random
from scipy import stats
import math
import random

from numpy.random import default_rng
rng = default_rng(2019)
np.random.seed(30)

from scipy.stats import expon
from scipy.stats import pareto
from scipy.stats import bernoulli
from itertools import combinations

random.seed(2019)


def Exp_sample(n, n_sample, params):
    mean_values = params
    samples = np.array([expon.rvs(scale=mean_values[i], size=n_sample, random_state=rng) for i in range(n)])
    return samples


def Pareto_sample(n, n_sample, params):
    shapes = params[:n]
    scales = params[-n:]
    samples = np.array([pareto.rvs(shapes[i], scale=scales[i], size=n_sample, random_state=rng) for i in range(n)])
    return samples 
    
    
def Exp_discrete(n, eps, params):

    means = params

    a = eps ** 2
    ## (1-eps)-quantile for ground elements
    tau = [expon(scale=means[i]).ppf(1-eps) for i in range(n)]
    ## support size
    J = int(np.ceil(np.log(a) / np.log(1-eps)))
    ## xstar
    xstar = [expon(scale=means[i]).expect(lambda x: x, lb = tau[i], conditional=True) for i in range(n)]
    ## middle parts
    Xmid = [[a*tau[j]/((1-eps)**(i)) for i in range(J)] for j in range(n)]
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


def Pareto_discrete(n, eps, params):

    shapes = params[:n]
    scales = params[-n:]

    a = eps ** 2
    ## (1-eps)-quantile for ground elements
    tau = [pareto(shapes[i], scale=scales[i]).ppf(1-eps) for i in range(n)]
    ## support size
    J = int(np.ceil(np.log(a) / np.log(1-eps))) 
    ## xstar
    xstar = [pareto(shapes[i], scale=scales[i]).expect(lambda x: x, lb = tau[i], conditional=True) for i in range(n)]
    ## middle parts
    Xmid = [[a*tau[j]/((1-eps)**i) for i in range(J)] for j in range(n)]
    # values
    values = np.c_[ np.zeros(n), Xmid, xstar]
    
    # probs
    ## replace last bin with threshold tau
    last_values = np.delete(values, -1, 1) 
    cdf_values = np.c_[last_values, tau]

    # prob as difference of grid cdfs for each element
    probs = [np.diff(pareto(shapes[i], scale=scales[i]).cdf(cdf_values[i,:])) for i in range(n)]
    probs = np.c_[probs, eps*np.ones(n)]
    return values, probs
    
    
def Discretized_sample(n, n_sample, values, probs):
    samples = []
    for i in range(n):
        discretized = stats.rv_discrete(values=(range(len(probs[i,:])),
             probs[i,:]), name='discretized')
        idx = discretized.rvs(size=n_sample, random_state=rng)
        sample = values[i,idx]
        samples.append(sample)
    return np.array(samples)
    

def get_items(n, k):
    return random.sample(range(n), k)
    
    
def f_value(items, samples, obj):
    samples_subset = samples[items]    
    if obj == "square_root":
        sum_samples = np.sum(samples_subset, axis=0)
        return np.mean(np.sqrt(sum_samples))
    
    elif obj == "max":
        return np.mean(np.max(samples_subset, axis=0))
    
    elif obj.startswith("CES-"):
        r = int(obj[4])
        powered = np.power(samples_subset, r)
        summed = np.sum(powered, axis=0)
        ces_values = np.power(summed, 1/r)
        return np.mean(ces_values)
    
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
    unvisited = set(items)  
    for j in range(k):
        if not unvisited:
            break
        scores_j = test_scores[list(unvisited), j]
        max_score = np.max(scores_j)
        res += max_score / (j + 1)
        # Find all indices with max score
        max_indices = [i for i in unvisited if test_scores[i,j] == max_score]
        unvisited.difference_update(max_indices)
    return res


def get_sampleP(n, n_sample, distribution, params):
    if distribution == "Exponential":
        samples = Exp_sample(n, n_sample, params)
    elif distribution[0:6] == "Pareto":        
        samples = Pareto_sample(n, n_sample, params)
    return samples


def get_sampleQ(n, eps, params, n_sample, distribution):
    if distribution == "Exponential":
        values, probs = Exp_discrete(n, eps, params) 
    elif distribution[0:6] == "Pareto":
        values, probs = Pareto_discrete(n, eps, params)
    samples_discrete = Discretized_sample(n, n_sample, values, probs)
    return samples_discrete