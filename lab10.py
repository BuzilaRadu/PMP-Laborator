import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import seaborn as sns
import scipy.stats as stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
from theano import tensor as TT
import arviz as az
import csv
import random



#Generare 3 clusters si unite itntr-un CSV
def generate_random_array(row, col):
    a = []
    for i in range(100):
        l = [i]
        for j in range(row):
            l.append(random.sample(range(-10, -5), 2))
        a.append(l)
        
    for i in range(200):
        l = [i]
        for j in range(row):
            l.append(random.sample(range(-5, 5), 2))
        a.append(l)
        
    for i in range(200):
        l = [i]
        for j in range(row):
            l.append(random.sample(range(5, 10), 2))
        a.append(l)
        
    return a

if __name__ == '__main__':
    row = 1
    col = 500

    array = generate_random_array(row, col)

   
    

    #codul exemplu laborator - nu a fost folosit
    clusters = 3
    n_cluster = [200, 150]
    n_total = sum(n_cluster)
    means = [5, 0]
    std_devs = [2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster),
    np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix));
    
    #creeare CSV de unde citim datele
    headerList = ['exp']
    f = open('sample.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(headerList)
    w = csv.writer(f, lineterminator='\n')
    w.writerows(array)
    f.close()
    
    
    
    cs = pd.read_csv('sample.csv')
    cs_exp = cs['exp']
    az.plot_kde(cs_exp)
    plt.hist(cs_exp, density=True, bins=30, alpha=0.3)
    plt.yticks([])
    
    clusters = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',mu=np.linspace(cs_exp.min(), cs_exp.max(), cluster), sd=10, shape=cluster,transform=pm.distributions.transforms.ordered)
            
            sd = pm.HalfNormal('sd', sd=10)
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
            idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)
    
    ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = np.ravel(ax)
    x = np.linspace(cs_exp.min(), cs_exp.max(), 100)
    for idx, idata_x in enumerate(idatas):
        posterior_x = idata_x.posterior.stack(samples=("chain", "draw"))
        x_ = np.array([x] * clusters[idx]).T
        for i in range(50):
            i_ = np.random.randint(0, posterior_x.samples.size)
            means_y = posterior_x['means'][:,i_]
            p_y = posterior_x['p'][:,i_]
            sd = posterior_x['sd'][i_]
            dist = stats.norm(means_y, sd)
            ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y.values, 1), 'C0', alpha=0.1)
        means_y = posterior_x['means'].mean("samples")
        p_y = posterior_x['p'].mean("samples")
        sd = posterior_x['sd'].mean()
        dist = stats.norm(means_y, sd)
        ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y.values, 1), 'C0', lw=2)
        ax[idx].plot(x, dist.pdf(x_) * p_y.values, 'k--', alpha=0.7)
        az.plot_kde(cs_exp, plot_kwargs={'linewidth':2, 'color':'k'}, ax=ax[idx])
        ax[idx].set_title('K = {}'.format(clusters[idx]))
        ax[idx].set_yticks([])
        ax[idx].set_xlabel('x')
        
    ppc_mm = [pm.sample_posterior_predictive(idatas[i], 500, models[i]) for i in range(3)]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    ax = np.ravel(ax)
    def iqr(x, a=0):
        return np.subtract(*np.percentile(x, [75, 25], axis=a))
    T_obs = iqr(cs_exp)
    for idx, d_sim in enumerate(ppc_mm):
        T_sim = iqr(d_sim['y'][:100].T, 1)
        p_value = np.mean(T_sim >= T_obs)
        az.plot_kde(T_sim, ax=ax[idx])
        ax[idx].axvline(T_obs, 0, 1, color='k', ls='--')
        ax[idx].set_title(f'K = {clusters[idx]} \n p-value {p_value:.2f}')
        ax[idx].set_yticks([])