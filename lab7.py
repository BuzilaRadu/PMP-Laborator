import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values
    
    
    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0,0].scatter(speed, price, alpha=0.6)
    axes[0,1].scatter(hardDrive, price, alpha=0.6)
    axes[1,0].scatter(ram, price, alpha=0.6)
    axes[1,1].scatter(premium, price, alpha=0.6)
    axes[0,0].set_ylabel("Price")
    axes[0,0].set_xlabel("Speed")
    axes[0,1].set_xlabel("HardDrive")
    axes[1,0].set_xlabel("Ram")
    axes[1,1].set_xlabel("Premium")
    plt.savefig('price_correlations.png')
    
    
    cog_score_pc_model = pm.Model()

    with cog_score_pc_model:
        a = pm.Normal('a', mu=0, sd=10)
        
        # bEdu = pm.Normal('bEdu', mu=0, sd=10)
        bHard = pm.Normal('bHard', mu=0, sd=10)
        
        bRam = pm.Normal('bRam', mu=0, sd=10)
        
        sigma = pm.HalfNormal('sigma', sd=1)

        # mu = pm.Deterministic('mu',a + bEdu * educ_cat)
        mu = pm.Deterministic('mu',a + bHard * hardDrive + bRam * ram)
        
        price = pm.Normal('price_like', mu=mu, sd=sigma, observed=price)

        step = pm.Slice()
        step = pm.Metropolis()
        trace = pm.sample(1000, step=step, tune=1000, cores=4)

    
    a_mean = trace['a'].mean().item()
    
    price = data['Price'].values
    
    ppc = pm.sample_posterior_predictive(trace, samples=100, model=cog_score_pc_model)
    
    # plt.plot(educ_cat, a_mean + bEdu_mean * educ_cat, 'r')
    # sig = az.plot_hdi(educ_cat, ppc['ppvt_like'], hdi_prob=0.97, color='k')
    # plt.xlabel('Education level')
    # plt.ylabel('Cog. score', rotation=0)
    # plt.savefig('bayesian_regression_line_mom_edu.png')
    
    
    plt.plot(price, a_mean + bHard * hardDrive + bRam * ram, 'r')
    sig = az.plot_hdi(price, ppc['price_like'], hdi_prob=0.97, color='k')
    plt.xlabel('Hard')
    plt.xlabel('Ram')
    plt.ylabel('Cog. score', rotation=0)
    
    
    