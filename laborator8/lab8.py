import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import seaborn as sns
import scipy.stats as stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import arviz as az

if __name__ == "__main__":

    df = pd.read_csv('Admission.csv')

    admis = df['Admission'].values
    note = df['GRE'].values
    medie = df['GPA'].values
    
    
    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0,0].scatter(note, admis, alpha=0.6)
    axes[0,1].scatter(medie, admis, alpha=0.6)
    axes[0,0].set_ylabel("admis")
    axes[0,0].set_xlabel("note")
    axes[0,1].set_xlabel("medie")
    
    plt.savefig(sns.pairplot(df, hue ='Admission'))
    
  