import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az


if __name__ == '__main__':

    with pm.Model() as model:
    
        nr_clienti = pm.Poisson("NC", 20)              
        timp_comanda = pm.Uniform('TC', lower=0.5, upper=1.5)
        timp_preparare = pm.Exponential('TP', 5)
        trace = pm.sample(2000)



        dictionary = {
                'nr_clienti': trace['NC'].tolist(),
                'timp_comanda': trace['TC'].tolist(),
                'timp_preparare': trace['TP'].tolist()
                }
        df = pd.DataFrame(dictionary)
        