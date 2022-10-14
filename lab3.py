import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    cutremur = pm.Bernoulli('C', 0.0005)
    incendiu_c = pm.Deterministic('I', pm.math.switch(cutremur, 0.03, 0.01 ))
    incendiu = pm.Bernoulli('I', p=incendiu_c)
    alarma_i = pm.Deterministic('A_i', pm.math.switch(incendiu, pm.math.switch(incendiu, 0.98, 0.0001), pm.math.switch(cutremur, 0.03, 0.0001)))
    alarma = pm.Bernoulli('A', p=alarma)
    trace = pm.sample(20000)


    dictionary = {
              'cutremur': trace['C'].tolist(),
              'incendiu': trace['I'].tolist(),
              'alarma': trace['A'].tolist()
              }
df = pd.DataFrame(dictionary)


p_cutremur = [(df['alarma'] == 1)].shape[0]
print(p_cutremur)

