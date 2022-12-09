import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import theano
import csv
import random



az.style.use('arviz-darkgrid')


if __name__ == '__main__':
    
    #inceput cod generare CSV
    def generate_random_array(row, col):
        a = []
        for i in range(col):
            l = [i]
            for j in range(row):
                l.append(random.sample(range(-10, 10), 2))
            a.append(l)
        return a

    row = 1
    col = 500

    array = generate_random_array(row, col)

    f = open('sample.csv', 'w')
    w = csv.writer(f, lineterminator='\n')
    w.writerows(array)
    f.close()
    #terminare cod generare CSV

    dummy_data = np.loadtxt('date.csv')
    #dummy_data = np.loadtxt('sample.csv')


    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    #order = 5
    order1 = 2
    order2 = 3
    
    #model poli
    x_1p = np.vstack([x_1**i for i in range(1, order1+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    
    #model cubic
    x_1c = np.vstack([x_1**i for i in range(1, order2+1)])
    x_1x = (x_1c - x_1c.mean(axis=1, keepdims=True))
    x_1c.std(axis=1, keepdims=True)
    y_1x = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1x[0], y_1x)
    plt.xlabel('x')
    plt.ylabel('y')

    
    theano.config.blas__ldflags = ''

    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=order1)
        #β = pm.Normal('β', mu=0, sd=np.aray([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)
        
    with pm.Model() as model_c:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=order2)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_c = pm.sample(2000, return_inferencedata=True)
        
    

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

    #modeul liniar
    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new

    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    #model poli
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order1}')
    
    #model cubic
    α_c_post = idata_c.posterir['α'].mean(("chain", "draw")).values
    β_c_post = idata_c.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_c_post = α_c_post + np.dot(β_c_post, x_1x)

    plt.plot(x_1x[0][idx], y_c_post[idx], 'C2', label=f'model order {order2}')
    
    #waic_l = az.waic(idata_l, scale="deviance")
    #waic_l
    
    cmp_df = az.compare({'model_l':idata_l, 'model_p':idata_p, 'model_c':idata_c },
    method='BB-pseudo-BMA', ic="waic", scale="deviance")
    cmp_df

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    
  