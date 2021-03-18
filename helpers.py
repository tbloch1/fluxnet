import numpy as np
import pandas as pd
import datetime as dt
import glob
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

def mse_dim(true, pred):
    return np.mean( (pred-true)**2, axis=0)

def rmse_dim(true, pred):
    return np.sqrt(mse_dim(true,pred))

def mae_dim(true, pred):
    return np.mean( ((pred-true)**2)**0.5, axis=0)

def mape_dim(true, pred):
        return 100 * np.mean( np.abs((true-pred)/true), axis=0)

def smape_dim(true, pred):
        return 100 * np.mean( np.abs(pred-true)
                              / ((np.abs(true) + np.abs(pred))/2), axis=0)

def skill_score(ref_metric,pred_metric):
    return(1-pred_metric/ref_metric)

# def hetero_aleatoric(true,predval,predvar):
#     rel_error = ((true-predval)**2)
#     residual = 0.5*predvar # Training for the log_e(var)
#     return (rel_error/(2*torch.exp(predvar))) + residual

def hetero_aleatoric(true,predval,predvar):
    rel_error = ((true-predval)**2)
    residual = 0.5*torch.log(torch.exp(predvar)+1e-04) # Training for the log_e(var)
    return (rel_error/(2*(torch.exp(predvar)+1e-04))) + residual

def negative_log_likelihood(true, mu, sigma):
    dist = D.MultivariateNormal(mu, torch.diag_embed(sigma))
    return -dist.log_prob(true)

def timeseries_sample(data, nbins=50, n_min=100,
                      train_frac=0.7, val_frac=0.2, test_frac=0.1):
    data = data.sort_values('datetime')
    data['seconds'] = [(i-dt.datetime(1970,1,1)).total_seconds()
                       for i in data.datetime]
    n, edges = np.histogram(data.seconds,bins=nbins)

    train, val, test = [],[],[]

    for amt, l_edge, r_edge in zip(n,edges[:-1],edges[1:]):

        subset = data[(data.seconds > l_edge)
                          & (data.seconds < r_edge)]

        if amt > n_min:
            train.append(subset[:int(train_frac*len(subset))])
            val.append(subset[int(train_frac*len(subset))
                              :int((train_frac+val_frac)*len(subset))])
            test.append(subset[int((1-test_frac)*len(subset)):])
        else:
            train.append(subset)

    return(pd.concat(train).sort_index(),
           pd.concat(val).sort_index(),
           pd.concat(test).sort_index())
