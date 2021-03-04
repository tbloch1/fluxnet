import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


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

def hetero_aleatoric(true,predval,predvar):
    rel_error = ((true-predval)**2)
    residual = 0.5*predvar # Training for the log_e(var)
    return (rel_error/(2*torch.exp(predvar))) + residual