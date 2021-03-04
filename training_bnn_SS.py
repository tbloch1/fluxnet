import numpy as np
import pandas as pd
import datetime as dt
import glob
import os
from tqdm.notebook import trange, tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import time
stime = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import helpers

import wandb
os.environ["WANDB_API_KEY"] = "2d250527070517d3f57f16e524cff14bfac02a58"
os.environ["WANDB_CONSOLE"] = "wrap" # https://github.com/wandb/client/issues/1581

torch.manual_seed(0)
np.random.seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str)
parser.add_argument('--heads',type=str)
parser.add_argument('--mltlim', type=float)
parser.add_argument('--params', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--optimiser',type=str)
parser.add_argument('--lr',type=float)
parser.add_argument('--loss',type=str)
parser.add_argument('--target',type=str)
parser.add_argument('--n_ensembles', type=int)
parser.add_argument('--n_hidden', type=int)
args = parser.parse_args()

model = getattr(models, args.model)
heads = args.heads
mltlim = args.mltlim
parameters = args.params.split(' ')
n_epochs = args.epochs
optimiser = getattr(torch.optim, args.optimiser)
lr = args.lr
loss_f = getattr(torch.nn, args.loss)
target = args.target
n_ensembles = args.n_ensembles
n_hidden = args.n_hidden

wandb.init(project="fluxnet", entity="tbloch",
           dir='/storage/silver/stfc_cg/hf832176/data/goes/training/')
wandb.config.update(args)

with open('config.json', 'w') as f:
        json.dump(dict(wandb.config), f)

fpath_g = '/storage/silver/stfc_cg/hf832176/data/goes/'
fpath_t = '/storage/silver/stfc_cg/hf832176/data/THEMIS/'
fpath_o = '/storage/silver/stfc_cg/hf832176/data/OMNI/1_min/'

fnames_g = np.sort(glob.glob(fpath_g+'goes??.parquet'))
fnames_t = np.sort(glob.glob(fpath_t+'???_mag.parquet'))


print('THEMIS Data')
def clean_themis(df):
    # Limit to equator and boundary region.
    df = df[(df.r > 7.75) & (df.r < 8.75) &
            (df.pos_z_mag.abs() < 0.5)]
    
    # Columns with bad data
    cols = ['esa_E_1', 'esa_E_2', 'esa_E_3', 'esa_E_4', 'esa_E_5',
            'esa_E_6', 'esa_E_7', 'esa_E_8', 'esa_E_9', 'esa_E_10',
            'esa_E_11', 'esa_E_12','esa_E_13', 'esa_E_14',
            'esa_E_15', 'esa_E_16', 'esa_E_17', 'esa_E_32',
            'E_12', 'E_13', 'E_14', 'E_15', 'E_16',
            'spin_ra', 'spin_dec', 'spin_per','spin_phase',
            'pos_x_gei', 'pos_y_gei', 'pos_z_gei','mlt_mag']
    df = df.drop(columns=cols)
    
    # Remove data with NaNs
    cols2 = ['esa_E_18', 'esa_E_19', 'esa_E_20', 'esa_E_21',
             'esa_E_22', 'esa_E_23','esa_E_24', 'esa_E_25',
             'esa_E_26', 'esa_E_27', 'esa_E_28', 'esa_E_29',
             'esa_E_30', 'esa_E_31',
             'E_1', 'E_2','E_3', 'E_4', 'E_5','E_6', 'E_7',
             'E_8', 'E_9', 'E_10', 'E_11']
    df.loc[:,cols2] = df.loc[:,cols2].replace(0,np.nan).values
    df = df.dropna(subset=cols2)
    
    print(df.shape)
    return df


tha = clean_themis(pd.read_parquet(fnames_t[0]))
thb = clean_themis(pd.read_parquet(fnames_t[1]))
thc = clean_themis(pd.read_parquet(fnames_t[2]))
thd = clean_themis(pd.read_parquet(fnames_t[3]))
the = clean_themis(pd.read_parquet(fnames_t[4]))


print('GOES Data')
def clean_goes(df):
    df = df[(df['800kevqual']==0) & (df['2mevqual']==0)]
    
    print(df.shape)
    return df


g13 = clean_goes(pd.read_parquet(fnames_g[0]))
g14 = clean_goes(pd.read_parquet(fnames_g[1]))
g15 = clean_goes(pd.read_parquet(fnames_g[2]))

omni = pd.read_parquet(fpath_o+'2000_2020_nan.parquet')


def omni_goes_align(omni,goes,parameters):
    omni_params = [i for i in parameters if i in omni.columns]
    omni = omni[omni_params].interpolate()
    om_go = goes.join(omni,how='inner',rsuffix='_o')
    # params = [i+'_o' for i in omni_params]
    om_go = om_go.dropna()
    return(om_go)

g13 = omni_goes_align(omni,g13,parameters)
g14 = omni_goes_align(omni,g14,parameters)
g15 = omni_goes_align(omni,g15,parameters)


def date_to_frac(date):
    return((date - dt.datetime(date.year, 1, 1)).total_seconds()
           / (dt.datetime(date.year+1, 1, 1)
              - dt.datetime(date.year, 1, 1)).total_seconds())

def date_to_tod(date):
    return ((date - date.replace(hour=0,minute=0,second=0)).total_seconds()
            / (24*60*60))

print('THEMIS-GOES Conjunction Data')
def mlt_conjunction(themis,goes,mlt_lim):
    th_g = themis.join(goes,how='inner',rsuffix='_g')
    th_g['mlt_diff'] = [i if i <= 12 else np.abs(i-24)
                        for i in np.abs(th_g.mlt-th_g.mlt_g)]
    th_g = th_g[th_g['mlt_diff'] < mlt_lim]
    th_g['mlt_sin'] = np.sin(th_g.mlt)
    th_g['mlt_cos'] = np.cos(th_g.mlt)
    th_g['mlt_sin_g'] = np.sin(th_g.mlt_g)
    th_g['mlt_cos_g'] = np.cos(th_g.mlt_g)

    th_g['year_frac'] = [date_to_frac(date) for date in th_g.index]
    th_g['date_sin'] = np.sin(th_g.year_frac)
    th_g['date_cos'] = np.cos(th_g.year_frac)

    th_g['day_frac'] = [date_to_tod(date) for date in th_g.index]
    th_g['day_sin'] = np.sin(th_g.day_frac)
    th_g['day_cos'] = np.cos(th_g.day_frac)

    print(th_g.shape)
    return th_g

th_g = []
for themis in [tha,thb,thc,thd,the]:
    for goes in [g13,g14,g15]:
        th_g.append(mlt_conjunction(themis,goes,mltlim))
        
thgdf = pd.concat(th_g, ignore_index=True)
print('Final Dataset Size: ',thgdf.shape)

data_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
devstr = 'cuda' if torch.cuda.is_available() else 'cpu'

X = thgdf[parameters].values

target_parameters = [target]
if heads == 'Multi':
    target_parameters = ['E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'E_6',
                         'E_7', 'E_8', 'E_9', 'E_10', 'E_11']
                         
y = np.log10(thgdf[target_parameters]).values

Xscaler = StandardScaler().fit(X[:int(0.6*len(X))])
X = Xscaler.transform(X)

yscaler = StandardScaler().fit(y[:int(0.6*len(y))])
y = yscaler.transform(y)

Xtrain = torch.from_numpy(X[:int(0.6*len(X))])
ytrain = torch.from_numpy(y[:int(0.6*len(y))])

Xval = torch.from_numpy(X[int(0.6*len(X)):int(0.8*len(X))])
yval = torch.from_numpy(y[int(0.6*len(y)):int(0.8*len(y))])

Xtest = torch.from_numpy(X[int(0.8*len(X)):])
ytest = torch.from_numpy(y[int(0.8*len(y)):])


if devstr == 'cuda':
    Xtrain = Xtrain.to(device)
    ytrain = ytrain.to(device)
    Xval = Xval.to(device)
    yval = yval.to(device)
    Xtest = Xtest.to(device)
    ytest = ytest.to(device)

    
net = model(n_feature=len(parameters), n_hidden=n_hidden, n_output=1)
if heads == 'Multi':
    net = model(n_feature=len(parameters), n_hidden=n_hidden, n_output=11)
if devstr == 'cuda':
    net = net.to(device)

optimizer = optimiser(net.parameters(), lr=lr)
# loss_func = loss_f()
# if heads == 'Multi':
#     loss_func = loss_f(reduction='none')

loss_func = helpers.hetero_aleatoric

savepath = '/storage/silver/stfc_cg/hf832176/data/goes/training/'
model_folder = 'models/{0:05d}_ensemble/model_{1}/'.format(len(glob.glob(savepath+'*')),1)
if not os.path.exists(savepath+model_folder):
    try:
        os.makedirs(savepath+model_folder)
    except Exception as e:
        print(e)


for epoch in range(n_epochs):
    outputs = net(Xtrain.float())
    prediction = outputs[:,:,0].view(-1,11)
    variance = outputs[:,:,1].view(-1,11)
    
    valout = net(Xval.float())
    valpred = valout[:,:,0].view(-1,11)
    valvar = valout[:,:,1].view(-1,11)


    trainloss = loss_func(ytrain.float(),prediction.float(),
                            variance.float()).mean()
    loss_report = loss_func(ytrain.float(),prediction.float(),
                            variance.float())
    valloss = loss_func(yval.float(),valpred.float(),
                        valvar.float())
    
    if epoch%100 == 0:
        testout = net(Xtest.float())
        testpred = testout[:,:,0].view(-1,11)
        testvar = testout[:,:,1].view(-1,11)
        testloss = (loss_func(ytest.float(), testpred.float(),
                              testvar.float()))
    

    optimizer.zero_grad()   # clear gradients for next train
    try:
        trainloss.backward()         # backpropagation, compute gradients
    except:
        print('Training loss val: ', trainloss)
        print('MLT limit: ', mltlim)
        trainloss.backward()
    optimizer.step()        # apply gradients

    valpred = yscaler.inverse_transform(valpred.data.cpu().numpy())
    yval_np = yscaler.inverse_transform(yval.data.cpu().numpy())
    error_report = (((valpred-yval_np)**2)/yval_np)


    if devstr == 'cuda':
        wandb.log({'Epoch': epoch,
                   'Train Loss': loss_report.mean().data.cpu().numpy(),
                   'Val Loss': valloss.mean().data.cpu().numpy(),
                   'Val RSE max': error_report.max(),
                   'Val RSE median': np.median(error_report),
                   'Test Loss': testloss.mean().data.cpu().numpy(),
                   'Val RSE E1 mean': error_report[0].mean(),
                   'Val RSE E2 mean': error_report[1].mean(),
                   'Val RSE E3 mean': error_report[2].mean(),
                   'Val RSE E4 mean': error_report[3].mean(),
                   'Val RSE E5 mean': error_report[4].mean(),
                   'Val RSE E6 mean': error_report[5].mean(),
                   'Val RSE E7 mean': error_report[6].mean(),
                   'Val RSE E8 mean': error_report[7].mean(),
                   'Val RSE E9 mean': error_report[8].mean(),
                   'Val RSE E10 mean': error_report[9].mean(),
                   'Val RSE E11 mean': error_report[10].mean()})
    else:
        wandb.log({'Epoch': epoch,
                   'Train Loss': loss_report.mean().data.numpy(),
                   'Val Loss': valloss.mean().data.numpy(),
                   'Val RSE max': error_report.max(),
                   'Val RSE median': np.median(error_report),
                   'Test Loss': testloss.mean().data.numpy(),
                   'Val RSE E1 mean': error_report[0].mean(),
                   'Val RSE E2 mean': error_report[1].mean(),
                   'Val RSE E3 mean': error_report[2].mean(),
                   'Val RSE E4 mean': error_report[3].mean(),
                   'Val RSE E5 mean': error_report[4].mean(),
                   'Val RSE E6 mean': error_report[5].mean(),
                   'Val RSE E7 mean': error_report[6].mean(),
                   'Val RSE E8 mean': error_report[7].mean(),
                   'Val RSE E9 mean': error_report[8].mean(),
                   'Val RSE E10 mean': error_report[9].mean(),
                   'Val RSE E11 mean': error_report[10].mean()})
