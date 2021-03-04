import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats
import glob
import os
import psutil
import sys
# from tqdm.notebook import trange, tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import properscoring as ps

import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from helpers import mse_dim, rmse_dim, mae_dim, skill_score
from helpers import mape_dim, smape_dim

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


thgdf = pd.concat(th_g, ignore_index=True).copy()
print('Final Dataset Size: ',thgdf.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
devstr = 'cuda' if torch.cuda.is_available() else 'cpu'

X_i = thgdf[parameters].values

target_parameters = [target]
if heads == 'Multi':
    target_parameters = ['E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'E_6',
                         'E_7', 'E_8', 'E_9', 'E_10', 'E_11']
                         
y = np.log10(thgdf[target_parameters]).values

Xscaler = StandardScaler().fit(X_i[:int(0.6*len(X_i))])
X = Xscaler.transform(X_i)

Xtrain_i = torch.from_numpy(X[:int(0.6*len(X))])
ytrain_i = torch.from_numpy(y[:int(0.6*len(y))])

Xval_i = torch.from_numpy(X[int(0.6*len(X)):int(0.8*len(X))])
yval_i = torch.from_numpy(y[int(0.6*len(y)):int(0.8*len(y))])

Xtest_i = torch.from_numpy(X[int(0.8*len(X)):])
ytest_i = torch.from_numpy(y[int(0.8*len(y)):])

# Cleaning variables
del omni; omni = []
del g13; g13 = []
del g14; g14 = []
del g15; g15 = []
del tha; tha = [] 
del thb; thb = [] 
del thc; thc = [] 
del thd; thd = [] 
del the; the = [] 
del themis; themis = []
del goes; goes = []


savepath = '/storage/silver/stfc_cg/hf832176/data/goes/training/'
folder_number = len(glob.glob(savepath+'models/*'))

test_predictions = []

for i in range(n_ensembles):
    random_idx = torch.randint(0,len(Xtrain_i),
                               (len(Xtrain_i),)
                              )
    Xtrain = Xtrain_i[random_idx]
    ytrain = ytrain_i[random_idx]

    # if devstr == 'cuda':
    Xtrain = Xtrain.to(device)
    ytrain = ytrain.to(device)
    Xval = Xval_i.to(device)
    yval = yval_i.to(device)
    Xtest = Xtest_i.to(device)
    ytest = ytest_i.to(device)

        
    net = model(n_feature=len(parameters), n_hidden=n_hidden, n_output=1)
    if heads == 'Multi':
        net = model(n_feature=len(parameters), n_hidden=n_hidden, n_output=11)
    if devstr == 'cuda':
        net = net.to(device)

    # wandb.watch(net)

    optimizer = optimiser(net.parameters(), lr=lr)
    loss_func = loss_f()
    if heads == 'Multi':
        loss_func = loss_f(reduction='none')

    model_folder = 'models/{0:05d}_ensemble/model_{1}/'.format(folder_number,i)
    if not os.path.exists(savepath+model_folder):
        try:
            os.makedirs(savepath+model_folder)
        except Exception as e:
            print(e)
            pass
    else:
        pass

    val_100s = [999.999]
    val_counter = 0

    # pbar = trange(n_epochs)
    for epoch in range(n_epochs+1):
        prediction = net(Xtrain.float())
        valpred = net(Xval.float())

        if heads != 'Multi':
            trainloss = loss_func(prediction.float(), ytrain.float())
            valloss = loss_func(valpred.float(), yval.float())

        if heads == 'Multi':
            trainloss = (loss_func(prediction.float(), ytrain.float())/ytrain).max()
            loss_report = (loss_func(prediction.float(), ytrain.float())/ytrain)
            valloss = (loss_func(valpred.float(), yval.float())/yval)
        
        if epoch%100 == 0:
            if valloss.mean().data.cpu().numpy() > np.min(val_100s):
                val_counter += 1
            else:
                val_counter = 0

                ckpt_name = 'epoch_{}.pt'.format(epoch)
                torch.save(net.state_dict(),
                           savepath+model_folder+ckpt_name)

            val_100s.append(valloss.mean().data.cpu().numpy())

        
        optimizer.zero_grad()   # clear gradients for next train
        trainloss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if devstr == 'cuda':
            wandb.log({'Epoch': epoch,
                       "Train Loss": loss_report.max().data.cpu().numpy(),
                       "Val Loss": valloss.mean().data.cpu().numpy(),
                       "Memory_MB": psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)})
        else:
            wandb.log({'Epoch': epoch,
                       "Train Loss": loss_report.max().data.numpy(),
                       "Val Loss": valloss.mean().data.numpy(),
                       "Memory_MB": psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2})
        
        if val_counter == 20:
            print('Stopped early\nEpoch - {}'.format(epoch))
            break
    
    # Evalutation
    best_epoch = 100 * (np.nanargmin(val_100s) - 1) # -1 due to nan at [0]
    best_model = savepath+model_folder+'epoch_{}.pt'.format(best_epoch)
    
    eval_model = model(n_feature=len(parameters), n_hidden=n_hidden, n_output=11)
    if devstr == 'cuda':
        eval_model = eval_model.to(device)
    eval_model.load_state_dict(torch.load(best_model, map_location=device))
    eval_model.eval()
    
    # Fixing memory problems?
    for param in eval_model.parameters():
        param.requires_grad = False


    if devstr == 'cuda':
        test_predictions.append(10**eval_model(Xtest.float()).data.cpu().numpy())
    else:
        test_predictions.append(10**eval_model(Xtest.float()).data.numpy())


    # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    # Fixing memory problems?
    del Xtrain; Xtrain = []
    del ytrain; ytrain = []
    del Xval; Xval = []
    del yval; yval = []
    del Xtest; Xtest = []
    del ytest; ytest = []
    del net; net = []
    del prediction; prediction = []
    del valpred; valpred = []
    del trainloss; trainloss = []
    del loss_report; loss_report = []
    del valloss; valloss = []
    del optimizer; optimizer = []
    del loss_func; loss_func = []
    del eval_model; eval_model = []
    del param; param = []
    torch.cuda.empty_cache()

test_predictions = np.array(test_predictions)
ens_mean = test_predictions.mean(axis=0)
ens_std = test_predictions.std(axis=0)
ens_uqt = np.quantile(test_predictions,0.75,axis=0)
ens_lqt = np.quantile(test_predictions,0.25,axis=0)
y_exp = 10**y[int(0.8*len(y)):]

# reference prediction
y800kev = thgdf['800kevflux'].values[int(0.8*len(X)):]
y800kev = np.transpose([y800kev]*11)

#region Plot reliability diagram per energy channel

plt.figure(figsize=(16,9),dpi=400)
axs = [plt.subplot(3,4,i+1) for i in range(11)]

[axs[i].scatter(ens_mean[:,i],y_exp[:,i],
                s=1,c='C0')
 for i in range(11)]

[i.plot([1e3,1e8],[1e4,1e8], c='k', linestyle='-', linewidth=1) for i in axs]

[axs[i].set_title('Energy Channel: {0}'.format(i+1)) for i in range(11)]
[i.set_ylabel('True Values') for i in axs[::4]]
[i.set_xlabel('Predicted Values') for i in axs[-3:]]
[i.set_yscale('log') for i in axs]
[i.set_xscale('log') for i in axs]
[ax.set_xlim(0.9*ens_mean[:,i].min(), 1.1*ens_mean[:,i].max())
 for i,ax in zip(range(11),axs)]
[ax.set_ylim(0.9*y_exp[:,i].min(), 1.1*y_exp[:,i].max())
 for i,ax in zip(range(11),axs)]

plt.tight_layout()
model_folder = 'models/{0:05d}_ensemble/'.format(folder_number)
plt.savefig(savepath+model_folder+'Reliability.pdf')
plt.close()
#endregion


#region Plot metric performance on logged data 

plt.figure(figsize=(8,4.5),dpi=400)
axs = [plt.subplot(2,3,i+1) for i in range(6)]

# MSE
axs[0].plot(np.arange(11)+1,
            mse_dim(np.log10(ens_mean),np.log10(y_exp)))
# RMSE
axs[1].plot(np.arange(11)+1,
            rmse_dim(np.log10(ens_mean),np.log10(y_exp)))
# MAE
axs[2].plot(np.arange(11)+1,
            mae_dim(np.log10(ens_mean),np.log10(y_exp)))
# MAPE
axs[3].plot(np.arange(11)+1,
            mape_dim(np.log10(ens_mean),np.log10(y_exp)))
# SMAPE
axs[4].plot(np.arange(11)+1,
            smape_dim(np.log10(ens_mean),np.log10(y_exp)))
# Spearman's Rank
axs[5].plot(np.arange(11)+1,
            [scipy.stats.spearmanr(ens_mean[:,i],np.log10(y_exp)[:,i])[0]
             for i in range(11)])

# [i.set_yscale('log') for i in axs[:3]]
[i.set_title(j)
 for i,j in zip(axs,
                ['MSE','RMSE','MAE','MAPE','SMAPE',"Spearman's Rank"])]
[i.set_ylabel('Metric Values') for i in axs[:6]]
[i.set_xlabel('Energy Channel') for i in axs[:6]]

plt.tight_layout()
model_folder = 'models/{0:05d}_ensemble/'.format(folder_number)
plt.savefig(savepath+model_folder+'Metrics_log.pdf')
plt.close()
#endregion


#region Plot skill based on logged data

plt.figure(figsize=(8,4.5),dpi=400)
axs = [plt.subplot(2,3,i+1) for i in range(6)]

# MSE
axs[0].plot(np.arange(11)+1,
            skill_score(mse_dim(np.log10(ens_mean),np.log10(y800kev)),
                        mse_dim(np.log10(ens_mean),np.log10(y_exp))))
# RMSE
axs[1].plot(np.arange(11)+1,
            skill_score(rmse_dim(np.log10(ens_mean),np.log10(y800kev)),
                        rmse_dim(np.log10(ens_mean),np.log10(y_exp))))
# MAE
axs[2].plot(np.arange(11)+1,
            skill_score(mae_dim(np.log10(ens_mean),np.log10(y800kev)),
                        mae_dim(np.log10(ens_mean),np.log10(y_exp))))
# MAPE
axs[3].plot(np.arange(11)+1,
            skill_score(mape_dim(np.log10(ens_mean),np.log10(y800kev)),
                        mape_dim(np.log10(ens_mean),np.log10(y_exp))))
# SMAPE
axs[4].plot(np.arange(11)+1,
            skill_score(smape_dim(np.log10(ens_mean),np.log10(y800kev)),
                        smape_dim(np.log10(ens_mean),np.log10(y_exp))))
# Spearman's Rank
ref_sr = [scipy.stats.spearmanr(np.log10(ens_mean)[:,i],
                                np.log10(y800kev)[:,i])[0]
           for i in range(11)]
pred_sr = [scipy.stats.spearmanr(np.log10(ens_mean)[:,i],
                                 np.log10(y_exp)[:,i])[0]
           for i in range(11)]

axs[5].plot(np.arange(11)+1,
            skill_score(np.array(ref_sr), np.array(pred_sr)))

# [i.set_yscale('log') for i in axs[:3]]
[i.set_title(j)
 for i,j in zip(axs,
                ['MSE','RMSE','MAE','MAPE','SMAPE',"Spearman's Rank"])]
[i.set_ylabel('Skill Score') for i in axs[:6]]
[i.set_xlabel('Energy Channel') for i in axs[:6]]

plt.tight_layout()
model_folder = 'models/{0:05d}_ensemble/'.format(folder_number)
plt.savefig(savepath+model_folder+'Skill_scores_log.pdf')
plt.close()
#endregion

#region Plot CRaPS and CRaPS SS
test_preds_T = np.transpose(test_predictions, (1,0,2))
crps = [np.mean(ps.crps_ensemble(np.log10(y_exp)[:,i],
                                 np.log10(test_preds_T)[:,:,i]))
        for i in range(11)]
crps_ref = [np.mean(ps.crps_ensemble(np.log10(y_exp)[:,i],
                                     np.log10(y800kev)[:,i]))
            for i in range(11)]
crps = np.array(crps)
crps_ref = np.array(crps_ref)

plt.figure(figsize=(8,4.5))
axs = [plt.subplot(1,2,i+1) for i in range(2)]

axs[0].plot(np.arange(11),crps)
axs[1].plot(np.arange(11), skill_score(crps_ref,crps))

[i.set_xlabel('Energy Channel') for i in axs]
axs[0].set_ylabel('CRPS')
axs[1].set_ylabel('Skill Score')

plt.tight_layout()
plt.savefig(savepath+model_folder+'CRaPS_log.pdf')
plt.close()
#endregion


#region Plot violins of distributions of pred vs true

plt.figure(figsize=(8,4.5),dpi=400)

sns.violinplot(x = [i for i in range(ens_mean.shape[1])]*len(ens_mean)*2,
               y = np.concatenate([np.log10(ens_mean.flatten()),
                                   np.log10(y_exp.flatten())]),
               hue = np.concatenate([['Predicted']*len(ens_mean.flatten()),
                                     ['True']*len(ens_mean.flatten())]),
               palette="muted", split=True, cut=0)

# plt.yscale('log')

plt.title('Distributions')
plt.ylabel('Flux')
plt.xlabel('Energy Channel')

plt.tight_layout()
model_folder = 'models/{0:05d}_ensemble/'.format(folder_number)
plt.savefig(savepath+model_folder+'Distributions.pdf')
plt.close()
#endregion


#region Timeseries plots
if not os.path.exists(savepath+model_folder+'ts/'):
    try:
        os.makedirs(savepath+model_folder+'ts/')
    except Exception as e:
        print(e)
        pass
else:
    pass

test_slices = thgdf[-482-1478:-482]
split = test_slices.r.values[:-1]-test_slices.r.values[1:]
test_ens_mean = ens_mean[-482-1478:-482]
test_ens_std = ens_std[-482-1478:-482]
test_ens_uqt = ens_uqt[-482-1478:-482]
test_ens_lqt = ens_lqt[-482-1478:-482]

orb_idx = 0
orb_idxs = []
for i in split:
    if (split < 0).sum() < (split > 0).sum():
        if i<0:
            orb_idxs.append(orb_idx)
        else:
            orb_idxs.append(orb_idx)
            orb_idx += 1
    else:
        if i>0:
            orb_idxs.append(orb_idx)
        else:
            orb_idxs.append(orb_idx)
            orb_idx += 1

orb_idxs = np.array(orb_idxs)

idx_cnt = [((orb_idxs==i).sum(),i) for i in range(63)]
idx_cnt = np.array(idx_cnt)

#region Plotting each SC crossing with panel for energy
for count,idx in idx_cnt:
    if count > 20:
        tslice = test_slices[:-1][orb_idxs == idx]
        ts_pred = test_ens_mean[:-1][orb_idxs == idx]
        ts_pred_std = test_ens_std[:-1][orb_idxs == idx]
        ts_pred_uqt = test_ens_uqt[:-1][orb_idxs == idx]
        ts_pred_lqt = test_ens_lqt[:-1][orb_idxs == idx]
        assert len(tslice) == len(ts_pred)

        plt.figure(figsize=(16,9))
        axs = [plt.subplot(3,4,i+1) for i in range(11)]

        [ax.plot(tslice.index,tslice[j],c='C0',label='True')
         for ax,j in zip(axs,target_parameters)]
        [ax.plot(tslice.index,ts_pred[:,j],c='C1',label='Predicted')
         for ax,j in zip(axs,np.arange(11))]
        # [ax.plot(tslice.index,
        #           ts_pred[:,j]+ts_pred_std[:,j],
        #           c='C1',label='Predicted',linestyle=':')
        #  for ax,j in zip(axs,np.arange(11))]

        # [ax.plot(tslice.index,
        #          ts_pred[:,j]-ts_pred_std[:,j],
        #          c='C1',label='Predicted',linestyle=':')
        #  for ax,j in zip(axs,np.arange(11))]
        
        [ax.plot(tslice.index,
                  ts_pred_uqt[:,j],
                  c='C1',label='Predicted',linestyle=':')
         for ax,j in zip(axs,np.arange(11))]

        [ax.plot(tslice.index,
                ts_pred_lqt[:,j],
                 c='C1',label='Predicted',linestyle=':')
         for ax,j in zip(axs,np.arange(11))]

        [ax.set_title('Energy Channel: {0} '.format(i+1) +
                      '| MAPE: {0:.2f}'.format(mape_dim(tslice[j],ts_pred[:,i])))
         for ax,i,j in zip(axs,np.arange(11),target_parameters)]

        [ax.set_yscale('log') for ax in axs]
        [ax.set_ylabel('Flux') for ax in axs[::4]]
        [ax.set_xlabel('Time') for ax in axs[-3:]]

        plt.tight_layout()
        model_folder = 'models/{0:05d}_ensemble/'.format(folder_number)
        plt.savefig(savepath+model_folder+'ts/{0}.pdf'.format(idx))
        plt.close()
#endregion

#region Plotting every SC crossing, one plot per energy
# for i in range(11):
#     plt.figure()
#     [plt.plot(test_slices.index[:-1][orb_idxs == idx],
#      test_slices[target_parameters[i]].values[:-1][orb_idxs == idx],c='C0')
#      for idx in idx_cnt[idx_cnt[:,0] > 20][:,1]]
    
#     [plt.plot(test_slices.index[:-1][orb_idxs == idx],
#      test_ens_mean[:-1,i][orb_idxs == idx],c='C1')
#      for idx in idx_cnt[idx_cnt[:,0] > 20][:,1]]

#     plt.yscale('log')
#     plt.xlabel('Time Index')
#     plt.ylabel('Flux')

#     plt.savefig(savepath+model_folder+'ts/timeseries_E{}.pdf'.format(i+1))
#     plt.close()

for i in range(11):
    plt.figure()
    [plt.scatter(np.median(test_slices.index[:-1][orb_idxs == idx].values),
     np.median(test_slices[target_parameters[i]].values[:-1][orb_idxs == idx]),c='C0')
     for idx in idx_cnt[idx_cnt[:,0] > 20][:,1]]
    
    [plt.errorbar(x = np.median(test_slices.index[:-1][orb_idxs == idx].values),
                  y = np.median(test_ens_mean[:-1,i][orb_idxs == idx]),
                  yerr = [[np.median(test_ens_mean[:-1,i][orb_idxs == idx])
                          - np.median(test_ens_lqt[:-1,i][orb_idxs == idx])],
                          [np.median(test_ens_uqt[:-1,i][orb_idxs == idx])
                          - np.median(test_ens_mean[:-1,i][orb_idxs == idx])]],
                  c='C1',fmt='o',capsize=6)
     for idx in idx_cnt[idx_cnt[:,0] > 20][:,1]]

    # [plt.scatter(np.median(test_slices.index[:-1][orb_idxs == idx].values),
    #  np.median(test_ens_uqt[:-1,i][orb_idxs == idx]),c='C1')
    #  for idx in idx_cnt[idx_cnt[:,0] > 20][:,1]]

    # [plt.scatter(np.median(test_slices.index[:-1][orb_idxs == idx].values),
    #  np.median(test_ens_lqt[:-1,i][orb_idxs == idx]),c='C1')
    #  for idx in idx_cnt[idx_cnt[:,0] > 20][:,1]]

    plt.yscale('log')
    plt.xlabel('Time Index')
    plt.ylabel('Flux')

    plt.savefig(savepath+model_folder+'ts/timeseries_E{}.pdf'.format(i+1))
    plt.close()
#endregion

#endregion

sys.exit()