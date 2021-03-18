import numpy as np
import pandas as pd
import datetime as dt
from scipy.sparse import data
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
import helpers
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
fpath_o = '/storage/silver/stfc_cg/hf832176/data/OMNI/'

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

omni = pd.read_parquet(fpath_o+'1_min/2000_2020_nan.parquet')
omni_h =  pd.read_parquet(fpath_o+'omni_H_2000_2020.parquet')
omni_h = omni_h[['R1800','F10_INDEX1800']]
omni_h.columns = ['ssn','f107']
omni = omni.join(omni_h,how='left')

def omni_goes_align(omni,goes,parameters):
    omni_params = [i for i in parameters if i in omni.columns]
    omni = omni[omni_params].interpolate()
    omni = omni.interpolate('ffill')
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
def mlt_conjunction(themis,goes,mlt_lim,scname1,scname2):
    th_g = themis.join(goes,how='inner',rsuffix='_g')
    th_g['mlt_diff'] = [i if i <= 12 else np.abs(i-24)
                        for i in np.abs(th_g.mlt-th_g.mlt_g)]
    th_g = th_g[th_g['mlt_diff'] < mlt_lim]
    th_g['mlt_sin'] = np.sin(th_g.mlt)
    th_g['mlt_cos'] = np.cos(th_g.mlt)
    th_g['mlt_sin_g'] = np.sin(th_g.mlt_g)
    th_g['mlt_cos_g'] = np.cos(th_g.mlt_g)

    th_g['datetime'] = th_g.index

    th_g['year_frac'] = [date_to_frac(date) for date in th_g.index]
    th_g['date_sin'] = np.sin(th_g.year_frac)
    th_g['date_cos'] = np.cos(th_g.year_frac)

    th_g['day_frac'] = [date_to_tod(date) for date in th_g.index]
    th_g['day_sin'] = np.sin(th_g.day_frac)
    th_g['day_cos'] = np.cos(th_g.day_frac)

    th_g['tha'] = np.ones(len(th_g)) if scname1 == 'tha' else np.zeros(len(th_g))
    th_g['thb'] = np.ones(len(th_g)) if scname1 == 'thb' else np.zeros(len(th_g))
    th_g['thc'] = np.ones(len(th_g)) if scname1 == 'thc' else np.zeros(len(th_g))
    th_g['thd'] = np.ones(len(th_g)) if scname1 == 'thd' else np.zeros(len(th_g))
    th_g['the'] = np.ones(len(th_g)) if scname1 == 'the' else np.zeros(len(th_g))

    th_g['g13'] = np.ones(len(th_g)) if scname2 == 'g13' else np.zeros(len(th_g))
    th_g['g14'] = np.ones(len(th_g)) if scname2 == 'g14' else np.zeros(len(th_g))
    th_g['g15'] = np.ones(len(th_g)) if scname2 == 'g15' else np.zeros(len(th_g))

    print(th_g.shape)
    return th_g


th_g = []
for themis,scname1 in zip([tha,thb,thc,thd,the],['tha','thb','thc','thd','the']):
    for goes,scname2 in zip([g13,g14,g15],['g13','g14','g15']):
        th_g.append(mlt_conjunction(themis,goes,mltlim,scname1,scname2))

thgdf = pd.concat(th_g, ignore_index=True).copy()
print('Final Dataset Size: ',thgdf.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
devstr = 'cuda' if torch.cuda.is_available() else 'cpu'

target_parameters = [target]
if heads == 'Multi':
    target_parameters = ['E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'E_6',
                         'E_7', 'E_8', 'E_9', 'E_10', 'E_11']


trainset, valset, testset = helpers.timeseries_sample(thgdf)

# plt.figure(figsize=(16,4.5),dpi=500)
# plt.plot(testset[testset['the']==1].sort_values('seconds').seconds,
#          testset[testset['the']==1].sort_values('seconds').r)
# plt.savefig('graphs/radial_pos.pdf')
# import pdb; pdb.set_trace()

Xscaler = StandardScaler().fit(trainset[parameters].values)
yscaler = StandardScaler().fit(np.log10(trainset[target_parameters]).values)

Xtrain_i = torch.from_numpy(Xscaler.transform(trainset[parameters].values))
ytrain_i = torch.from_numpy(yscaler.transform(np.log10(trainset[target_parameters]).values))

Xval_i = torch.from_numpy(Xscaler.transform(valset[parameters].values))
yval_i = torch.from_numpy(yscaler.transform(np.log10(valset[target_parameters]).values))

Xtest_i = torch.from_numpy(Xscaler.transform(testset[parameters].values))
ytest_i = torch.from_numpy(yscaler.transform(np.log10(testset[target_parameters]).values))

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
test_sigmas = []

for i in range(n_ensembles):
    random_idx = torch.randint(0,len(Xtrain_i),
                               (len(Xtrain_i),))

    Xtrain = Xtrain_i[random_idx]
    ytrain = ytrain_i[random_idx]

    # if devstr == 'cuda':
    Xtrain = Xtrain.to(device)
    ytrain = ytrain.to(device)
    Xval = Xval_i.to(device)
    yval = yval_i.to(device)
    Xtest = Xtest_i.to(device)
    ytest = ytest_i.to(device)

        
    net = model(n_feature=len(parameters), n_hidden=n_hidden, n_output=11)
    net = net.to(device)

    # wandb.watch(net)

    optimizer = optimiser(net.parameters(), lr=lr)
    loss_func = helpers.negative_log_likelihood

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

    MIN_SIGMA = 1e-04
    MAX_SIGMA = 10

    for epoch in range(n_epochs+1):
        outputs = net(Xtrain.float())
        prediction = outputs[:,:,0].view(-1,11)
        sigma = outputs[:,:,1].view(-1,11)

        sigma = sigma.to(device).float()
        sigma = MIN_SIGMA + (MAX_SIGMA - MIN_SIGMA) * torch.sigmoid(sigma)

        valout = net(Xval.float())
        valpred = valout[:,:,0].view(-1,11)
        val_sigma = valout[:,:,1].view(-1,11)
        val_sigma = MIN_SIGMA + (MAX_SIGMA - MIN_SIGMA) * torch.sigmoid(val_sigma)

        trainloss = loss_func(ytrain.float(),
                              prediction.float(),
                              sigma.float()).mean()
        
        loss_report = loss_func(ytrain.float(),
                                prediction.float(),
                                sigma.float())
        valloss = loss_func(yval.float(),
                            valpred.float(),
                            val_sigma.float())
        
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
                       "Val Loss": valloss.mean().data.cpu().numpy()})
        else:
            wandb.log({'Epoch': epoch,
                       "Train Loss": loss_report.max().data.numpy(),
                       "Val Loss": valloss.mean().data.numpy()})
        
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

    testout = eval_model(Xtest.float())
    testpred = testout[:,:,0].view(-1,11)
    test_sigma = testout[:,:,1].view(-1,11)
    test_sigma = test_sigma.to(device).float()
    test_sigma = MIN_SIGMA + (MAX_SIGMA - MIN_SIGMA) * torch.sigmoid(test_sigma)
    testloss = (loss_func(ytest.float(), testpred.float(),
                            test_sigma.float()))


    if devstr == 'cuda':
        testpred1 = yscaler.inverse_transform(testpred.data.cpu().numpy())
        # test_sigma1 = yscaler.inverse_transform(test_sigma.data.cpu().numpy())
        test_sigma1 = (test_sigma.data.cpu().numpy()
                       * np.log10(trainset[target_parameters]).values.std(axis=0))
        ytest_np = yscaler.inverse_transform(np.exp(ytest.data.cpu().numpy()))

        test_predictions.append(testpred1)
        test_sigmas.append(test_sigma1)
    else:
        testpred1 = yscaler.inverse_transform(testpred.data.numpy())
        # test_sigma1 = yscaler.inverse_transform(test_sigma.data.numpy())
        test_sigma1 = (test_sigma.data.numpy()
                       * np.log10(trainset[target_parameters]).values.std(axis=0))
        ytest_np = yscaler.inverse_transform(np.exp(ytest.data.numpy()))

        test_predictions.append(testpred1)
        test_sigmas.append(test_sigma1)


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

ytestset = np.log10(testset[target_parameters].values)
test_predictions = np.array(test_predictions)
test_sigmas = np.array(test_sigmas)
ens_mean = test_predictions.mean(axis=0)
ens_std = test_predictions.std(axis=0)
ens_uqt = np.quantile(test_predictions,0.75,axis=0)
ens_lqt = np.quantile(test_predictions,0.25,axis=0)

ens_var = ens_std**2
ens_pred_var_mean = (test_sigmas**2).mean(axis=0)

# Law of total variance:
# Var(Y) = E[var(y|x)] + var(E[y|x])
# Total var = the mean of the variances + the variance of the means
total_variance = ens_pred_var_mean + ens_var

# import pdb; pdb.set_trace()

# reference prediction
y800kev = np.log10(testset['800kevflux'].values)
y800kev = np.transpose([y800kev]*11)

#region Plot reliability diagram per energy channel

plt.figure(figsize=(16,9),dpi=400)
axs = [plt.subplot(3,4,i+1) for i in range(11)]

[axs[i].scatter(ens_mean[:,i],ytestset[:,i],
                s=1,c='C0')
 for i in range(11)]

[i.plot([3,8],[4,8], c='k', linestyle='-', linewidth=1) for i in axs]

[axs[i].set_title('Energy Channel: {0}'.format(i+1)) for i in range(11)]
[i.set_ylabel('True Flux,\n'
              +'$eVcm^{-2}s^{-1}sr^{-1}eV^{-1}$') for i in axs[::4]]
[i.set_xlabel('Predicted Flux,\n'
              +'$eVcm^{-2}s^{-1}sr^{-1}eV^{-1}$') for i in axs[-3:]]
# [i.set_yscale('log') for i in axs]
# [i.set_xscale('log') for i in axs]
[ax.set_xlim(0.99*ens_mean[:,i].min(), 1.01*ens_mean[:,i].max())
 for i,ax in zip(range(11),axs)]
[ax.set_ylim(0.99*ytestset[:,i].min(), 1.01*ytestset[:,i].max())
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
            mse_dim(ens_mean,ytestset))
# RMSE
axs[1].plot(np.arange(11)+1,
            rmse_dim(ens_mean,ytestset))
# MAE
axs[2].plot(np.arange(11)+1,
            mae_dim(ens_mean,ytestset))
# MAPE
axs[3].plot(np.arange(11)+1,
            mape_dim(ens_mean,ytestset))
# SMAPE
axs[4].plot(np.arange(11)+1,
            smape_dim(ens_mean,ytestset))
# Spearman's Rank
axs[5].plot(np.arange(11)+1,
            [scipy.stats.spearmanr(ens_mean[:,i],ytestset[:,i])[0]
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
            skill_score(mse_dim(y800kev,ytestset),
                        mse_dim(ens_mean,ytestset)))
# RMSE
axs[1].plot(np.arange(11)+1,
            skill_score(rmse_dim(y800kev,ytestset),
                        rmse_dim(ens_mean,ytestset)))
# MAE
axs[2].plot(np.arange(11)+1,
            skill_score(mae_dim(y800kev,ytestset),
                        mae_dim(ens_mean,ytestset)))
# MAPE
axs[3].plot(np.arange(11)+1,
            skill_score(mape_dim(y800kev,ytestset),
                        mape_dim(ens_mean,ytestset)))
# SMAPE
axs[4].plot(np.arange(11)+1,
            skill_score(smape_dim(y800kev,ytestset),
                        smape_dim(ens_mean,ytestset)))
# Spearman's Rank
ref_sr = [scipy.stats.spearmanr(y800kev[:,i],
                                ytestset[:,i])[0]
           for i in range(11)]
pred_sr = [scipy.stats.spearmanr(ens_mean[:,i],
                                 ytestset[:,i])[0]
           for i in range(11)]

axs[5].plot(np.arange(11)+1,
            skill_score(np.array(ref_sr), np.array(pred_sr)))

# [i.set_yscale('log') for i in axs[:3]]
[i.set_title('Metric: '+j)
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
crps = [np.mean(ps.crps_ensemble(ytestset[:,i],
                                 test_preds_T[:,:,i]))
        for i in range(11)]
crps_ref = [np.mean(ps.crps_ensemble(ytestset[:,i],
                                     y800kev[:,i]))
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

sns.violinplot(x = [i+1 for i in range(ens_mean.shape[1])]*len(ens_mean)*2,
               y = np.concatenate([ens_mean.flatten(),
                                   ytestset.flatten()]),
               hue = np.concatenate([['Predicted']*len(ens_mean.flatten()),
                                     ['True']*len(ens_mean.flatten())]),
               palette="muted", split=True, cut=0)

# plt.yscale('log')

plt.title('Distributions of True and Predicted Fluxes')
plt.ylabel('Log$_{10}$ Flux,\n'+'$eVcm^{-2}s^{-1}sr^{-1}eV^{-1}$')
plt.xlabel('Energy Channel')

plt.tight_layout()
model_folder = 'models/{0:05d}_ensemble/'.format(folder_number)
plt.savefig(savepath+model_folder+'Distributions.pdf')
plt.close()
#endregion

testset2 = testset.copy()
testset2.index = testset2.datetime
testset2[target_parameters] = np.log10(testset2[target_parameters])

for i in range(11):
    testset2['pred_mu_'+str(i+1)] = ens_mean[:,i]
    # testset2['pred_var_'+str(i)] = ens_pred_var_mean[:,i]
    # testset2['ens_mu_var_'+str(i)] = ens_var[:,i]
    testset2['pred_total_var_'+str(i+1)] = total_variance[:,i]

# Number of samples per day in the groupby
group_counts = testset2.groupby(pd.Grouper(freq='D')).sum()['the']
group_mask = group_counts > 50

# Average values from groupby
testset2_g = testset2.groupby(pd.Grouper(freq='D')).mean()

fig = plt.figure(figsize=(8,8))
axs = [plt.subplot(6,2,i+1) for i in range(11)]

data_plot = testset2_g[group_mask][(testset2_g.seconds>1.46e+9)
                                   & (testset2_g.seconds<1.469e+9)]


# Plotting predictions and total standard deviation
[i.errorbar(x = data_plot.seconds,
            y = data_plot['pred_mu_'+str(j+1)],
            yerr = data_plot['pred_total_var_'+str(j+1)]**0.5,
            c='C1',fmt='o',capsize=3,linewidth=2,markersize=4)
 for i,j in zip(axs,range(11))]

[i.scatter(data_plot.seconds, data_plot['E_'+str(j+1)],
           s=2,c='C0',marker='x',zorder=1000)
 for i,j in zip(axs,range(11))]

[i.fill_between(data_plot.seconds,
                data_plot['E_'+str(j+1)].values+np.log10(1/5),
                data_plot['E_'+str(j+1)].values+np.log10(5),
                color='grey',alpha=1/3,
                edgecolor=(0.5019607843137255, 0.5019607843137255,
                           0.5019607843137255, 1/3))
 for i,j in zip(axs,range(11))]

[i.set_title('Energy Channel: '+str(j+1))
 for i,j in zip(axs,range(11))]

[i.set_ylabel('Log$_{10}$(Flux)',size=8)
 for i in axs[::2]]
# [i.set_xlabel('Date') for i in axs[-2:]]
[i.set_ylim(4.75,7.75) for i in axs]


xticks = np.linspace(data_plot.seconds.min(),
                     data_plot.seconds.max(), 5)
xticklabels = [dt.datetime(1970,1,1)+dt.timedelta(seconds=i)
               for i in xticks]
xticklabels = [i.date().strftime('%d/%m/%y') for i in xticklabels]

[i.set_xticks(xticks) for i in axs]
axs[9].set_xticklabels(xticklabels)
axs[10].set_xticklabels(xticklabels)
# axs[9].xaxis.set_tick_params(rotation=45)
# axs[10].xaxis.set_tick_params(rotation=45)

[i.set_xticklabels([]) for i in axs[:9]]

axs[0].scatter([],[],s=6,c='C1',label='Mean Prediction + Uncertainty')
axs[0].scatter([],[],s=6,c='C0',label='Mean Measured Flux')
axs[0].scatter([],[],s=50,marker='s',c='gray',alpha=0.5,
               label='Factor of 5 Envelope')

plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
order = [0,1,2]
axs[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
              loc='center', ncol=1, frameon=False,
              bbox_to_anchor=[0.75,0.09],bbox_transform=fig.transFigure)

plt.savefig(savepath+model_folder+'timeseries_test.pdf')
plt.close()

sys.exit()