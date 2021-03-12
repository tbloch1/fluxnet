#!/bin/bash

# Accounts
#SBATCH --account=stfc_cg
#SBATCH --partition=gpu_limited #project

# CPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# GPUs
#SBATCH --gres=gpu:V100:1

# Memory
#SBATCH --mem-per-cpu=8000

# Time limit
#SBATCH --time=12:00:00

# Admin details
#SBATCH --job-name=Training
#SBATCH --output=training_ouput.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.bloch@pgr.reading.ac.uk
#SBATCH --nodes=1-1

# Diagnostics
hostname
echo 
nvidia-smi
echo
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
echo


module load python3/anaconda/5.1.0
source activate tbenv7
python training_bnn.py \
--model 'MultiheadBNN' \
--heads 'Multi' \
--target 'E_11' \
--mltlim 23 \
--params '800kevflux 2mevflux 800kevstd 2mevstd AE_INDEX AL_INDEX AU_INDEX SYM_D SYM_H ASY_D ASY_H mlt_sin mlt_cos mlt_sin_g mlt_cos_g date_sin date_cos' \
--epochs 20000 \
--optimiser 'Adam' \
--lr 0.002734713254258632 \
--loss 'MSELoss' \
--n_ensemble 50 \
--n_hidden 160

# Flux
# '800kevflux 2mevflux 800kevstd 2mevstd'

# Flux, mlt, year frac
# '800kevflux 2mevflux 800kevstd 2mevstd mlt_sin mlt_cos mlt_sin_g mlt_cos_g date_sin date_cos'

# Flux, indices
# '800kevflux 2mevflux 800kevstd 2mevstd AE_INDEX AL_INDEX AU_INDEX SYM_D SYM_H ASY_D ASY_H'

# Flux, indices, mlt, year frac
# '800kevflux 2mevflux 800kevstd 2mevstd AE_INDEX AL_INDEX AU_INDEX SYM_D SYM_H ASY_D ASY_H mlt_sin mlt_cos mlt_sin_g mlt_cos_g date_sin date_cos' 

# Flux, SW
# '800kevflux 2mevflux 800kevstd 2mevstd BX_GSE BY_GSM BZ_GSM Vx Vy Vz proton_density T Pressure'

# Flux, SW, mlt, year frac
# '800kevflux 2mevflux 800kevstd 2mevstd BX_GSE BY_GSM BZ_GSM Vx Vy Vz proton_density T Pressure mlt_sin mlt_cos mlt_sin_g mlt_cos_g date_sin date_cos'

# Flux, SW, indices
# '800kevflux 2mevflux 800kevstd 2mevstd BX_GSE BY_GSM BZ_GSM Vx Vy Vz proton_density T Pressure AE_INDEX AL_INDEX AU_INDEX SYM_D SYM_H ASY_D ASY_H'

# Flux, SW, indices, mlt, year frac
# '800kevflux 2mevflux 800kevstd 2mevstd BX_GSE BY_GSM BZ_GSM Vx Vy Vz proton_density T Pressure AE_INDEX AL_INDEX AU_INDEX SYM_D SYM_H ASY_D ASY_H mlt_sin mlt_cos mlt_sin_g mlt_cos_g date_sin date_cos'
