#!/bin/bash

# Accounts
#SBATCH --account=stfc_cg
#SBATCH --partition=gpu_limited

# CPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# GPUs
#SBATCH --gres=gpu:T4:1

# Memory
#SBATCH --mem-per-cpu=8000

# Time limit
#SBATCH --time=12:00:00

# Admin details
#SBATCH --job-name=Training_sweep_idx
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
wandb init -p fluxnet
wandb agent tbloch/fluxnet/4ovbzuw3