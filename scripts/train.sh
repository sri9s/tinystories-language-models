#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=G-4GPU-32Cpu-235GB
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4

# activate conda env
conda activate tinystories


# set environment variables
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_SOCKET_IFNAME=^docker0,lo

# run script
srun python tinystories-language-models/src/train.py trainer=ddp
