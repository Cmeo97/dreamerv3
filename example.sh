#!/bin/bash

#SBATCH --job-name=dreamerv3
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=160G                                     


#conda_env=${1}

# 1. Load the required modules
module load anaconda/3
module load cuda/11.2
conda activate ~/.conda/envs/tf211-jax044-py310
#conda activate ~/.conda/envs/mamba/envs/ECS
#source  ~/.venvs/${conda_env}/bin/activate

env=$1
task=$2
f=$3

nohup python dreamerv3/train.py \
  --logdir ~/scratch/dreamerv3/logdir/${env}/${task}/${f} \
  --configs ${env} \
  --task ${task} \
> logs_training/dreamer_training_"${task}""-"$(date +%Y%m%d-%H%M%S).out 2> logs_training/dreamer_training_"${task}""-"$(date +%Y%m%d-%H%M%S).err
