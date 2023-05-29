#!/bin/bash

#SBATCH --job-name=dreamerv3
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=40G                                     


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
#010 runs bad optimization scheme, with new image_carry
#others run better optimization scheme, with new image_carry
nohup python dreamerv3/train.py \
  --logdir ~/scratch/directorv2/logdir/${env}/${task}/${f}-$(date +%Y%m%d-%H%M%S) \
  --configs ${env} \
  --task ${task} \
> logs_training/dreamer_training_"${task}""-"$(date +%Y%m%d-%H%M%S).out 2> logs_training/dreamer_training_"${task}""-"$(date +%Y%m%d-%H%M%S).err
