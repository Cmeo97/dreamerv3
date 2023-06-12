#!/bin/bash

#SBATCH --job-name=dreamerv3
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
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
config=$3
f=$4

python -m cProfile dreamerv3/train.py \
  --exp_config ${config} \
  --logdir ~/scratch/directorv2/${config}/logdir/${env}/${task}/${f}-$(date +%Y%m%d-%H%M%S) \
  --configs ${env} \
  --task ${task} \
> logs_training/dreamer_training_"${task}"_"${config}""-"$(date +%Y%m%d-%H%M%S).out 2> logs_training/dreamer_training_"${task}"_"${config}""-"$(date +%Y%m%d-%H%M%S).err
