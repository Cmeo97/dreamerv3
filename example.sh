#!/bin/bash

#SBATCH --job-name=dreamerv3
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=120G                                     


#conda_env=${1}

# 1. Load the required modules
module load anaconda/3
module load cuda/11.2
conda activate ~/.conda/envs/tf211-jax044-py310
#conda activate ~/.conda/envs/mamba/envs/ECS
#source  ~/.venvs/${conda_env}/bin/activate

env=$1
task=$2
seed=$3

echo 'Running experimnet - env: ' ${env}, 'task: ' ${task}, 'seed: ' ${seed}

python dreamerv3/train.py \
  --logdir ~/scratch/dreamerv3/logdir/${env}/${task}/${seed} \
  --configs ${env} \
  --task ${task} \
> logs_training/dreamer_training_"${task}""-"$(date +%Y%m%d-%H%M%S).out 2> logs_training/dreamer_training_"${task}""-"$(date +%Y%m%d-%H%M%S).err
