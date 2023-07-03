#!/bin/bash

#SBATCH --job-name=dreamerv3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1                                   
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=2-23:59:00
#SBATCH --mem=40G      

# 1. Load the required modules
module load cuda/11.7
conda activate directorv2

env=$1
task=$2
config=$3
f=$4

python dreamerv3/train.py \
  --exp_config ${config} \
  --logdir ~/scratch/directorv2/${config}/logdir/${env}/${task}/${f}-$(date +%Y%m%d-%H%M%S) \
  --configs ${env} \
  --task ${task} \
> logs_training/dreamer_training_"${task}"_"${config}""-"$(date +%Y%m%d-%H%M%S).out 2> logs_training/dreamer_training_"${task}"_"${config}""-"$(date +%Y%m%d-%H%M%S).err
