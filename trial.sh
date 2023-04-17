#!/bin/bash

declare -a All_Envs=(loconav)

declare -a All_Tasks=(loconav_ant_maze_xl loconav_ant_maze_m)


for Env in "${All_Envs[@]}"
do
	for Task in "${All_Tasks[@]}"
	do

        target='/home/mila/c/cristian.meo/scratch/dreamerv3/logdir/'${Env}'/'${Task}
        #echo $target
        pushd "$target" > /dev/null
        declare -a All_Files=(*)
        #echo ${All_Files[@]}
        workdir='/home/mila/c/cristian.meo/HRL/dreamerv3'
        pushd "$workdir" > /dev/null
        for f in "${All_Files[@]}"
        do
            sbatch example.sh $Env $Task $f   
        done
	done
done 


