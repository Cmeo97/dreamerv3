#declare -a All_Envs=(loconav)
#
#declare -a All_Tasks=(loconav_ant_maze_m)
#
#declare -a All_Seeds=(1 2 3)
#
#
#for Env in "${All_Envs[@]}"
#do
#	for Task in "${All_Tasks[@]}"
#	do
#        for seed in "${All_Seeds[@]}"
#        do
#            sbatch example.sh $Env $Task $seed
#		done
#	done
#done 
#
#
#
declare -a All_Envs=(pinpad)

declare -a All_Tasks=(pinpad_five)

declare -a All_Configs=(director)

declare -a All_Seeds=(4)


for Env in "${All_Envs[@]}"
do
	for Task in "${All_Tasks[@]}"
	do
		for config in "${All_Configs[@]}"
		do
        	for seed in "${All_Seeds[@]}"
        	do
        	    sbatch train_multi.sh $Env $Task $config $seed
			done
		done
	done
done 
#
#
#
declare -a All_Envs=(loconav)

declare -a All_Tasks=(loconav_ant_maze_m)

declare -a All_Configs=(director)

declare -a All_Seeds=(4)


for Env in "${All_Envs[@]}"
do
	for Task in "${All_Tasks[@]}"
	do
		for config in "${All_Configs[@]}"
		do
        	for seed in "${All_Seeds[@]}"
        	do
        	    sbatch train_multi.sh $Env $Task $config $seed
			done
		done
	done
done 
#


declare -a All_Envs=(dmc_vision)

declare -a All_Tasks=(dmc_walker_walk)

declare -a All_Configs=(director)

declare -a All_Seeds=(1 2 3)


for Env in "${All_Envs[@]}"
do
	for Task in "${All_Tasks[@]}"
	do
		for config in "${All_Configs[@]}"
		do
        	for seed in "${All_Seeds[@]}"
        	do
        	    sbatch train_multi.sh $Env $Task $config $seed
			done
		done
	done
done


declare -a All_Envs=(loconav)

declare -a All_Tasks=(loconav_ant_maze_m)

declare -a All_Configs=(directorV2 directorV2-rssm directorV2-symlog directorV2-fb directorV2-klb directorV2-perc directorV2-buckets)

declare -a All_Seeds=(0)


for Env in "${All_Envs[@]}"
do
	for Task in "${All_Tasks[@]}"
	do
		for config in "${All_Configs[@]}"
		do
        	for seed in "${All_Seeds[@]}"
        	do
        	    sbatch train_multi.sh $Env $Task $config $seed
			done
		done
	done
done 




