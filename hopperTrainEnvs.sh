#!/usr/bin/env bash

algo="ppo2"
prefix_exp_name=""


for i in 0.4,0.33 0.45,0.36 0.5,0.39 0.55,0.42 0.6,0.45
do

    IFS=',' read leg foot <<< "${i}"

    for size in 0.5 0.75 1 1.25 1.5
    do
        for damping in 0.25 0.5 1 2 4
        do
            for seed in 1000 2000 3000
            do
                exp_name="$prefix_exp_name"leg"$leg"-foot"$foot-"size"$size-"damp"$damping"
                exp_name=${exp_name//.}
                cmd="python train.py --env RoboschoolHopper-v1 --no-tensorboard --play 2000\
                --algo $algo --seed $seed --exp-name $exp_name \
                --leg_length $leg --foot_length $foot --size $size --damping $damping"
                echo $cmd
            done
        done
    done
done

