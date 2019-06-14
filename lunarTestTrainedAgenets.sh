#!/usr/bin/env bash

#algo="ppo2"
#n_envs=40

algo="sac"
n_envs=1

for mep in 10 13 18 23 28 33 38
do
    for scl in 25 30 35 40 45
    do
        for main_engine_power in 10 13 18 23 28 33 38
        do
            for scale in 25 30 35 40 45
            do

                exp_name="$main_engine_power-$scale"
                train_name="$mep-$scl"
                cmd="python test.py --trained-agent logs/$algo/LunarLanderContinuous-v2_""$train_name""_1/LunarLanderContinuous-v2.pkl \
                 --exp-name $exp_name --MAIN_ENGINE_POWER $main_engine_power --SCALE $scale --n-envs $n_envs"
                echo $cmd

            done
        done
    done
done