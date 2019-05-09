#!/usr/bin/env bash

for mep in 10 13 18 23 28 33 38
do
    for scl in 25 30 35 40 45
    do
        for main_engine_power in 10 13 18 23 28 33 38
        do
            for scale in 25 30 35 40 45
            do
                for ind in 1 2 3
                do
                    exp_name="$main_engine_power-$scale_$ind"
                    train_name="$mep-$scl"
                    cmd="python test.py --trained-agent logs/ppo2/LunarLanderContinuous-v2_""$train_name""_1/LunarLanderContinuous-v2.pkl \
                     --exp-name $exp_name --MAIN_ENGINE_POWER $main_engine_power --SCALE $scale --n-envs 40"
                    echo $cmd
                done
            done
        done
    done
done