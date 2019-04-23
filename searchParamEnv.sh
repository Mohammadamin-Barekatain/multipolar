#!/usr/bin/env bash

for main_engine_power in 1 7 13 26 39
do
    for scale in 10 20 30 40 50
    do
        for seed in 1000 2000 3000
        do
            exp_name="$main_engine_power-$scale"
            cmd="python train.py --env LunarLanderContinuous-v2 --algo ppo2 --seed $seed --exp-name $exp_name --MAIN_ENGINE_POWER $main_engine_power --SCALE $scale --play 1000"
            `$cmd`
        done
    done
done