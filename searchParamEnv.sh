#!/usr/bin/env bash

for main_engine_power in 10 13 18 23 28 33 38
do
    for scale in 25 30 35 40 45
    do
        for seed in 1000 2000 3000
        do
            exp_name="$main_engine_power-$scale"
            cmd="python train.py --env LunarLanderContinuous-v2 --algo ppo2 --seed $seed --exp-name $exp_name \
            --play 1000 --save_video_interval 10 --save_video_length 500 \
             --MAIN_ENGINE_POWER $main_engine_power --SCALE $scale"
            echo $cmd
        done
    done
done