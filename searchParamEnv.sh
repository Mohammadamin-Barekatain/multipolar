#!/usr/bin/env bash

#algo="ppo2"
#video_interval=10

algo="sac"
video_interval=400000
prefix_exp_name="mlap-stateDep"


for main_engine_power in 10 13 18 23 28 33 38
do
    for scale in 25 30 35 40 45
    do
        for seed in 1000 2000 3000
        do
            exp_name="$prefix_exp_name$main_engine_power-$scale"
            cmd="python train.py --env LunarLanderContinuous-v2 --algo $algo -n 1000000 \
            --play 1000 --save_video_interval $video_interval --save_video_length 300 --seed $seed --exp-name $exp_name \
            --MAIN_ENGINE_POWER $main_engine_power --SCALE $scale"
            echo $cmd
        done
    done
done