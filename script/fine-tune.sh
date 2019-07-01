#!/usr/bin/env bash

#algo="ppo2"
#video_interval=10

cd ..

algo="sac"
video_interval=400000
n=300000

for env_mep in 10 13 18 23 28 33 38
do
    for env_scale in 25 30 35 40 45
    do
        for main_engine_power in 10 13 18 23 28 33 38
        do
            for scale in 25 30 35 40 45
            do
                exp_name="$env_mep-$env_scale-from-$main_engine_power-$scale"
                trained_agent="logs/sac/LunarLanderContinuous-v2_$main_engine_power-$scale"_1"/LunarLanderContinuous-v2.pkl"

                cmd="python train.py --env LunarLanderContinuous-v2 --algo $algo --no-tensorboard -n $n \
                --play 500 --save_video_interval $video_interval --save_video_length 300 --seed 1000 --exp-name $exp_name \
                --MAIN_ENGINE_POWER $env_mep --SCALE $env_scale --trained-agent $trained_agent"

                echo $cmd

            done
        done
    done
done