#!/usr/bin/env bash


algo="mlap-sac"
prefix_exp_name="SDW-linear-no-bias-"


for i in 33,40 13,30 23,40 38,45
do
    IFS=',' read main_engine_power scale <<< "${i}"

    for seed in 1000 2000 3000
    do
        exp_name="$prefix_exp_name$main_engine_power-$scale"
        cmd="python train.py --env LunarLanderContinuous-v2 --algo $algo \
        --seed $seed --exp-name $exp_name \
        --MAIN_ENGINE_POWER $main_engine_power --SCALE $scale"
        echo $cmd
    done
done