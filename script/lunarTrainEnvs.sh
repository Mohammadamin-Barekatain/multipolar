#!/usr/bin/env bash

cd ..

algo=$1
prefix_exp_name=$2
log=$3


if [ $# -ne 3 ]
  then
    echo "3 arguments must be provided"
    exit
fi


for main_engine_power in 10 13 18 23 28 33 38
do
    for scale in 25 30 35 40 45
    do
        for seed in 1000 2000 3000
        do
            exp_name="$prefix_exp_name-enginePower$main_engine_power-scale$scale"
            cmd="python train.py --env LunarLanderContinuous-v2 --algo $algo \
            --seed $seed --exp-name $exp_name --log-folder $log \
            --MAIN_ENGINE_POWER $main_engine_power --SCALE $scale"

            if [ ! -d "$log/$algo/LunarLanderContinuous-v2_$exp_name"_1"" ]; then
                echo $cmd
            fi

        done
    done
done