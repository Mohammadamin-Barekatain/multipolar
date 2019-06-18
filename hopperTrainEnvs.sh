#!/usr/bin/env bash

algo=$1
prefix_exp_name=$2
log=$3


if [ $# -ne 3 ]
  then
    echo "3 arguments must be provided"
    exit
fi

for i in 0.41,0.3 0.44,0.33 0.47,0.36 0.5,0.39 0.53,0.42
do

    IFS=',' read leg foot <<< "${i}"

    for size in  0.75 0.8 0.85 0.9 0.95 1
    do
        for damping in 0.5 1 1.5 2 2.5
        do
            for seed in 1000 2000 3000
            do
                exp_name="$prefix_exp_name"leg"$leg"-foot"$foot-"size"$size-"damp"$damping"
                exp_name=${exp_name//.}
                cmd="python train.py --env RoboschoolHopper-v1 --no-tensorboard \
                --algo $algo --seed $seed --exp-name $exp_name --log-folder $log \
                --leg_length $leg --foot_length $foot --size $size --damping $damping"

                if [ ! -d "$log/$algo/RoboschoolHopper-v1_$exp_name"_1"" ]; then
                    echo $cmd
                fi
            done
        done
    done
done

