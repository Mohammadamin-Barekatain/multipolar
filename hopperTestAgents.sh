#!/usr/bin/env bash

algo=$1
prefix_exp_name=$2
log=$3
n_envs=$4

if [ $# -ne 4 ]
  then
    echo "4 arguments must be provided"
    exit
fi


for i in 0.41,0.3 0.44,0.33 0.47,0.36 0.5,0.39 0.53,0.42
do

    IFS=',' read s_leg s_foot <<< "${i}"

    for s_size in  0.75 0.8 0.85 0.9 0.95 1
    do
        for s_damping in 0.5 1 1.5 2 2.5
        do
            for i in 0.41,0.3 0.44,0.33 0.47,0.36 0.5,0.39 0.53,0.42
            do

                IFS=',' read leg foot <<< "${i}"

                for size in  0.75 0.8 0.85 0.9 0.95 1
                do
                    for damping in 0.5 1 1.5 2 2.5
                    do
                        exp_name="leg$leg-foot$foot-size$size-damp$damping"
                        exp_name=${exp_name//.}

                        train_name="$prefix_exp_name-leg$s_leg-foot$s_foot-size$s_size-damp$s_damping"
                        train_name=${train_name//.}


                        cmd="python test.py \
                            --trained-agent $log/$algo/RoboschoolHopper-v1_""$train_name""_1/RoboschoolHopper-v1.pkl \
                            --exp-name $exp_name \
                            --n-envs $n_envs \
                            --play 1500\
                            --leg_length $leg --foot_length $foot --size $size --damping $damping "

                        $cmd

                    done
                done
            done
        done
    done
done

