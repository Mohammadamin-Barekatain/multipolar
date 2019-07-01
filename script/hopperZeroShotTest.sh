#!/usr/bin/env bash

cd ..

trained_agent=$1
n_envs=$2

if [ $# -ne 2 ]
  then
    echo "2 arguments must be provided"
    exit
fi


for i in 0.41,0.3 0.44,0.33 0.47,0.36 0.5,0.39 0.53,0.42
do

    IFS=',' read leg foot <<< "${i}"

    for size in  0.75 0.8 0.85 0.9 0.95 1
    do
        for damping in 0.5 1 1.5 2 2.5
        do
            exp_name="leg$leg-foot$foot-size$size-damp$damping"
            exp_name=${exp_name//.}


            cmd="python test.py \
                --trained-agent $trained_agent \
                --exp-name $exp_name \
                --n-envs $n_envs \
                --play 1500\
                --leg_length $leg --foot_length $foot --size $size --damping $damping "

            $cmd

        done
    done
done

