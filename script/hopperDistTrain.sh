#!/usr/bin/env bash

cd ..

algo=$1
prefix_exp_name=$2
IFS=',' read leg_min leg_max <<< "${3}"
IFS=',' read foot_min foot_max <<< "${4}"
IFS=',' read size_min size_max <<< "${5}"
IFS=',' read damp_min damp_max <<< "${6}"


if [ $# -ne 6 ]
  then
    echo "6 args must be provided"
    exit
fi



exp_name="meta-$prefix_exp_name-leg$leg_min-$leg_max-foot$foot_min-$foot_max-size$size_min-$size_max-damp$damp_min-$damp_max"
exp_name=${exp_name//.}

for seed in 1000 2000 3000
do
    cmd="python train.py --env RoboschoolHopper-v1 --algo $algo --seed $seed \
     --exp-name $exp_name -n 20000000 \
     --params-ranges leg_length,$leg_min,$leg_max foot_length,$foot_min,$foot_max size,$size_min,$size_max damping,$damp_min,$damp_max"

    echo $cmd
done



