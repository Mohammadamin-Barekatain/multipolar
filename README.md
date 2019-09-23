# MULTIPOLAR: <br/> Multi-Source Policy Aggregation for Transfer Reinforcement Learning between Diverse Environmental Dynamics

[Mohammadamin Barekatain](http://barekatain.me), [Ryo Yonetani](https://yonetaniryo.github.io), [Masashi Hamaya](https://sites.google.com/view/masashihamaya/home)

Under review of ICLR 2020.

**This repository is only for ICLR reviews, and we do not permit any modifications and redistributions.**

# TL;DR
We propose MULTIPOLAR, a transfer RL method that leverages a set of source policies collected under unknown diverse environmental dynamics to efficiently learn an optimal target policy in another dynamics.

#### Source agents (Roboschool Ants with different leg designs)
<img src='assets/ant1.gif' width=120> <img src='assets/ant2.gif' width=120> <img src='assets/ant3.gif' width=120> <img src='assets/ant4.gif' width=120>

#### Target agent trained with MULTIPOLAR (left) vs. trained from scratch (right)
<img src='assets/multipolar-ant.gif' width=240> <img src='assets/MLP-ant.gif' width=240>

# Introduction

This is a TensorFlow-based implementation of our submission to ICLR 2020 titled *MULTIPOLAR: Multi-Source Policy Aggregation for Transfer Reinforcement Learning between Diverse Environmental Dynamics*. 

This repository makes it possible to reproduce all of our experiments presented in the paper.

The code has been tested on **Ubuntu 16.04** as well as **Mac OS Mojave 10.14.06**.


# Installation

### Prerequisites
*  python3 (>=3.5) with the development headers.
*  TensorFlow (>=1.14.0)

### Ubuntu
```
sudo apt-get update && apt-get install ssh swig cmake libopenmpi-dev python3-dev zlib1g-dev ffmpeg parallel libpcre3-dev libsm6 libxext6 libxrender-dev
```
### Mac OS X

```
brew install cmake openmpi ffmpeg parallel
```

### Install using pip

```
pip install stable-baselines==2.4.0 box2d box2d-kengz pyyaml pybullet==2.1.0 box2d-py gym==0.10.9 roboschool pytablewriter bootstrapped opencv-python
```

Make sure that `gym` version is correct: `gym==0.10.9`. 
Please see [Stable Baselines README](https://github.com/hill-a/stable-baselines) 
and [Rl baseline zoo](https://github.com/araffin/rl-baselines-zoo) for alternative installations.

If you're using Mac OS and have problem installing `pybullet`, use the following:

```
CFLAGS='-stdlib=libc++' pip install pybullet==2.1.0
```

### Testing the installations

To test the installations, first install `pytest`, and then:
```
python -m pytest -v tests/
```


# Training MULTIPOLAR in Acrobot

In this section, we present how to train MULTIPOLAR in Acrobot environment. Since our full experiments are computationally expensive, here we use 10 environment instances instead of 100 for both the baseline and MULTIPOLAR agents. Below commands will execute 3 trainings in parallel. This number must be configured based on the available number of CPUs.

1. Train the baseline agents with multi-layer perceptron (MLP) policy network three times (with different random seeds) in 10 randomly sampled environment instances. 

```
python random_env_dynamic_train_cmd_gen.py --num-samples 10 --algo ppo2 --seed 0 --env Acrobot-v1 \
--params-ranges LINK_LENGTH_1,0.3,1.3 LINK_LENGTH_2,0.3,1.3 LINK_MASS_1,0.5,1.5 LINK_MASS_2,0.5,1.5 LINK_COM_POS_1,0.05,0.95 LINK_COM_POS_2,0.05,0.95 LINK_MOI,0.25,1.5

parallel -a /tmp/out.txt --eta -j 3
```

We also use these trained MLPs as a pool of source policy candidates from which we sample to train MULTIPOLAR policies. Using the `Source_histogram.ipynb` notebook, it's possible to visualize the histogram of final episodic reward (average rewards of the last 100 training episodes) for these candidates similar to the Figure 5 of the paper.

2. For each environment instance, train 3 MULTIPOLAR policies with distinct sets of source policies of size 4 selected randomly from the pool of source policy candidates.

```
python train_multipolar_random_source.py --num-jobs 3 --sources-dir logs/ppo2/ --env Acrobot-v1 \
--algo multipolar-ppo2 --num-set 3 --num-sources 4 --num-subopt-sources 0 --num-samples 10 \
--params-ranges LINK_LENGTH_1,0.3,1.3 LINK_LENGTH_2,0.3,1.3 LINK_MASS_1,0.5,1.5 LINK_MASS_2,0.5,1.5 LINK_COM_POS_1,0.05,0.95 LINK_COM_POS_2,0.05,0.95 LINK_MOI,0.25,1.5
```

3. Repeat *step 2* with one source policy as another baseline policy.

```
python train_multipolar_random_source.py --num-jobs 3 --sources-dir logs/ppo2/ --env Acrobot-v1 \
--algo multipolar-ppo2 --num-set 3 --num-sources 1 --num-subopt-sources 0 --num-samples 10 \
--params-ranges LINK_LENGTH_1,0.3,1.3 LINK_LENGTH_2,0.3,1.3 LINK_MASS_1,0.5,1.5 LINK_MASS_2,0.5,1.5 LINK_COM_POS_1,0.05,0.95 LINK_COM_POS_2,0.05,0.95 LINK_MOI,0.25,1.5
```

4. Using `Sampling_efficiency.ipynb`, compare the sample efficiency of MULTIPOLAR(K=4) to the baselines similar to Table 1 of the paper.

5. Using `plot_learning_curves.ipynb`, plot the learning curves similar to Figure 3 of the paper.

# Training MULTIPOLAR in other environments

Below you can find the commands we used for performing our other experiments.



