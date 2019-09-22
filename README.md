# MULTIPOLAR: <br/> Multi-Source Policy Aggregation for Transfer Reinforcement Learning between Diverse Environmental Dynamics
#### [Mohammadamin Barekatain](http://barekatain.me), [Ryo Yonetani](https://yonetaniryo.github.io), [Masashi Hamaya](https://sites.google.com/view/masashihamaya/home)

Under review of ICLR 2020.

**This repository is only for ICLR reviews, and we do not permit any modifications and redistributions.**


# Introduction

This is a tensorflow-based implementation of our submission to ICLR 2020 titled *MULTIPOLAR: Multi-Source Policy Aggregation for Transfer Reinforcement Learning between Diverse Environmental Dynamics*.

Here, we propose MULTIPOLAR, a transfer RL method that leverages a set of source policies collected under unknown 
diverse environmental dynamics to efficiently learn a target policy in another dynamics.

This repository makes it possible to reproduce all of our experiments presented in the paper.

The code has been tested on **Ubuntu 16.04** as well as **macOS Mojave 10.14.06**.


# Installation

### Prerequisites
*  python3 (>=3.5)
*  TensorFlow (>=1.14.0)

### Ubuntu
```
sudo apt-get update && apt-get install swig cmake libopenmpi-dev zlib1g-dev ffmpeg
```
### Mac OS X

```
brew install cmake openmpi ffmpeg
```

### Install using pip

```
pip install stable-baselines==2.4.0 box2d box2d-kengz pyyaml pybullet==2.1.0 pytablewriter roboschool
pip install box2d-py gym==0.10.9
```

Make sure that `gym` version is correct: `gym==0.10.9`. 
Please see [Stable Baselines README](https://github.com/hill-a/stable-baselines) 
and [Rl baseline zoo](https://github.com/araffin/rl-baselines-zoo) for alternative installations.


### Testing the installations

To test the installations, first install `pytest`, and then:
```
python -m pytest -v tests/
```


# Training MULTIPOLAR



