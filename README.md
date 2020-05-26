# Instance Weighted Incremental Evolution Strategies (IW-IES)

This repo contains code accompaning the paper: [Zhi Wang, Chunlin Chen, and Daoyi Dong, "Instance Weighted Incremental Evolution Strategies for Reinforcement Learning in Dynamic Environments", submitted.]()
It contains code for running the incremental learning tasks, including 2D navigation, Swimmer, Hopper, and HalfCheetah domains. The basic reinforcement learning algorithms are implemented using natrural evolution strategies.

### Dependencies
This code requires the following:
* python 3.5+
* pytorch 0.4+
* gym
* MuJoCo license

### Data
* For the 2D navigation domain, data is generated from `myrllib/envs/navigation.py`
* For the Swimmer/Hopper/HalfCheetah Mujoco domains, the modified Mujoco enviornments are in `myrllib/envs/mujoco/*`

### Usage 
* For example, to run the code in the Ant domain, just run the bash script `swimmer.sh`, also see the usage instructions in the script and `main.py`.
* When getting the results in `output/*/*.npy` files, plot the results using `data_process.py`. For example, the results for `navigation_v1.sh` and `swimmer.sh` are as follow:

navigation_v1 | swimmer
------------ | -------------
![experimental results for navigation_v1 domain](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi1.png) | ![experimental results for half cheetah domain](https://github.com/HeyuanMingong/iwies/blob/master/exp/swimmer.png)

Also, the results for other demo scripts are shown in `exp/*`

### Contact 
For safety reasons, the source code is coming soon.

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/iwies/issues), or email to zhiwang@nju.edu.cn.
 




