# Instance Weighted Incremental Evolution Strategies (IW-IES)

This repo contains code accompaning the paper: [Zhi Wang, Chunlin Chen, and Daoyi Dong, "Instance Weighted Incremental Evolution Strategies for Reinforcement Learning in Dynamic Environments", submitted.]()
It contains code for running the incremental learning tasks, including 2D navigation, HalfCheetah, and Ant domains. The basic reinforcement learning algorithms are implemented using evolution strategies.

### Dependencies
This code requires the following:
* python 3.\*
* pytorch 0.4+
* gym
* MuJoCo license

### Data
* For the 2D navigation domain, data is generated from `myrllib/envs/navigation.py`
* For the Hopper/HalfCheetah/Ant Mujoco domains, the modified Mujoco enviornments are in `myrllib/envs/mujoco/*`

### Usage 
* For example, to run the code in the 2D Navigation domain with Case I dynamic environment, just run the bash script `./navigation_v1.sh`, also see the usage instructions in the script and `main.py`
* When getting the results in .mat files, plot the results using `data_process.py`. For example, the results for `./navigation_v1.sh` is as follow:
![experimental results for navigation domain](https://github.com/HeyuanMingong/iwies/blob/master/exp/demo_navigation_v1.png)

Also, the results for other demo scripts are shown in `exp/*`

### Contact 
For safety reasons, the source code is coming soon.

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/iwies/issues), or email to njuwangzhi@gmail.com.
 




