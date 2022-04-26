# Instance Weighted Incremental Evolution Strategies (IW-IES)

This repo contains code accompanying the paper: [Zhi Wang, Chunlin Chen, and Daoyi Dong, "Instance Weighted Incremental Evolution Strategies for Reinforcement Learning in Dynamic Environments", IEEE Transactions on Neural Networks and Learning Systems, 2022.](https://ieeexplore.ieee.org/abstract/document/9744521/)
It contains code for running the incremental learning tasks, including 2D navigation, Swimmer, Hopper, and HalfCheetah domains. The basic reinforcement learning algorithms are implemented using natural evolution strategies.

### Dependencies
This code requires the following:
* python 3.5+
* pytorch 0.4+
* gym
* MuJoCo license

### Data
* For the 2D navigation domain, data is generated from `myrllib/envs/navigation.py`
* For the Swimmer/Hopper/HalfCheetah Mujoco domains, the modified Mujoco environments are in `myrllib/envs/mujoco/*`

### Usage 
* For example, in Case I of the navigation domain, just run the bash script `navi_v1_iwies.sh` to get the results of iwies and its ablation methods, also see the usage instructions in the script and `main.py`; just run the bash script `navi_v1_baselines.sh` to get the results of the baselines including Robust, Hist, SO-CMA, and ES-MAML, also see the usage instructions in the script and `baselines.py`
* When getting the results in `output/*/*.npy` files, plot the results using `data_process.py`. For example, the results for the navigation domains are as follows:

Case I | Case II | Complex Case
------------ | ------------- | -------------
![iwies results for Case I](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi1_iwies_onerun.png) | ![iwies results for Case II](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi2_iwies_onerun.png) | ![iwies results for Complex Case](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi3_iwies_onerun.png)
![baseline results for Case I](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi1_baselines_onerun.png) | ![baselines results for Case II](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi2_baselines_onerun.png) | ![baselines results for Complex Case](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi3_baselines_onerun.png)

Note that these results are from a single run of the code. You can randomly change the environment to a new one, and record the performance of all tested methods when adapting to the new environment. In our paper, we repeat the process ten times and report the mean and standard error to demonstrate the performance for learning in stochastic dynamic environments. For example, the results for Case I of navigation domain and the swimmer domain are as follows:

navigation_v1 | swimmer
------------ | -------------
![experimental results for navigation_v1 domain](https://github.com/HeyuanMingong/iwies/blob/master/exp/navi1.png) | ![experimental results for half cheetah domain](https://github.com/HeyuanMingong/iwies/blob/master/exp/swimmer.png)

Also, the results for other demo scripts are shown in `exp/*`

### Contact 

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/iwies/issues), or email to zhiwang@nju.edu.cn.
 




