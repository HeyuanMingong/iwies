#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:52:14 2018

@author: qiutian
"""
from gym.envs.registration import register

# 2D Navigation
# ----------------------------------------

register(
    'Navigation2D-v1',
    entry_point='envs.navigation:Navigation2DEnvV1',
    max_episode_steps=100
)

register(
        'Navigation2D-v2',
        entry_point='envs.navigation:Navigation2DEnvV2',
        max_episode_steps=100 
        )

register(
        'Navigation2D-v3',
        entry_point='envs.navigation:Navigation2DEnvV3',
        max_episode_steps=100 
        )

register(
        'AntVel-v1',
        entry_point = 'envs.mujoco.ant:AntVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.ant:AntVelEnv'},
        max_episode_steps = 100 
        )

register(
        'HalfCheetahVel-v1',
        entry_point = 'envs.mujoco.half_cheetah:HalfCheetahVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
        max_episode_steps = 100 
        )

register(
        'HopperVel-v1',
        entry_point = 'envs.mujoco.hopper:HopperVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.hopper:HopperVelEnv'},
        max_episode_steps = 100 
        )

register(
        'SwimmerVel-v1',
        entry_point = 'envs.mujoco.swimmer:SwimmerVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.hopper:HopperVelEnv'},
        max_episode_steps = 100 
        )

