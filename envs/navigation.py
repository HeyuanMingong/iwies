#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:22:59 2018

@author: qiutian
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import copy

class Navigation2DEnvV1(gym.Env):
    def __init__(self):
        super(Navigation2DEnvV1, self).__init__()
        self._goal = np.array([0.25,0.25], dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
                shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                shape=(2,), dtype=np.float32)
        self._state = np.zeros(2, dtype=np.float32)
        self.oracle = False

    def reset_oracle(self, oracle=False):
        if oracle:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(4,), dtype=np.float32)
            self._state = np.zeros(4, dtype=np.float32)
            self._state[-2:] = self._goal
            self.oracle = True
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(2,), dtype=np.float32)
            self._state = np.zeros(2, dtype=np.float32)
            self.oracle = False

    def reset(self):
        if self.oracle:
            self._state = np.zeros(4, dtype=np.float32)
            self._state[-2:] = self._goal
        else:
            self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        #assert self.action_space.contains(action)
        self._state[:2] = self._state[:2] + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = - np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.05 * np.square(action).sum()
        reward = reward_dist + reward_ctrl
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))
        return self._state, reward, done, {}

    def reset_task(self, goal):
        self._goal = np.array(goal, dtype=np.float32).reshape(-1)

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(0, 0.5, size=(num_tasks, 2))
        return tasks


class Navigation2DEnvV2(gym.Env):
    def __init__(self):
        super(Navigation2DEnvV2, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)
        self._goal = np.array([0, 0.5], dtype=np.float32)
        self._start = np.array([0, -0.5], dtype=np.float32)
        self._state = np.copy(self._start)
        self.oracle = False

        self._wall_width = 0.3; self._wall_height = 0.3
        self.reset_task([0.5, 0.5])

    def reset_task(self, task):
        ### task: wall center
        self._wall_center = np.array(task, dtype=np.float32).reshape(-1)
        self._wall = [self._wall_center[0]-self._wall_width, 
                self._wall_center[0]+self._wall_width, 
                self._wall_center[1]-self._wall_height,
                self._wall_center[1]+self._wall_height]

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        return tasks

    def reset_oracle(self, oracle=False):
        self.oracle = oracle
        if oracle:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(4,), dtype=np.float32)
            self._state = np.zeros(4, dtype=np.float32)
            self._state[:2] = np.copy(self._start)
            self._state[2:] = np.copy(self._wall_center)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(2,), dtype=np.float32)
            self._state = np.copy(self._start)

    def reset(self):
        if self.oracle:
            self._state = np.zeros(4, dtype=np.float32)
            self._state[:2] = np.copy(self._start)
            self._state[2:] = np.copy(self._wall_center)
        else:
            self._state = np.copy(self._start)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        #assert self.action_space.contains(action)
        temp_state = np.copy(self._state)
        temp_state[:2] = self._state[:2] + action
        pre_state = np.copy(self._state)

        ### check the three-sided wall
        navigable = self.check_wall(temp_state, pre_state)
        if navigable:
            self._state = np.copy(temp_state)

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)

        reward_ctrl = - 0.05 * np.square(action).sum()
        reward = reward_dist + reward_ctrl 
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {}

    def check_wall(self, pos, pre_pos):
        navigable = True 
        x = pos[0]; y = pos[1]
        pre_x = pre_pos[0]; pre_y = pre_pos[1]

        ### check whether the point is inside the wall
        if x>=self._wall[0] and x<=self._wall[1]:
            if y>=self._wall[2] and y<=self._wall[3]:
                navigable = False

        ### check whether the robot steps across the wall
        if x != pre_x and y != pre_y:
            slope = (y - pre_y) / (x - pre_x)
            x1 = pre_x + (self._wall[2] - pre_y) / slope
            x2 = pre_x + (self._wall[3] - pre_y) / slope
            y1 = pre_y + slope * (self._wall[0] - pre_x)
            y2 = pre_y + slope * (self._wall[1] - pre_x)
            if x1 >= self._wall[0] and x1<= self._wall[1]:
                if (x1 - x) * (x1 - pre_x) < 0:
                    navigable = False
            if x2 >= self._wall[0] and x2 <= self._wall[1]:
                if (x2 - x) * (x2 - pre_x) < 0:
                    navigable = False
            if y1 >= self._wall[2] and y1 <= self._wall[3]:
                if (y1 - y) * (y1 - pre_y) < 0:
                    navigable = False
            if y2 >= self._wall[2] and y2 <= self._wall[3]:
                if (y2 - y) * (y2 - pre_y) < 0:
                    navigable = False

        if x<=-1.0 or x>=1.0 or y<=-1.0 or y>=1.0:
            navigable = False
        
        return navigable 


class Navigation2DEnvV3(gym.Env):
    def __init__(self):
        super(Navigation2DEnvV3, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._state = np.zeros(2, dtype=np.float32)
        self.oracle = False

        self._r_small = 0.1; self._r_medium = 0.15; self._r_large = 0.2
        self.reset_task([-0.25, 0.25, 0, -0.25, 0.25, -0.25, 0.5, 0.5])

    def reset_oracle(self, oracle=False):
        self.oracle = oracle
        if oracle:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(10,), dtype=np.float32)
            self._state = np.zeros(10, dtype=np.float32)
            self._state[2:] = np.copy(self._task)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(2,), dtype=np.float32)
            self._state = np.zeros(2, dtype=np.float32)

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(-0.5, 0.5, size=(num_tasks, 8))
        return tasks

    def reset_task(self, task):
        task = np.array(task, dtype=np.float32).reshape(-1)
        assert task.shape[0] == 8
        self._task = task
        self._small = task[:2]
        self._medium = task[2:4]
        self._large = task[4:6]
        self._goal = task[-2:]

    def reset(self):
        if self.oracle:
            self._state = np.zeros(10, dtype=np.float32)
            self._state[2:] = np.copy(self._task)
        else:
            self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        #assert self.action_space.contains(action)

        temp_state = self._state[:2] + action
        navigable = self.check_puddle(temp_state)
        if navigable:
            self._state[:2] = np.copy(temp_state)

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.05 * np.square(action).sum()
        reward = reward_dist + reward_ctrl 
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        navigable = True 
        x = pos[0]; y = pos[1]
        dist_small = np.sqrt((x-self._small[0])**2 + (y-self._small[1])**2)
        if dist_small <= self._r_small:
            navigable = False 
        dist_medium = np.sqrt((x-self._medium[0])**2 + (y-self._medium[1])**2)
        if dist_medium <= self._r_medium:
            navigable = False 
        dist_large = np.sqrt((x-self._large[0])**2 + (y-self._large[1])**2)
        if dist_large <= self._r_large:
            navigable = False 
        if x<=-1.0 or x>=1.0 or y<=-1.0 or y>=1.0:
            navigable = False
        return navigable 

