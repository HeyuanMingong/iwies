import numpy as np
from gym import utils, spaces
#from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import HopperEnv 

class HopperVelEnv(HopperEnv):
    def __init__(self):
        self._goal_vel = 0.5
        self.oracle = False
        super(HopperVelEnv, self).__init__()

    def reset_oracle(self, oracle=False):
        self.oracle = oracle
        if oracle:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(12,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(11,), dtype=np.float32)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        #alive_bonus = 1.0
        #reward = (posafter - posbefore) / self.dt
        #reward += alive_bonus
        #reward -= 1e-3 * np.square(a).sum()
        
        alive_bonus = 1.0
        forward_vel = (posafter - posbefore) / self.dt 
        forward_reward = - 2.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = - 0.01 * np.square(a).sum()
        reward = forward_reward + ctrl_cost + alive_bonus
        #reward_norm = np.clip(reward, 0, np.inf)

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        if self.oracle: 
            next_state = np.zeros(12, dtype=np.float32)
            next_state[:-1] = np.copy(ob)
            next_state[-1] = self._goal_vel
        else:
            next_state = np.copy(ob)
        
        return next_state, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ]).astype(np.float32).flatten()

    def reset_task(self, task):
        self._goal_vel = task

    def sample_task(self, num_tasks=1):
        ### SO-CMA
        #tasks = np.random.uniform(0.5, 1.0, size=(num_tasks,))

        ### MAML, Adaptive, Robust
        tasks = np.random.uniform(0.3, 0.7, size=(num_tasks,))
        return tasks

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.oracle:
            state = np.zeros(12, dtype=np.float32)
            state[:-1] = np.copy(ob)
            state[-1] = self._goal_vel
        else:
            state = np.copy(ob)
        return state

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
                low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
                low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()



