import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import SwimmerEnv

class SwimmerVelEnv(SwimmerEnv):
    def __init__(self):
        self._goal_vel = 0.2
        self.oracle = False
        super(SwimmerVelEnv, self).__init__()

    def reset_oracle(self, oracle=False):
        self.oracle = oracle
        if oracle:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(9,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(8,), dtype=np.float32)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        forward_vel = (xposafter - xposbefore) / self.dt
        reward_fwd = 1.0 - 2 * abs(forward_vel - self._goal_vel)

        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl

        ob = self._get_obs()
        if self.oracle:
            next_state = np.zeros(9, dtype=np.float32)
            next_state[:-1] = np.copy(ob)
            next_state[-1] = self._goal_vel
        else:
            next_state = np.copy(ob)
    
        return next_state, reward, False, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([
            qpos.flat[2:], qvel.flat]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, 
                size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, 
                size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_task(self, task):
        self._goal_vel = task

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(0, 0.5, size=(num_tasks,))
        return tasks

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.oracle:
            state = np.zeros(9, dtype=np.float32)
            state[:-1] = np.copy(ob)
            state[-1] = self._goal_vel
        else:
            state = np.copy(ob)
        return state
