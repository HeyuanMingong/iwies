import numpy as np
from gym.envs.mujoco import HalfCheetahEnv
from gym import utils, spaces


class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self):
        self._goal_vel = 0.5
        self.oracle = False
        super(HalfCheetahVelEnv, self).__init__()

    def reset_oracle(self, oracle=False):
        self.oracle = oracle
        if oracle:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(21,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                    shape=(20,), dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = - 1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = - 0.01 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward + ctrl_cost
        done = False

        if self.oracle:
            next_state = np.zeros(21, dtype=np.float32)
            next_state[:-1] = np.copy(observation)
            next_state[-1] = self._goal_vel
        else:
            next_state = np.copy(observation)
        return next_state, reward, done, {}

    def sample_task(self, num_tasks=1):
        ### for Robust, MAML
        #tasks = np.random.uniform(0.3, 0.7, size=(num_tasks,))

        ### for Adaptive
        tasks = np.random.uniform(0.45, 0.55, size=(num_tasks,))

        ### for SO-CMA
        #tasks = np.random.uniform(0.5, 2.0, size=(num_tasks,))
        return tasks

    def reset_task(self, task):
        self._goal_vel = task

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
                low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.oracle:
            state = np.zeros(21, dtype=np.float32)
            state[:-1] = np.copy(ob)
            state[-1] = self._goal_vel
        else:
            state = np.copy(ob)
        return state
