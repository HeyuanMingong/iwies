from __future__ import absolute_import, division, print_function

import os
import math
import numpy as np
from tqdm import tqdm
from gym.spaces import Discrete, Box
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
import gym
from models import Policy, LSTMPolicy


def flatten(raw_results, index):
    notflat_results = [result[index] for result in raw_results]
    return [item for sublist in notflat_results for item in sublist]

def fitness_shaping(returns):
    """
    A rank transformation on the rewards, which reduces the chances
    of falling into local optima early in training.
    """
    sorted_returns_backwards = sorted(returns)[::-1]
    lamb = len(returns)
    shaped_returns = []
    denom = sum([max(0, math.log(lamb/2 + 1, 2) -
                     math.log(sorted_returns_backwards.index(r) + 1, 2))
                 for r in returns])
    for r in returns:
        num = max(0, math.log(lamb/2 + 1, 2) -
                  math.log(sorted_returns_backwards.index(r) + 1, 2))
        shaped_returns.append(num/denom + 1/lamb)
    return np.array(shaped_returns).reshape(-1)

def build_model(env, recurrent=False, device='cpu'):
    ### build the policy network, for discrete or continuous control problems
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])
    if recurrent:
        model = LSTMPolicy(state_dim, action_dim, max_action, device=device)
    else:
        model = Policy(state_dim, action_dim, max_action, device=device)
    return model


#############################################################################
class ES(object):
    def __init__(self, model, env, args, device='cpu', recurrent=False, oracle=False):
        self.model = model
        print("Num params in network %d" % self.model.count_parameters())
        self.env = env
        self.args = args
        self.recurrent = recurrent
        self.oracle = oracle
        self.device = device

    def do_rollouts(self, models,envs,random_seeds,return_queue,are_negative):
        all_returns = []
        for env, model in zip(envs, models):
            this_model_return = 0.0
            s = env.reset()
            s_hist = [s.reshape(1,-1)]
            for step in range(self.args.max_steps):
                if self.recurrent:
                    s_seq = np.concatenate(s_hist, axis=0)[-self.args.seq_len:]
                    s_seq = torch.from_numpy(s_seq).float().unsqueeze(1).to(
                            self.device)
                    a = model(s_seq).cpu().data.numpy()[0]
                else:
                    s = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                    a = model(s).cpu().data.numpy()[0]

                s_next, r, done, _ = env.step(a)
                s = s_next
                s_hist.append(s.reshape(1, -1))

                this_model_return += r
                if done:
                    break
            all_returns.append(this_model_return)
        return_queue.put((random_seeds, are_negative, all_returns))

    def perturb_model(self, model, random_seed):
        """
        Modifies the given model with a pertubation of its parameters,
        as well as the negative perturbation, and returns both perturbed models.
        """
        new_model = build_model(self.env, recurrent=self.recurrent, device=self.device)
        anti_model = build_model(self.env, recurrent=self.recurrent, device=self.device)
        for param in new_model.parameters(): param.requires_grad = False
        for param in anti_model.parameters(): param.requires_grad = False

        new_model.load_state_dict(model.state_dict())
        anti_model.load_state_dict(model.state_dict())
        np.random.seed(random_seed)
        for (k, v), (anti_k, anti_v) in zip(new_model.es_params(), anti_model.es_params()):
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(self.args.sigma*eps).float().to(self.device)
            anti_v += torch.from_numpy(self.args.sigma*-eps).float().to(self.device)
        return [new_model, anti_model]

    def generate_seeds_and_models(self, model):
        """
        Returns a seed and 2 perturbed models
        """
        np.random.seed()
        random_seed = np.random.randint(2**30)
        two_models = self.perturb_model(model, random_seed)
        return random_seed, two_models

    def gradient_update(self, returns, random_seeds, neg_list):
        batch_size = len(returns)
        assert batch_size == self.args.batch_size
        assert len(random_seeds) == batch_size
        shaped_returns = fitness_shaping(returns)
        
        for i in range(self.args.batch_size):
            np.random.seed(random_seeds[i])
            multiplier = -1 if neg_list[i] else 1
            reward = shaped_returns[i]
            for k, v in self.model.es_params():
                eps = np.random.normal(0, 1, v.size())
                grad = self.args.lr / (self.args.batch_size * 
                        self.args.sigma) * (reward * multiplier * eps)
                v += torch.from_numpy(grad).float().to(self.device)

    def inner_gradient(self, model, env):
        processes = []
        return_queue = mp.Queue()
        all_seeds, all_models = [], []
        for j in range(int(self.args.batch_size/2)):
            random_seed, two_models = self.generate_seeds_and_models(model)
            all_seeds.append(random_seed); all_seeds.append(random_seed)
            all_models += two_models
        assert len(all_models) == len(all_seeds)

        is_negative = True
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            p = mp.Process(target=self.do_rollouts, args=([perturbed_model], 
                [env], [seed], return_queue, [is_negative]))
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(all_seeds) == 0

        for p in processes: p.join()

        raw_results = [return_queue.get() for p in processes]
        seeds, neg_list, results = [flatten(raw_results, index)
                for index in [0, 1, 2]]

        assert len(results) == self.args.batch_size

        shaped_returns = fitness_shaping(results)
        for i in range(self.args.batch_size):
            np.random.seed(seeds[i])
            multiplier = -1 if neg_list[i] else 1
            reward = shaped_returns[i]
            for k, v in model.es_params():
                eps = np.random.normal(0, 1, v.size())
                grad =  self.args.lr / (self.args.batch_size * 
                    self.args.sigma) * (reward * multiplier * eps)
                v += torch.from_numpy(grad).float().to(self.device)
        return model

    def train_loop(self, robust=False, maml=False):
        epochs = self.args.max_epochs
        rews = np.zeros((4, epochs), dtype=np.float32)
        for i_step in tqdm(range(epochs)):
            processes = []
            return_queue = mp.Queue()
            all_seeds, all_models, all_envs = [], [], []
            # Generate a perturbation and its antithesis
            for j in range(int(self.args.batch_size/2)):
                random_seed,two_models=self.generate_seeds_and_models(self.model)
                # Add twice because we get two models with the same seed
                all_seeds.append(random_seed); all_seeds.append(random_seed)
                all_models += two_models

                ### train the model with a distribution of tasks
                if robust or maml:
                    env = gym.make(self.args.env).unwrapped
                    env.reset_oracle(oracle=self.oracle)
                    task = env.sample_task()
                    env.reset_task(task)
                    all_envs.append(env); all_envs.append(env)
                else:
                    all_envs.append(self.env); all_envs.append(self.env)

            is_negative = True
            while all_models:
                model = all_models.pop()
                env = all_envs.pop()
                seed = all_seeds.pop()
                if maml: model = self.inner_gradient(model, env)

                p = mp.Process(target=self.do_rollouts, args=([model], [env],
                    [seed], return_queue, [is_negative]))
                p.start()
                processes.append(p)
                is_negative = not is_negative
            assert len(all_seeds) == 0

            p = mp.Process(target=self.do_rollouts, args=([self.model], 
                [self.env], ['dummy_seed'], return_queue, ['dummy_neg']))
            p.start()
            processes.append(p)
            for p in processes: p.join()
            
            raw_results = [return_queue.get() for p in processes]
            seeds, neg_list, results = [flatten(raw_results, index) for index in [0, 1, 2]]
            
            rews[1, i_step] = max(results)
            rews[2, i_step] = min(results)
            rews[3, i_step] = np.mean(np.array(results))

            unperturbed_index = seeds.index('dummy_seed')
            seeds.pop(unperturbed_index)
            unperturbed_result = results.pop(unperturbed_index)
            _ = neg_list.pop(unperturbed_index)

            rews[0, i_step] = unperturbed_result

            self.gradient_update(results, seeds, neg_list)
        return rews





