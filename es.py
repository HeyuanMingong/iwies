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

from models import Policy


def softmax_normalize(arr, tau=1.0):
    arr = np.array(arr).reshape(-1)
    arr -= arr.mean()
    arr = np.exp(tau * arr) / np.sum(np.exp(tau * arr)) * arr.shape[0]
    return arr

def normalize(arr, eps=1e-10, scale=3.0):
    ### normalize a numpy array
    arr = np.array(arr).reshape(-1)
    arr_mean = arr.mean(); arr_var = arr.var()
    arr = (arr - arr_mean) / (np.sqrt(arr_var) + eps)
    arr = np.clip(arr, -scale, scale)
    arr -= arr.min(); arr /= (arr.mean() + eps)
    return arr

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
        math.log(sorted_returns_backwards.index(r) + 1, 2)) for r in returns])

    for r in returns:
        num = max(0, math.log(lamb/2 + 1, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2))
        shaped_returns.append(num/denom + 1/lamb)
    return np.array(shaped_returns).reshape(-1)

def build_model(env, device='cpu'):
    ### build the policy network, for discrete or continuous control problems
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])
    model = Policy(state_dim, action_dim, max_action, device=device)
    return model


#############################################################################
class ES(object):
    def __init__(self, model, env, args, bc_func, bc_dist_func=None,
            method='baseline', bc_pre=None, device='cpu'):
        self.model = model
        print("Num params in network %d" % self.model.count_parameters())
        self.env = env
        self.args = args
        self.device = device

        """
        'baseline': default method
        'nov': IWIES-N
        'qua': IWIES-Qu
        'mix': IWIES-Mix        
        """
        self.method = method
        
        self.bc_func = bc_func
        self.bc_dist_func = bc_dist_func
        self.bc_pre = bc_pre

    def do_rollouts(self, models, random_seeds, return_queue, are_negative):
        """
        For each model, do a rollout. Supports multiple models per thread but
        don't do it -- it's inefficient (it's mostly a relic of when I would run
        both a perturbation and its antithesis on the same thread).
        """
        all_returns = []; all_dists = []
        for model in models:
            this_model_return, this_model_bc = self.bc_func(model, self.env, self.args)

            ### for methods with instance novelty, use a distance function
            if self.method in ['nov', 'mix']:
                this_model_dist = self.bc_dist_func(self.bc_pre, this_model_bc,
                        metric=self.args.dist_metric)
            else:
                this_model_dist = None

            all_returns.append(this_model_return)
            all_dists.append(this_model_dist)
        return_queue.put((random_seeds, are_negative, all_returns, all_dists))


    def perturb_model(self, random_seed):
        """
        Modifies the given model with a pertubation of its parameters,
        as well as the negative perturbation, and returns both perturbed models.
        """
        new_model = build_model(self.env, device=self.device)
        anti_model = build_model(self.env, device=self.device)
        for param in new_model.parameters(): param.requires_grad = False
        for param in anti_model.parameters(): param.requires_grad = False

        new_model.load_state_dict(self.model.state_dict())
        anti_model.load_state_dict(self.model.state_dict())
        np.random.seed(random_seed)
        for (k, v), (anti_k, anti_v) in zip(new_model.es_params(), anti_model.es_params()):
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(self.args.sigma*eps).float().to(self.device)
            anti_v += torch.from_numpy(self.args.sigma * -eps).float().to(self.device)
        return [new_model, anti_model]

    def generate_seeds_and_models(self):
        ### Returns a seed and 2 perturbed models
        np.random.seed()
        random_seed = np.random.randint(2**30)
        two_models = self.perturb_model(random_seed)
        return random_seed, two_models

    def gradient_update(self, returns, random_seeds, neg_list, dists=None, tau_qua=1.0, tau_nov=1.0):
        batch_size = len(returns)
        assert batch_size == self.args.batch_size
        assert len(random_seeds) == batch_size
        shaped_returns = fitness_shaping(returns)
        
        """
        For each model, generate the same random numbers as we did
        before, and update parameters. We apply weight decay once.
        """
        if self.method == 'baseline':
            objective = shaped_returns
        elif self.method == 'nov':
            weights_n = softmax_normalize(dists, tau=tau_nov)
            #print(weights_n.min(), weights_n.max())
            objective = shaped_returns * weights_n
        elif self.method == 'qua':
            ### importance weighting using the original returns 
            weights_qu = softmax_normalize(returns, tau=tau_qua)
            #print(weights_qu.min(), weights_qu.max())
            objective = shaped_returns * weights_qu
        elif self.method == 'mix':
            weights_qu = softmax_normalize(returns, tau=tau_qua)
            weights_n = softmax_normalize(dists, tau=tau_nov)
            #print(weights_n.min(), weights_n.max())
            weights = weights_qu * weights_n
            weights = weights / np.sum(weights) * weights.shape[0]
            #weights = softmax_normalize(weights_qu*weights_n, tau=tau_mix)
            #print('weights mix...'); print(weights.min(), weights.max())
            objective = shaped_returns * weights

        for i in range(self.args.batch_size):
            np.random.seed(random_seeds[i])
            multiplier = -1 if neg_list[i] else 1
            reward = objective[i]
            for k, v in self.model.es_params():
                eps = np.random.normal(0, 1, v.size())
                grad = self.args.lr / (self.args.batch_size * 
                    self.args.sigma) * (reward * multiplier * eps)
                v += torch.from_numpy(grad).float().to(self.device)

    def train_loop(self, tau_qua=[1]*10000, tau_nov=[1]*10000):
        epochs = self.args.max_epochs
        rews = np.zeros((4, epochs), dtype=np.float32)

        for i_step in tqdm(range(epochs)):
            processes = []
            return_queue = mp.Queue()
            all_seeds, all_models = [], []
            # Generate a perturbation and its antithesis
            for j in range(int(self.args.batch_size/2)):
                random_seed, two_models = self.generate_seeds_and_models()
                # Add twice because we get two models with the same seed
                all_seeds.append(random_seed)
                all_seeds.append(random_seed)
                all_models += two_models
            assert len(all_seeds) == len(all_models)
            # Keep track of which perturbations were positive and negative
            # Start with negative true because pop() makes us go backwards
            is_negative = True
            # Add all peturbed models to the queue
            raw_results = []
            all_models.insert(0, self.model)
            all_seeds.insert(0, 'dummy_seed')
            while all_models:
                processes = []
                for i_core in range(self.args.num_workers):
                    if all_models:
                        perturbed_model = all_models.pop()
                        seed = all_seeds.pop()
                        p = mp.Process(target=self.do_rollouts, args=([perturbed_model], 
                            [seed], return_queue, [is_negative]))
                        p.start()
                        processes.append(p)
                        is_negative = not is_negative

                for p in processes:
                    p.join()

                results = [return_queue.get() for p in processes]
                raw_results += results

            assert len(all_seeds) == 0


            seeds, neg_list, results, dists = [flatten(raw_results, index) 
                    for index in [0, 1, 2, 3]]

            rews[1, i_step] = max(results)
            rews[2, i_step] = min(results)
            rews[3, i_step] = np.mean(np.array(results))

            # Separate the unperturbed results from the perturbed results
            _ = unperturbed_index = seeds.index('dummy_seed')
            seeds.pop(unperturbed_index)
            unperturbed_results = results.pop(unperturbed_index)
            _ = neg_list.pop(unperturbed_index)
            dists.pop(unperturbed_index)

            rews[0, i_step] = unperturbed_results

            self.gradient_update(results, seeds, neg_list, dists=dists, 
                    tau_qua=tau_qua[i_step], tau_nov=tau_nov[i_step])
        return rews






