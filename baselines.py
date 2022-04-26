from __future__ import absolute_import, division, print_function

import os
import argparse
import gym
import torch
import numpy as np
import time
start_time = time.time()
import scipy.io as sio
import cma
from tqdm import tqdm

#from es2 import build_model, ES
from es_baselines import build_model, ES
from models import Policy, LSTMPolicy
import envs

parser = argparse.ArgumentParser(description='Work in process')
parser.add_argument('--env', default='Navigation2D-v1',
        help='(self-defined) gym environment')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
        help='learning rate for natural evolution strategies')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
        help='noise standard deviation')
parser.add_argument('--batch_size', type=int, default=16,
        help='batch size, must be even')
parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--output', type=str, default='output/navi_v1',
        help='output folder to save the experimental results')
parser.add_argument('--model_path', type=str, default='saves/navi_v1',
        help='output folder to save the pre-trained model')
parser.add_argument('--seq_len', type=int, default=10, 
        help='for the baseline Hist using an LSTM encoder')
parser.add_argument('--seed', type=int, default=950418)
parser.add_argument('--so', action='store_true', default=False,
        help='the baseline SO-CMA')
parser.add_argument('--hist', action='store_true', default=False,
        help='the baseline Hist')
parser.add_argument('--robust', action='store_true', default=False,
        help='the baseline Robust')
parser.add_argument('--maml', action='store_true', default=False,
        help='the baseline ES-MAML')
parser.add_argument('--stage', type=str, default='pretrain')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

np.random.seed(args.seed); torch.manual_seed(args.seed)
np.set_printoptions(precision=3)
device = torch.device(args.device if torch.cuda.is_available else 'cpu')
    
assert args.batch_size % 2 == 0
if not os.path.exists(args.output):
    os.makedirs(args.output)
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

### boundness for SO-CMA
if args.env == 'Navigation2D-v1':
    BOUND = [-0.5, 0.5]
elif args.env == 'Navigation2D-v2':
    BOUND = [-0.1, 0.1]
elif args.env == 'Navigation2D-v3':
    BOUND = [-0.5, 0.5]
elif args.env == 'HopperVel-v1':
    BOUND = [0.0, 1.0]
elif args.env == 'HalfCheetahVel-v1':
    BOUND = [0.0, 2.0]
elif args.env == 'SwimmerVel-v1':
    BOUND = [0.0, 0.5]

### get the new environment, make it the same with that of iwies
new_task = np.load(os.path.join(args.model_path, 'tasks.npy'))[0]

##############################################################################
if args.robust:
    print('\n---------- the baseline Robust ---------------------------')
    print('Make the environment: %s'%args.env)
    env = gym.make(args.env).unwrapped

    ### pretrain with a distribution of tasks
    model_robust = build_model(env, device=device)
    for param in model_robust.parameters(): param.requires_grad = False
    learner_robust = ES(model_robust, env, args, device=device)
    print('Training the robust policy...')
    rews = learner_robust.train_loop(robust=True)


    ### test in a new environment
    print('Test in a new task...')
    model = build_model(env, device=device)
    for param in model.parameters(): param.requires_grad = False
    learner = ES(model, env, args, device=device)

    print('New task: ', new_task)
    env.reset_task(new_task)

    model.load_state_dict(model_robust.state_dict())
    rews = learner.train_loop()
    np.save(os.path.join(args.output, 'rews_robust.npy'), rews[0])


##############################################################################
if args.hist:
    print('\n---------- the baseline Hist ----------------------')
    print('Make the environment: %s' % args.env)
    env = gym.make(args.env).unwrapped

    ### pretrain with a distribution of tasks
    model_hist = build_model(env, recurrent=True, device=device)
    for param in model_hist.parameters(): param.requires_grad = False
    learner_hist = ES(model_hist, env, args, device=device, recurrent=True)
    print('Training the adaptive model with LSTM...')
    rews = learner_hist.train_loop(robust=True)

    ### test in the new environment
    print('Test in downstream tasks...')
    model = build_model(env, recurrent=True, device=device)
    for param in model.parameters(): param.requires_grad = False
    learner = ES(model, env, args, recurrent=True, device=device)

    print('New task: ', new_task)
    env.reset_task(new_task)

    model.load_state_dict(model_hist.state_dict())
    rews = learner.train_loop()
    np.save(os.path.join(args.output, 'rews_hist.npy'), rews[0])


##############################################################################
if args.maml:
    print('\n---------- the baseline ES-MAML ----------------------')
    print('Make the environment: %s' % args.env)
    env = gym.make(args.env).unwrapped

    ### pretrain with a distribution of tasks
    model_maml = build_model(env, device=device)
    for param in model_maml.parameters(): param.requires_grad = False
    learner_maml = ES(model_maml, env, args, device=device)

    print('Training the es maml model')
    rews = learner_maml.train_loop(maml=True)

    ### test in a new environment
    print('Test in downstream tasks...')
    model = build_model(env, device=device)
    for param in model.parameters(): param.requires_grad = False
    learner = ES(model, env, args, device=device)

    print('New task: ', new_task)
    env.reset_task(new_task)

    model.load_state_dict(model_maml.state_dict())
    rews = learner.train_loop()
    np.save(os.path.join(args.output, 'rews_maml.npy'), rews[0])


##############################################################################
if args.so:
    print('\n---------- the baseline SO-CMA -------------------')
    def fitness(policy, env, x):
        r_cum = 0.0
        s = env.reset()
        for step in range(args.max_steps):
            s[-len(x):] = x
            s_tensor = torch.from_numpy(s)
            a = policy(s_tensor).cpu().numpy()
            s_next, r, done, _ = env.step(a)
            r_cum += r
            s = s_next
            if done:
                break
        return -r_cum

    print('Make the environment: %s' % args.env)
    env = gym.make(args.env).unwrapped
    env.reset_oracle(oracle=True)

    ### pretrain with a distribution of tasks
    model_so = build_model(env)
    for param in model_so.parameters(): param.requires_grad = False

    learner_so = ES(model_so, env, args, oracle=True)
    print('Training the strategy model with task parameters...')
    rews = learner_so.train_loop(robust=True)

    ### test in a new environment
    model = build_model(env)
    for param in model_so.parameters(): param.requires_grad = False
    model.load_state_dict(model_so.state_dict())

    print('Test in the downstream tasks...')
    print('New task: ', new_task)
    env.reset_task(new_task)

    if args.env in ['HopperVel-v1', 'HalfCheetahVel-v1', 'SwimmerVel-v1']:
        init_guess = np.zeros(2)
    else:
        init_guess = np.zeros(len(new_task))

    init_std = args.sigma
    options = {'bounds':BOUND, 'maxiter':args.max_epochs, 'popsize':args.batch_size}
    es = cma.CMAEvolutionStrategy(init_guess, init_std, options)
    
    rews_so = np.zeros(args.max_epochs)
    for step in tqdm(range(args.max_epochs)):
        X = es.ask()
        if args.env in ['HopperVel-v1', 'HalfCheetahVel-v1', 'SwimmerVel-v1']:
            fits = [fitness(model_so, env, [x[0]]) for x in X]
        else:
            fits = [fitness(model_so, env, x) for x in X]
        rews_so[step] = -np.array(fits).mean()
        es.tell(X, fits)
    np.save(os.path.join(args.output, 'rews_so.npy'), rews_so)
















print('Running time: %.2f min'%((time.time() - start_time)/60.0))
