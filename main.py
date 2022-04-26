from __future__ import absolute_import, division, print_function

import os
import argparse
import gym
import torch
import numpy as np
import time
start_time = time.time()
import scipy.io as sio
import random

from es import build_model, ES
from models import Policy
import envs
from novelty import bc_func, bc_dist_func

parser = argparse.ArgumentParser(description='Work in process')
parser.add_argument('--env', default='Navigation2D-v1',
        help='(self-defined) gym environment')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
        help='learning rate for natural evolution strategies')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
        help='noise standard deviation')
parser.add_argument('--batch_size', type=int, default=16,
        help='batch size, must be even')
parser.add_argument('--max_steps', type=int, default=100,
        help='maximum length of an episode')
parser.add_argument('--max_epochs', type=int, default=100,
        help='maximum number of updates')
parser.add_argument('--output', type=str, default='output/navi_v1',
        help='output folder to save the experimental results')
parser.add_argument('--model_path', type=str, default='saves/navi_v1',
        help='output folder to save the pre-trained model')
parser.add_argument('--stage', type=str, default='pretrain',
        help='pretrain (original environment) or finetune (new environment)')
parser.add_argument('--CA', action='store_true',
        help='the baseline Continuous Adaptation')
parser.add_argument('--IW_IES_Qu', action='store_true',
        help='IWIES with Instance Quality')
parser.add_argument('--IW_IES_N', action='store_true', 
        help='IWIES with Instance Novelty')
parser.add_argument('--IW_IES_Mix', action='store_true',
        help='IWIES with mix weighting')
parser.add_argument('--FS', action='store_true', default=False,
        help='the baseline From Scratch')
parser.add_argument('--dist_metric', type=str, default='trajectory')
parser.add_argument('--seed', type=int, default=950418)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--num_workers', type=int, default=16)
args = parser.parse_args()
    
np.set_printoptions(precision=3)
np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

assert args.batch_size % 2 == 0
print('Make the environment: %s'%args.env)
env = gym.make(args.env).unwrapped
env.seed(args.seed)
print('State dim: %d'%env.observation_space.shape[0])
print('Action dim: %d'%env.action_space.shape[0])
print('Max action: %.2f'%env.action_space.high[0])


################################################################################
### Hyperparameters
epochs = args.max_epochs
if args.env in ['Navigation2D-v1']:
    TAU_QUA = list(np.linspace(0.5, 0, epochs))
    TAU_NOV = list(np.linspace(10.0, 0, epochs//3)) + [0] * epochs

elif args.env == 'Navigation2D-v2':
    TAU_NOV = list(np.linspace(100, 0, epochs//3)) + [0]*epochs
    TAU_QUA = list(np.linspace(0.05, 0, epochs)) + [0]*epochs

elif args.env == 'Navigation2D-v3':
    TAU_NOV = list(np.linspace(10, 0, epochs//3)) + [0] * epochs
    TAU_QUA = list(np.linspace(0.5, 0, epochs)) + [0]*epochs

elif args.env == 'HopperVel-v1':
    TAU_NOV = list(np.linspace(20, 0, epochs//4)) + [0]*epochs
    TAU_QUA = list(np.linspace(0.5, 0, epochs))

elif args.env == 'HalfCheetahVel-v1':
    TAU_NOV = list(np.linspace(3, 0, epochs//4)) + [0]*epochs
    TAU_QUA = list(np.linspace(0.1, 0, epochs))

elif args.env == 'SwimmerVel-v1':
    TAU_NOV = list(np.linspace(5, 0, epochs//5)) + [0]*epochs
    TAU_QUA = list(np.linspace(0.5, 0, epochs))

### get the new environment
new_task = np.load(os.path.join(args.model_path, 'tasks.npy'))[0]

##############################################################################
if args.stage == 'pretrain':
    print('\n---------- Pretrain in the original environment -------')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    model_pre = build_model(env, device=device)
    for param in model_pre.parameters(): param.requires_grad = False
    learner_pre = ES(model_pre, env, args, bc_func, method='baseline', device=device)

    rews = learner_pre.train_loop()
    np.save(os.path.join(args.model_path, 'rews_pre.npy'), rews)

    ### obtain the behavior characterization of previous optimal policy
    r, bc_pre = bc_func(model_pre, env, args, device=device)
    np.save(os.path.join(args.model_path, 'bc_pre.npy'), bc_pre)

    name = os.path.join(args.model_path, 'model_pre.pth')
    torch.save(model_pre.state_dict(), name)
    print('------------------------------------------------------\n')

elif args.stage == 'finetune':
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    state_dict = torch.load(os.path.join(args.model_path, 'model_pre.pth'))
    bc_pre = np.load(os.path.join(args.model_path, 'bc_pre.npy'))

    ### continuous adaptation, CA
    model_ca = build_model(env, device=device)
    for param in model_ca.parameters(): param.requires_grad = False
    learner_ca = ES(model_ca, env, args, bc_func, method='baseline', device=device)
    rews_ca = np.zeros(args.max_epochs)

    ### IW-IES-N
    model_nov = build_model(env, device=device)
    for param in model_nov.parameters(): param.requires_grad = False
    learner_nov = ES(model_nov, env, args, bc_func, method='nov', 
            bc_dist_func=bc_dist_func, bc_pre=bc_pre, device=device)
    rews_nov = np.zeros(args.max_epochs)

    ### IW-IES-Qu
    model_qua = build_model(env, device=device)
    for param in model_qua.parameters(): param.requires_grad = False
    learner_qua = ES(model_qua, env, args, bc_func, method='qua', device=device)
    rews_qua = np.zeros(args.max_epochs)

    ### IW-IES-Mix
    model_mix = build_model(env, device=device)
    for param in model_mix.parameters(): param.requires_grad = False
    learner_mix = ES(model_mix, env, args, bc_func, method='mix', 
            bc_dist_func=bc_dist_func, bc_pre=bc_pre, device=device)
    rews_mix = np.zeros(args.max_epochs)

    ### From Scratch, FS
    rews_fs = np.zeros(args.max_epochs)

    
    print('Reset task:', new_task)
    env.reset_task(new_task)

    if args.CA:
        print('\n--- the Continuous Adaptation baseline ---')
        model_ca.load_state_dict(state_dict)
        rews = learner_ca.train_loop()
        rews_ca = rews[0]
        np.save(os.path.join(args.output, 'rews_ca.npy'), rews_ca)

    if args.IW_IES_Qu:
        print('\n--- IWIES with Instance Quality ---')
        model_qua.load_state_dict(state_dict)
        rews = learner_qua.train_loop(tau_qua=TAU_QUA)
        rews_qua = rews[0]
        np.save(os.path.join(args.output, 'rews_qua.npy'), rews_qua)

    if args.IW_IES_N:
        print('\n--- IWIES with Instance Novelty ---')
        model_nov.load_state_dict(state_dict)
        rews = learner_nov.train_loop(tau_nov=TAU_NOV)
        rews_nov = rews[0]
        np.save(os.path.join(args.output, 'rews_nov.npy'), rews_nov)

    if args.IW_IES_Mix:
        print('\n--- IWIES with Mix Weighting ---')
        model_mix.load_state_dict(state_dict)
        rews = learner_mix.train_loop(tau_nov=TAU_NOV, tau_qua=TAU_QUA)
        rews_mix = rews[0]
        np.save(os.path.join(args.output, 'rews_mix.npy'), rews_mix)

    if args.FS:
        print('\n--- the From Scratch method ---')
        model_fs = build_model(env, device=device)
        for param in model_fs.parameters(): param.requires_grad = False
        learner_fs = ES(model_fs, env, args, bc_func, method='baseline', device=device)
        rews = learner_fs.train_loop()
        rews_fs = rews[0]
        np.save(os.path.join(args.output, 'rews_fs.npy'), rews_fs)









print('Running time: %.2f seconds'%((time.time() - start_time)))
