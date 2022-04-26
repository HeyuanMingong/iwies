import numpy as np
import torch


def bc_func(model, env, args, device='cpu'):
    ### behavior characterization 
    xy_coordinates = np.zeros((2, args.max_steps), dtype=np.float32)
    def get_xy(state):
        if args.env in ['HopperVel-v1', 'HalfCheetahVel-v1', 'SwimmerVel-v1']:
            return np.array([env.sim.data.qpos[0], 0], dtype=np.float32)
        elif args.env in ['AntVel-v1']:
            return np.array([env.get_body_com("torso")[0], 0], dtype=np.float32)
        else:
            return state[:2]

    s = env.reset()
    reward = 0.0
    for i_step in range(args.max_steps):
        xy_coordinates[:, i_step] = get_xy(s)
        s = torch.from_numpy(s).float().unsqueeze(0).to(device)
        a = model(s).cpu().data.numpy()[0]
        s_, r, done, _ = env.step(a)
        s = s_
        reward += r
        if done:
            break

    if i_step < args.max_steps - 1:
        xy_coordinates[:, i_step+1:] = get_xy(s).reshape(-1,1).dot(
                np.ones((1, args.max_steps-1-i_step)))
    
    return reward, xy_coordinates


def bc_dist_func(bc1, bc2, metric='trajectory'):
    ### calculate the distance of two behavior charateristics, bc: (2, max_steps)
    assert bc1.shape == bc2.shape
    if metric == 'point':
        dist =  np.mean(np.sqrt(np.ones(2).dot((bc1[:, -1] - bc2[:,-1])**2)))
    else: 
        dist =  np.mean(np.sqrt(np.ones(2).dot((bc1 - bc2)**2)))
    return dist





