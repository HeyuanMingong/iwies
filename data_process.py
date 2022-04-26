#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:27:58 2018

@author: qiutian
"""

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as st

def arr_ave(arr, bs=1, dim=None):
    if dim is None:
        arr = arr.squeeze()
        nl = arr.shape[0]//bs
        arr_n = np.zeros(nl)
        for i in range(nl):
            arr_n[i] = np.mean(arr[bs*i:bs*(i+1)])
        return arr_n
    elif dim:
        arr_n = []
        for row in arr:
            row_n = arr_ave(row, bs=bs, dim=None)
            arr_n.append(row_n)
        return np.array(arr_n)
    

def bottom_scale(arr, scale=[-160, -100, -80], dim=None):
    if dim is None:
        arr = np.array(arr).reshape(-1); arr_n = np.zeros(arr.shape)
        ratio = (scale[2] - scale[1])/(scale[2] - scale[0])
        def shrink(num):
            if num < scale[2]:
                return scale[2] - ratio * (scale[2] - num)
            else:
                return num
        for idx, ii in enumerate(arr):
             arr_n[idx] = shrink(ii)
        return arr_n
    else:
        arr_n = []
        for row in arr:
            arr_n.append(bottom_scale(row, scale=scale, dim=None))
        return np.array(arr_n)


def cutoff_complement(arr, cutoff):
    arr = arr.reshape(-1)
    if len(arr) >= cutoff:
        return arr[:cutoff]
    else:
        arr_new = np.zeros(cutoff)
        arr_new[:len(arr)] = arr 
        arr_new[len(arr):] = arr[-1]
        return arr_new



###############################################################################
DOMAIN = 'navi_v3'
r_path = 'output/%s'%DOMAIN
s_path = 'saves/%s'%DOMAIN

#tasks = np.random.uniform(-0.1, 0.1, size=(1000, 2))
#np.save(os.path.join(s_path, 'tasks.npy'), tasks)

#selected = [7,22,25,42,45]
#np.save(os.path.join(s_path, 'task_selected.npy'), np.array(selected).reshape(-1))


###############################################################################
def rews_pretrain():
    cutoff = 200; num = 100; bs = cutoff // num
    f = np.load(os.path.join(s_path, 'rews_pre.npy'))
    rews = arr_ave(cutoff_complement(f[0],cutoff=cutoff), bs=bs)
    rews_max = arr_ave(cutoff_complement(f[1],cutoff=cutoff), bs=bs)
    #rews_min = arr_ave(f['rews_min'].squeeze()[:cutoff], bs=bs)
    #rews_mean = arr_ave(f['rews_mean'].squeeze()[:cutoff], bs=bs)
    
    xx = np.arange(0, num, 1)
    plt.figure(figsize=(6,4))
    plt.plot(xx, rews, xx, rews_max)
    plt.legend(['rews', 'max'])
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5), fontsize=10)
        
    return f
#  data = rews_pretrain()
    

###############################################################################
def ablation_study():
    rews_fs = np.load(os.path.join(r_path, 'rews_fs.npy'))
    rews_ca = np.load(os.path.join(r_path, 'rews_ca.npy'))
    rews_nov = np.load(os.path.join(r_path, 'rews_nov.npy'))
    rews_qua = np.load(os.path.join(r_path, 'rews_qua.npy'))
    rews_mix = np.load(os.path.join(r_path, 'rews_mix.npy'))
    #rews_mix = rews_fs = rews_ca
    
    print('Average return of FS: %.2f' % rews_fs.mean())
    print('Average return of CA: %.2f' % rews_ca.mean())
    print('Average return of IW-IES-N: %.2f' % rews_nov.mean())
    print('Average return of IW-IES-Qu: %.2f' % rews_qua.mean())
    print('Average return of IW-IES-Mix: %.2f' % rews_mix.mean())

    x = range(len(rews_fs))
    plt.figure()
    plt.plot(x, rews_fs, x, rews_ca, x, rews_nov, x, rews_qua, x, rews_mix)
    plt.legend(['FS', 'CA', 'IW-IES-N', 'IW-IES-Qu', 'IW-IES-Mix'])    
    plt.xlabel('Generations', fontsize=15)
    plt.ylabel('Return', fontsize=15)
    # plt.title('Navigation Case Complex')

# ablation_study()


###############################################################################
def comparison_to_baselines():
    rews_robust = np.load(os.path.join(r_path, 'rews_robust.npy'))
    rews_so = np.load(os.path.join(r_path, 'rews_so.npy'))
    rews_hist = np.load(os.path.join(r_path, 'rews_hist.npy'))
    rews_maml = np.load(os.path.join(r_path, 'rews_maml.npy'))
    rews_mix = np.load(os.path.join(r_path, 'rews_mix.npy'))
    #rews_so = rews_mix

    x = range(len(rews_robust))
    plt.figure()
    plt.plot(x, rews_robust, x, rews_so, x, rews_hist, x, rews_maml, x, rews_mix)
    plt.legend(['Robust', 'SO-CMA', 'Hist', 'ES-MAML', 'IW-IES'])  
    plt.xlabel('Generations', fontsize=15)
    plt.ylabel('Return', fontsize=15)
    plt.title('Navigation Case Complex')

comparison_to_baselines()


















###############################################################################
def data_stats():
    names = ['fs', 'da', 'nov', 'qua', 'mix','robust', 'so', 'adaptive', 'maml']
    
    if DOMAIN == 'navi_v1':
        selected = [3,4,5,6,8,21,26,31,35,48]
        cutoff = 200
    elif DOMAIN == 'navi_v2':
        selected = [7,22,25,42,45]
        cutoff = 1000
    elif DOMAIN == 'navi_v3':
        selected = [5,10,13,14,20,24,30,39,40,46]
        cutoff = 500
    elif DOMAIN == 'hopper':
        selected = [20,25,26,27,29,30,31,32,33,33]
        cutoff = 200
    elif DOMAIN == 'cheetah':
        selected = [6,7,8,9,16,17,19,26,27,29]
        cutoff = 200
    elif DOMAIN == 'swimmer':
        selected = [0,1,3,4,5,9,11,13,14,17]
        cutoff = 100
        
    '''
    names = ['robust_quadrant3', 'so_quadrant3', 'adaptive_quadrant3', 
             'maml_quadrant3', 'qua_quadrant3']
    selected = range(10); cutoff = 200
    '''
    
    rews_ave = []
    for name in names:
        rews = np.load(os.path.join(r_path, 'rews_%s.npy'%name))[selected, :cutoff]-50
        
        print('%s: %.2f'%(name, rews.mean(axis=0).mean()))
        rews_ave.append(rews.mean(axis=1))
        
        rews_up, rews_down = np.zeros(rews.shape[1]), np.zeros(rews.shape[1])
        
        for i_col in range(rews.shape[1]):
            i_rews = rews[:, i_col]
            i_rews_down, i_rews_up = st.t.interval(0.95, len(i_rews)-1, 
                loc=np.mean(i_rews), scale=st.sem(i_rews))
            rews_up[i_col] = i_rews_up; rews_down[i_col] = i_rews_down
            
        root_data = np.zeros((3, rews.shape[1]))
        root_data[0] = np.mean(rews, axis=0)
        root_data[1] = rews_down; root_data[2] = rews_up
        #np.save(os.path.join(r_path, 'rews_%s_stats.npy'%name), root_data)
        
    rews_ave = np.array(rews_ave).T
    return rews_ave
#data = data_stats()



###############################################################################
def rews_stats_finetune():
    if DOMAIN in ['navi_v1', 'hopper']:
        cutoff = 200; num = 100
    elif DOMAIN in ['navi_v2']:
        cutoff = 1000; num = 100
    elif DOMAIN in ['navi_v3']:
        cutoff = 500; num = 100
    elif DOMAIN in ['cheetah']:
        cutoff = 200; num = 50
    elif DOMAIN in ['swimmer']:
        cutoff = 100; num = 100
        
        
    bs = cutoff // num; mark = num // 10
    rews_fs = np.load(os.path.join(r_path, 'rews_fs_stats.npy'))[:, :cutoff]
    rews_da = np.load(os.path.join(r_path, 'rews_da_stats.npy'))[:, :cutoff]
    rews_nov = np.load(os.path.join(r_path, 'rews_nov_stats.npy'))[:, :cutoff]
    rews_qua = np.load(os.path.join(r_path, 'rews_qua_stats.npy'))[:, :cutoff]
    rews_mix = np.load(os.path.join(r_path, 'rews_mix_stats.npy'))[:, :cutoff]
    
    
    rews_fs = arr_ave(rews_fs, bs=bs, dim=1)
    rews_da = arr_ave(rews_da, bs=bs, dim=1)
    rews_nov = arr_ave(rews_nov, bs=bs, dim=1)
    rews_qua = arr_ave(rews_qua, bs=bs, dim=1)
    rews_mix = arr_ave(rews_mix, bs=bs, dim=1)
    
    if DOMAIN == 'hopper': 
        scale = [-75, 0, 20]
    elif DOMAIN == 'navi_v1':
        scale = [-120, -80, -60]
    elif DOMAIN == 'cheetah':
        scale = [-100, -50, -25]
    elif DOMAIN == 'swimmer':
        scale = [-60, 0, 5]
    
    if DOMAIN in ['hopper', 'navi_v1', 'cheetah', 'swimmer']:
        rews_fs = bottom_scale(rews_fs, scale=scale, dim=1)
        rews_da = bottom_scale(rews_da, scale=scale, dim=1)
        rews_qua = bottom_scale(rews_qua, scale=scale, dim=1)
        rews_nov = bottom_scale(rews_nov, scale=scale, dim=1)
        rews_mix = bottom_scale(rews_mix, scale=scale, dim=1)
    
    
    #plt.figure(figsize=(4,8/3), dpi=200)
    plt.figure(figsize=(3,2), dpi=300)
    alpha = 0.1; ms = 4; lw = 0.8; mew = 0.8
    tick_size = 6; label_size = 7
    
    plt.fill_between(range(rews_fs[0].shape[0]), rews_fs[1], rews_fs[2], 
                     color='c', alpha=alpha)
    plt.plot(rews_fs[0], color='c', lw=lw, ls='--')
    
    plt.fill_between(range(rews_da[0].shape[0]), rews_da[1], rews_da[2], 
                     color='darkorange', alpha=alpha)
    plt.plot(rews_da[0], color='darkorange', lw=lw, ls='-')
    
    plt.fill_between(range(rews_nov[0].shape[0]), rews_nov[1], rews_nov[2], 
                     color='g', alpha=alpha)
    plt.plot(rews_nov[0], color='g', lw=lw,
             marker='+', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    plt.fill_between(range(rews_qua[0].shape[0]), rews_qua[1], rews_qua[2], 
                     color='b', alpha=alpha)
    plt.plot(rews_qua[0], color='b', lw=lw,
             marker='^', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    plt.fill_between(range(rews_mix[0].shape[0]), rews_mix[1], rews_mix[2], 
                     color='r', alpha=alpha)
    plt.plot(rews_mix[0], color='r', lw=lw,
             marker='x', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    plt.legend(['FS', 'CA', 'IW-IES-N', 'IW-IES-Qu', 'IW-IES-Mix'], 
               #bbox_to_anchor = (1, 0.28),
               labelspacing=0.01,
               fancybox=True, shadow=True, fontsize=label_size)
    
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5), 
               fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel('Generations', fontsize=label_size)
    plt.ylabel('Return', fontsize=label_size)
    plt.grid(axis='y', ls='-', lw=0.2)
    plt.grid(axis='x', ls='-', lw=0.2)
    
    if DOMAIN == 'navi_v1':
        plt.axis([0, num, -80, 0])
        yticks = [-120, -60, -40, -20, 0]
        plt.yticks(np.arange(-80, 20, 20), yticks)
    elif DOMAIN == 'navi_v2':
        plt.axis([0, num, -180, 0])
        plt.yticks(np.arange(-180, 1, 30))
    elif DOMAIN == 'navi_v3':
        plt.axis([-0, num, -100, 0])
        plt.yticks(np.arange(-100, 20, 20))
    elif DOMAIN == 'hopper':
        plt.axis([0, num, 0, 80])
        yticks = [-75, 20, 40, 60, 80]
        plt.yticks(np.arange(0, 100, 20), yticks)
    elif DOMAIN == 'cheetah':
        plt.axis([0, num, -50, 50])
        yticks = [-100, -25, 0, 25, 50]
        plt.yticks(np.arange(-50, 75, 25), yticks)
    elif DOMAIN == 'swimmer':
        plt.axis([0, num, 0, 25])
        yticks = [-60, 5, 10, 15, 20, 25]
        plt.yticks(np.arange(0, 30, 5), yticks)
    
    return (rews_da)
# data = rews_stats_finetune()
    


###############################################################################
def baselines_stats_finetune():
    if DOMAIN in ['navi_v1', 'hopper']:
        cutoff = 200; num = 100
    elif DOMAIN in ['navi_v2']:
        cutoff = 1000; num = 100
    elif DOMAIN in ['navi_v3']:
        cutoff = 500; num = 100
    elif DOMAIN in ['cheetah']:
        cutoff = 200; num = 50
    elif DOMAIN in ['swimmer']:
        cutoff = 100; num = 50
        
    bs = cutoff // num; mark = num // 10
    rews_robust = np.load(os.path.join(r_path, 'rews_robust_stats.npy'))[:, :cutoff]
    rews_adaptive = np.load(os.path.join(r_path, 'rews_adaptive_stats.npy'))[:, :cutoff]
    rews_so = np.load(os.path.join(r_path, 'rews_so_stats.npy'))[:, :cutoff]
    rews_maml = np.load(os.path.join(r_path, 'rews_maml_stats.npy'))[:, :cutoff]
    rews_mix = np.load(os.path.join(r_path, 'rews_qua_stats.npy'))[:, :cutoff]
    
    rews_robust = arr_ave(rews_robust, bs=bs, dim=1)
    rews_adaptive = arr_ave(rews_adaptive, bs=bs, dim=1)
    rews_maml = arr_ave(rews_maml, bs=bs, dim=1)
    rews_so = arr_ave(rews_so, bs=bs, dim=1)
    rews_mix = arr_ave(rews_mix, bs=bs, dim=1)

    #plt.figure(figsize=(4,8/3), dpi=200)
    plt.figure(figsize=(3,2), dpi=300)
    alpha = 0.1; ms = 4; lw = 0.8; mew = 0.8
    tick_size = 6; label_size = 7
    
    plt.fill_between(range(rews_robust[0].shape[0]), rews_robust[1], rews_robust[2], 
                     color='c', alpha=alpha)
    plt.plot(rews_robust[0], color='c', lw=lw, ls='-')
    
    plt.fill_between(range(rews_so[0].shape[0]), rews_so[1], rews_so[2], 
                     color='darkorange', alpha=alpha)
    plt.plot(rews_so[0], color='darkorange', lw=lw, ls='--')
    
    plt.fill_between(range(rews_adaptive[0].shape[0]), rews_adaptive[1], rews_adaptive[2], 
                     color='g', alpha=alpha)
    plt.plot(rews_adaptive[0], color='g', lw=lw,ls='-.')
    
    plt.fill_between(range(rews_maml[0].shape[0]), rews_maml[1], rews_maml[2], 
                     color='b', alpha=alpha)
    plt.plot(rews_maml[0], color='b', lw=lw, ls=':')
    
    plt.fill_between(range(rews_mix[0].shape[0]), rews_mix[1], rews_mix[2], 
                     color='r', alpha=alpha)
    plt.plot(rews_mix[0], color='r', lw=lw,
             marker='x', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    
    plt.legend(['Robust', 'SO-CMA', 'Hist', 'ES-MAML', 'IW-IES'], 
               #bbox_to_anchor=(1, 0.28),
               labelspacing=0.01,
               fancybox=True, shadow=True, fontsize=label_size)
    
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5), fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel('Generations', fontsize=label_size)
    plt.ylabel('Return', fontsize=label_size)
    plt.grid(axis='y', ls='-', lw=0.2)
    plt.grid(axis='x', ls='-', lw=0.2)
    
    if DOMAIN == 'navi_v1':
        plt.axis([0, num, -60, 0])
        #plt.yticks(np.arange(-60, 20, 20))
        #plt.axis([0, num, -100, 0])
    elif DOMAIN == 'navi_v2':
        plt.axis([0, num, -180, 0])
        plt.yticks(np.arange(-180, 1, 30))
    elif DOMAIN == 'navi_v3':
        plt.axis([0, num, -80, 0])
        plt.yticks(np.arange(-80, 20, 20))
    elif DOMAIN == 'hopper':
        plt.axis([0, num, 0, 80])
    elif DOMAIN == 'cheetah':
        plt.axis([0, num, -75, 50])
        plt.yticks(np.arange(-75, 75, 25))
    elif DOMAIN == 'swimmer':
        plt.axis([0, num, -5, 25])
        #yticks = [-30, 5, 10, 15, 20, 25]
        plt.yticks(np.arange(-5, 30, 5))
    
    return (rews_mix)
#data = baselines_stats_finetune()





###############################################################################
def rews_baselines(N):
    if DOMAIN in ['navi_v1', 'hopper', 'cheetah', 'swimmer']:
        cutoff = 200
    elif DOMAIN in ['navi_v2']:
        cutoff = 1000
    elif DOMAIN in ['navi_v3']:
        cutoff = 500
        
    cutoff = 200
    num = 100; bs = cutoff // num
    rews_da = np.load(os.path.join(r_path, 'rews_da.npy'))[N, :cutoff].mean(axis=0)
    rews_mix = np.load(os.path.join(r_path, 'rews_mix.npy'))[N, :cutoff].mean(axis=0)
    rews_robust = np.load(os.path.join(r_path, 'rews_robust.npy'))[N, :cutoff].mean(axis=0)
    rews_adaptive = np.load(os.path.join(r_path, 'rews_adaptive.npy'))[N, :cutoff].mean(axis=0)
    rews_so = np.load(os.path.join(r_path, 'rews_so.npy'))[N, :cutoff].mean(axis=0)
    rews_maml = np.load(os.path.join(r_path, 'rews_maml.npy'))[N, :cutoff].mean(axis=0)
    #rews_adaptive = rews_robust
    
    print('da: %.2f'%np.mean(rews_da))
    print('robust: %.2f'%np.mean(rews_robust))
    print('adaptive: %.2f'%np.mean(rews_adaptive))
    print('so: %.2f'%np.mean(rews_so))
    print('maml: %.2f'%np.mean(rews_maml))
    print('iwies: %.2f'%np.mean(rews_mix))
    
    rews_da = arr_ave(rews_da, bs=bs)
    rews_mix = arr_ave(rews_mix, bs=bs)
    rews_robust = arr_ave(rews_robust, bs=bs)
    rews_adaptive = arr_ave(rews_adaptive, bs=bs)
    rews_so = arr_ave(rews_so, bs=bs)
    rews_maml = arr_ave(rews_maml, bs=bs)
    
    xx = np.arange(0, num, 1)
    plt.figure(figsize=(6,4))
    plt.plot(xx, rews_robust, xx, rews_adaptive, xx, rews_so, xx, rews_maml, xx, rews_da, xx, rews_mix)
    plt.legend(['robust', 'adaptive', 'so', 'maml', 'da', 'iwies'])
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5), fontsize=10)
    
    return (rews_so)
selected = [0,1,3,4,5,9,11,13,14,17]
#for idx in selected: data = rews_baselines([idx])
#data = rews_baselines(selected)




###############################################################################
def rews_finetune(N):
    if DOMAIN in ['navi_v1', 'hopper', 'cheetah']:
        cutoff = 200
    elif DOMAIN in ['navi_v2']:
        cutoff = 1000
    elif DOMAIN in ['navi_v3', 'ant']:
        cutoff = 500
        
    cutoff = 100
    num = 100; bs = cutoff // num
    rews_fs = np.load(os.path.join(r_path, 'rews_fs.npy'))[N, :cutoff].mean(axis=0)
    rews_da = np.load(os.path.join(r_path, 'rews_da.npy'))[N, :cutoff].mean(axis=0)
    rews_mix = np.load(os.path.join(r_path, 'rews_mix.npy'))[N, :cutoff].mean(axis=0)
    rews_nov = np.load(os.path.join(r_path, 'rews_nov.npy'))[N, :cutoff].mean(axis=0)
    rews_qua = np.load(os.path.join(r_path, 'rews_qua.npy'))[N, :cutoff].mean(axis=0)
    
    print('N =', N)
    print('fs: %.2f'%np.mean(rews_fs))
    print('da: %.2f'%np.mean(rews_da))
    print('nov: %.2f'%np.mean(rews_nov))
    print('qua: %.2f'%np.mean(rews_qua))
    print('iwies: %.2f'%np.mean(rews_mix))
    print('\n')
    
    rews_fs = arr_ave(rews_fs, bs=bs)
    rews_da = arr_ave(rews_da, bs=bs)
    rews_nov = arr_ave(rews_nov, bs=bs)
    rews_qua = arr_ave(rews_qua, bs=bs)
    rews_mix = arr_ave(rews_mix, bs=bs)
    
    xx = np.arange(0, num, 1)
    plt.figure(figsize=(6,4))
    plt.plot(xx, rews_fs, xx, rews_da, xx, rews_nov, xx, rews_qua, xx, rews_mix)
    plt.legend(['fs', 'da', 'nov', 'qua', 'mix'])
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5), fontsize=10)
    
    '''
    plt.figure(figsize=(6,5))
    plt.axis([-1, 1, -1, 1])  
    bc_da = np.load(os.path.join(r_path, 'bc_da.npy'))
    plt.plot(bc_da[0], bc_da[1])
    bc_nov = np.load(os.path.join(r_path, 'bc_nov.npy'))
    plt.plot(bc_nov[0], bc_nov[1])
    bc_qua = np.load(os.path.join(r_path, 'bc_qua.npy'))
    plt.plot(bc_qua[0], bc_qua[1])
    bc_mix = np.load(os.path.join(r_path, 'bc_mix.npy'))
    plt.plot(bc_mix[0], bc_mix[1])
    
    plt.legend(['da', 'nov', 'qua', 'mix'])
    plt.grid(axis='x', ls='--')
    plt.grid(axis='y', ls='--')
    '''
    
    return rews_mix
selected = [0,1,3,4,5,9,11,13,14,17]
#for idx in selected: data = rews_finetune([idx]) 
#data = rews_finetune(selected)








def cal_stats(rews):
    rews_up, rews_down = np.zeros(rews.shape[1]), np.zeros(rews.shape[1])
    
    for i_col in range(rews.shape[1]):
        i_rews = rews[:, i_col]
        i_rews_down, i_rews_up = st.t.interval(0.95, len(i_rews)-1, 
            loc=np.mean(i_rews), scale=st.sem(i_rews))
        rews_up[i_col] = i_rews_up; rews_down[i_col] = i_rews_down
        
    root_data = np.zeros((3, rews.shape[1]))
    root_data[0] = np.mean(rews, axis=0)
    root_data[1] = rews_down; root_data[2] = rews_up
    return root_data 

#rews_priw0 = np.load('per/navi_v2/rewards_per.npy')
#rews_priw = cal_stats(rews_priw0)
# np.save('per/navi_v2/rewards_per_stats.npy', rews_priw)




    
















