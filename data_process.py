#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as st



###############################################################################
DOMAIN = 'navi_v1'
r_path = 'output/%s'%DOMAIN
s_path = 'saves/%s'%DOMAIN
    

###############################################################################
def ablation_study():
    rews_fs = np.load(os.path.join(r_path, 'rews_fs.npy'))
    rews_ca = np.load(os.path.join(r_path, 'rews_ca.npy'))
    rews_nov = np.load(os.path.join(r_path, 'rews_nov.npy'))
    rews_qua = np.load(os.path.join(r_path, 'rews_qua.npy'))
    rews_mix = np.load(os.path.join(r_path, 'rews_mix.npy'))
    
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
    
ablation_study()


###############################################################################
def comparison_to_baselines():
    rews_robust = np.load(os.path.join(r_path, 'rews_robust.npy'))
    rews_so = np.load(os.path.join(r_path, 'rews_so.npy'))
    rews_hist = np.load(os.path.join(r_path, 'rews_hist.npy'))
    rews_maml = np.load(os.path.join(r_path, 'rews_maml.npy'))
    rews_mix = np.load(os.path.join(r_path, 'rews_mix.npy'))

    print('Average return of Robust: %.2f' % rews_robust.mean())
    print('Average return of SO-CAM: %.2f' % rews_so.mean())
    print('Average return of Hist: %.2f' % rews_hist.mean())
    print('Average return of ES-MAML: %.2f' % rews_maml.mean())
    print('Average return of IW-IES: %.2f' % rews_mix.mean())


    x = range(len(rews_robust))
    plt.figure()
    plt.plot(x, rews_robust, x, rews_so, x, rews_hist, x, rews_maml, x, rews_mix)
    plt.legend(['Robust', 'SO-CMA', 'Hist', 'ES-MAML', 'IW-IES'])  
    plt.xlabel('Generations', fontsize=15)
    plt.ylabel('Return', fontsize=15)
    # plt.title('Navigation Case Complex')

comparison_to_baselines()























    
















