# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:20:10 2024

@author: cdrg
"""

import numpy as np

from MogiSSM import MogiSSM
from Kalman import Kalman
#import time

class MogiKalman:
    def __init__(self, time_, mogi_num, data_num, data_points, poissons_ratio, shear_modulus):
        self.time = time_
        self.mogi_num = mogi_num
        self.data_num = data_num
        self.data_points = data_points
        self.poissons_ratio = poissons_ratio
        self.shear_modulus = shear_modulus
    
    def ssm_init(self, dt, st_twd97, mogi_1, mogi_2, q_mogi_dp, q_annaul, q_semi_annual, r_var):
        ssm = MogiSSM(self.data_num, self.data_points, self.mogi_num)
        
        mu0 = ssm.mu0
        cov0 = ssm.cov0
        
        F = ssm.F_matrix(dt)
        Q = ssm.Q_matrix(q_mogi_dp, q_annaul, q_semi_annual)
        H = ssm.H_matrix(self.time, st_twd97, [mogi_1["TWD"], mogi_2["TWD"]], \
                         [mogi_1['radius'], mogi_2['radius']], \
                         self.poissons_ratio, self.shear_modulus)
            
        R = ssm.R_matrix(r_var[0], r_var[1], r_var[2])
        
        return [mu0, cov0, F, Q, H, R]
    
    def kalman(self, mu0, cov0, F, Q, H, R, obs_vector):
        #a = time.time()
        kf = Kalman(mu0, cov0)
        
        filtered = np.zeros([self.data_num * 3, self.data_points])
        mu = np.zeros([self.data_points, 4 * 3 * self.data_num + 2 * self.mogi_num, 1])
        cov = np.zeros([self.data_points, 4 * 3 * self.data_num + 2 * self.mogi_num, \
                                          4 * 3 * self.data_num + 2 * self.mogi_num])
        
        for i in range(self.data_points):
            kf.correct(self.time[i], obs_vector[:,i].reshape([3 * self.data_num, 1]) , H[i], R) 
            #print(kf.filter_value.shape)
            filtered[:,i] = kf.filter_value.flatten()
            mu[i] = kf.x
            cov[i] = kf.P
            if i < self.data_points - 1:         
                kf.predict(self.time[i], F, Q)
        
        innov = kf.innovation_likelihood
        #b = time.time()
        #print(b - a)
        return [filtered, mu, cov, innov]

    
        
        
        