# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:24:33 2024

@author: cdrg
"""

import numpy as np
from mogi_green import mogi_func

class MogiSSM:
    def __init__(self, data_num, data_points, mogi_num):
        self.data_num = data_num
        self.data_points = data_points
        self.mogi_num = mogi_num
        
        self.mu0 = np.zeros([4 * 3 * self.data_num +  2 * self.mogi_num, 1])
        self.mu_dim = self.mu0.shape[0]
        
        self.cov0 = np.eye(self.mu_dim) * 1e5
        '''
        self.cov0 = np.eye(self.mu_dim) * 1e5
        self.cov0[4::, 4::] = 1e-10
        '''
        for i in range(0, 2 * mogi_num, 2):
            self.cov0[i, i] = 1e-6

    #=======================================================================================
    def F_matrix(self, dt):
        F = np.eye(4 * 3 * self.data_num +  2 * self.mogi_num)
        #============================
        # mogi
        for n in range(self.mogi_num):
            F[2*n, 2*n + 1] = dt
            
        return F
    
    #=======================================================================================
    def Q_matrix(self, q_mogi_P_rate, q_seasonal, q_semi_annual):
        # input : variance (not sigma)
        Q = np.eye(4 * 3 * self.data_num +  2 * self.mogi_num) * q_semi_annual
        
        #============================
        # mogi
        for i, a in enumerate(q_mogi_P_rate):
            Q[2*i, 2*i] = 0
            Q[2*i+1, 2*i+1] = a
        
        #============================
        # seasonal
        for i in range(2 * self.mogi_num, Q.shape[0], 4):
            Q[i, i] = q_seasonal
            Q[i+1, i+1] = q_seasonal
            
        return Q
    
    #=======================================================================================
    def H_matrix(self, time, station_TWD, source_TWD, source_radius, poissons_ratio, shear_modulus):
        H = []
        H_mogi = np.zeros([self.data_num * 3, 2 * self.mogi_num])
       
        #============================
        # mogi
        for i, (a, b) in enumerate(zip(source_TWD, source_radius)):
            [Ux, Uy, Uz] = mogi_func(station_TWD[:,0], station_TWD[:,1], station_TWD[:,2],\
                                     *a, b, poissons_ratio, shear_modulus)
        
            H_mogi[:, 2*i] = np.column_stack((Ux, Uy, Uz)).ravel()      
        
        #============================
        # seasonal
        for t in time:
            
            seasonal_tmp = np.array([(np.sin(2*np.pi*t), np.cos(2*np.pi*t),\
                                      np.sin(4*np.pi*t), np.cos(4*np.pi*t))]).reshape([1, 4])
            '''    
            seasonal_tmp = np.array([(0, 0,\
                                      0, 0)]).reshape([1, 4])
            '''    
            H_seasonal = np.kron(np.eye(self.data_num * 3, dtype = int), seasonal_tmp)
            H.append(np.concatenate((H_mogi, H_seasonal), axis = 1))
            
        return H
    
    #=======================================================================================
    def R_matrix(self, x_var, y_var, z_var):
        R = np.eye(self.data_num * 3, self.data_num * 3)
        for i in range(0, self.data_num * 3, 3):
           R[i, i] = x_var
           R[i + 1, i + 1] = y_var
           R[i + 2, i + 2] = z_var
           
        return R
   
            

            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        