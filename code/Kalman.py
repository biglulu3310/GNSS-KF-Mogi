# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:27:07 2024

@author: cdrg
"""

import numpy as np

'''
#from numba.experimental import jitclass
#from numba import jit, float64 

spec = [('x', float64[:, ::1]),\
        ('P', float64[:, :]),\
        ('innovation_likelihood', float64), \
        ('filter_value', float64[:, ::1]), \
        ('filter_cov', float64[:, :])]

#@jitclass(spec, nopython=True)
'''
class Kalman:
    def __init__(self, mu0, cov0):
        self.x = mu0
        self.P = cov0
        self.innovation_likelihood = 0
    
    def predict(self, time, F, Q) -> None:
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
    
    def correct(self, time, observation, H, R) -> None:
        v = observation - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ v
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        
        # over flow det
        # self.innovation_likelihood += np.log(np.linalg.det(S)) + v.T @ np.linalg.inv(S) @ v
        self.innovation_likelihood = self.innovation_likelihood + np.linalg.slogdet(S)[1] + (v.T @ np.linalg.inv(S) @ v)[0,0]
        
        self.filter_value = H @ self.x
        self.filter_cov = H @ self.P @ H.T
                    
    def rts(self, mu, cov, H, F, Q):
        x, P, P_pred, H_sm = mu.copy(), cov.copy(), cov.copy(), H.copy()
        
        time_dim = len(x)      
        K = np.zeros((time_dim,x.shape[1],x.shape[1])) 
        rts_y_mean = np.zeros([time_dim, H_sm[0].shape[0], 1])
        
        for k in range(time_dim-1, -1, -1):
            if k < time_dim-1:
                P_pred[k] = F @ P[k] @ F.T + Q
                K[k]  = P[k] @ F.T @ np.linalg.inv(P_pred[k])
                x[k] += K[k] @ (x[k+1] - (F @ x[k]))
                P[k] += K[k] @ (P[k+1] - P_pred[k]) @ K[k].T
                           
            rts_y_mean[k] = H_sm[k] @ x[k]
             
        return x, P, rts_y_mean        
   
            
            
            
            
            
        
        