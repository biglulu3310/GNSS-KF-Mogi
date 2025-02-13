# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:27:51 2023

@author: cdrg
"""

import numpy as np
import pandas as pd
from MogiKalman import MogiKalman
from constants import DATA_POINTS, DATA_NUM, MOGI_NUM 
class Apso :
    def __init__(self, particle_num, dim, lb, ub, pso_itr): 
        self.G = pso_itr
        self.lb = lb
        self.ub = ub
        self.v_max = (ub-lb)*0.2
        
        # def num & dim
        self.P_num = particle_num  # particle numbers
        self.D_num = dim           # Dimenssion numbers
        
        # Evolutionary state estimation(ESE) rule table
        self.Previous_State = 'S1'
        self.rule_base = pd.DataFrame(data=[['S3', 'S2', 'S2', 'S1', 'S1', 'S1', 'S4'],
                                            ['S3', 'S2', 'S2', 'S2', 'S1', 'S1', 'S4'],
                                            ['S3', 'S3', 'S2', 'S2', 'S1', 'S4', 'S4'],
                                            ['S3', 'S3', 'S2', 'S1', 'S1', 'S4', 'S4']])
        
        self.rule_base.columns = ['S3', 'S3&S2', 'S2', 'S2&S1', 'S1', 'S1&S4', 'S4']
        self.rule_base.index = ['S1', 'S2', 'S3', 'S4']
        
        # Init pso Pbest & Gbest
        self.pbest_X = np.zeros([self.P_num, self.D_num])
        self.pbest_F = np.zeros([self.P_num]) + np.inf
        self.gbest_X = np.zeros([self.D_num])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
        # Init pso parameters
        self.w = 0.9     # inertia
        self.c1 = 2      # self cognition
        self.c2 = 2      # socila influence
        
        self.w_max = 0.9
        self.w_min = 0.4
        
        # output
        self.Q = 0
    
    #================================================================================
    def coss_func(self, dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus, theta):  
        mogi_1 = {"TWD":theta[0:3], "radius":2e2}
        mogi_2 = {"TWD":theta[3:6], "radius":5e2}
        
        q_mogi_dp = theta[6:8]
        q_annaul = theta[8]
        q_semi_annual = theta[9]
        r_var = theta[10::]
        
        mogi_kf = MogiKalman(time, MOGI_NUM , DATA_NUM, DATA_POINTS, poissons_ratio, shear_modulus)
        [mu0, cov0, F, Q, H, R] = mogi_kf.ssm_init(dt, st_twd97, mogi_1, mogi_2, \
                                                   q_mogi_dp, q_annaul, q_semi_annual, r_var)
        
        [_, _, _, innov] = mogi_kf.kalman(mu0, cov0, F, Q, H, R, obs_vector)
        
        return float(innov)
            
    #================================================================================
    def opt(self, dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus):
        # sample
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P_num, self.D_num])
        self.V = -self.v_max + 2*self.v_max*np.random.rand(self.P_num, self.D_num)
        
        # init   
        self.F = np.zeros([self.P_num])
        
        # itr
        for self.g in range(self.G): 
            # cal likelihood for each particle
            for x_i, theta in enumerate(self.X): 
                # mu0 can also be in theta, but too time consuming. 
                self.F[x_i] = self.coss_func(dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus, theta)
                
                # update pbest 
                if self.F[x_i] < self.pbest_F[x_i]:                
                    self.pbest_X[x_i] = self.X[x_i].copy()
                    self.pbest_F[x_i] = self.F[x_i].copy()
             
            # update gbest
            if np.min(self.F) < self.gbest_F:
                idx = self.F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = self.F.min()

            self.loss_curve[self.g] = self.gbest_F   
            
            # ESE
            if self.g<self.G-1:
                self.Ese(dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus, theta)
                
                r1 = np.random.uniform(size=[self.P_num, self.D_num])
                r2 = np.random.uniform(size=[self.P_num, self.D_num])  
                
                # update pos and vel
                self.V = self.w * self.V + self.c1 * (self.pbest_X - self.X) * r1 \
                                         + self.c2 * (self.gbest_X - self.X) * r2
                                         
                self.V = np.clip(self.V, -self.v_max, self.v_max)
                                
                self.X = self.X + self.V 
                self.X = np.clip(self.X, self.lb, self.ub)
                
                
    def Ese(self, dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus, theta):
        #=============================================================== 
        # cal evolutionary factor [0,1]
        d = np.zeros([self.P_num])
        for i in range(self.P_num):
            f1 = np.sum( (self.X[i] - self.X)**2, axis=1 )
            f2 = np.sqrt( f1 )
            f3 = np.sum(f2)
            d[i] = f3/(self.P_num-1)

        idx = self.F.argmin()
        dmax = d.max()
        dmin = d.min()
        dg = d[idx]
        
        if dmax==dmin==dg==0 : f = 0
        elif (dmax-dmin) == 0 : f = 1
        elif (dg-dmin) == 0 : f = 0 
        else : f = (dg-dmin)/(dmax-dmin) 
        
        if not(0<=f<=1):
            if f>1 : f = 1       
            elif f<0 : f = 0 
            
        #===============================================================   
        # cal the stage of the pso
        # Case (a)—Exploration
        if 0.0<=f<=0.4 : uS1 = 0.0
        elif 0.4<f<=0.6 : uS1 = 5*f - 2
        elif 0.6<f<=0.7 : uS1 = 1.0
        elif 0.7<f<=0.8 : uS1 = -10*f + 8
        elif 0.8<f<=1.0 : uS1 = 0.0
        
        # Case (b)—Exploitation
        if 0.0<=f<=0.2 : uS2 = 0
        elif 0.2<f<=0.3 : uS2 = 10*f - 2
        elif 0.3<f<=0.4 : uS2 = 1.0
        elif 0.4<f<=0.6 : uS2 = -5*f + 3
        elif 0.6<f<=1.0 : uS2 = 0.0
        
        # Case (c)—Convergence
        if 0.0<=f<=0.1 : uS3 = 1.0
        elif 0.1<f<=0.3 : uS3 = -5*f + 1.5
        elif 0.3<f<=1.0 : uS3 = 0.0
        
        # Case (d)—Jumping Out
        if 0.0<=f<=0.7 : uS4 = 0.0
        elif 0.7<f<=0.9 : uS4 = 5*f - 3.5
        elif 0.9<f<=1.0 : uS4 = 1.0
            
        # =============================================================================
        if uS3!=0:
            Current_State = 'S3'
            if uS2!=0:
                Current_State = 'S3&S2'
        elif uS2!=0:
            Current_State = 'S2'
            if uS1!=0:
                Current_State = 'S2&S1'
        elif uS1!=0:
            Current_State = 'S1'
            if uS4!=0:
                Current_State = 'S1&S4'
        elif uS4!=0:
            Current_State = 'S4'
        
        Final_State = self.rule_base[Current_State][self.Previous_State]
        self.Previous_State = Final_State
        
        delta = np.random.uniform(low=0.05, high=0.1, size=2)
        
        if Final_State=='S1': # Exploration
            self.c1 = self.c1 + delta[0]
            self.c2 = self.c2 - delta[1]
        elif Final_State=='S2': # Exploitation
            self.c1 = self.c1 + 0.5*delta[0]
            self.c2 = self.c2 - 0.5*delta[1]
        elif Final_State=='S3': # Convergence
            self.c1 = self.c1 + 0.5*delta[0]
            self.c2 = self.c2 + 0.5*delta[1]
            self.Els(dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus, theta)
        elif Final_State=='S4': # Jumping Out
            self.c1 = self.c1 - delta[0]
            self.c2 = self.c2 + delta[1]
            
        self.c1 = np.clip(self.c1, 1.5, 2.5)
        self.c2 = np.clip(self.c2, 1.5, 2.5)
        if (3.0<=self.c1+self.c2<=4.0)==False:
            self.c1 = 4.0 * self.c1/(self.c1+self.c2)
            self.c2 = 4.0 * self.c2/(self.c1+self.c2)

        self.w = 1/(1+1.5*np.exp(-2.6*f))
        self.w = np.clip(self.w, self.w_min, self.w_max)                

    def Els(self, dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus, theta):
        P = self.gbest_X.copy()
        d = np.random.randint(low=0, high=self.D_num)
        
        mu = 0
        sigma = 1 - 0.9*self.g/self.G
        P[d] = P[d] + (self.ub[d]-self.lb[d])*np.random.normal(mu, sigma**2)
        
        P = np.clip(P, self.lb, self.ub)
        
        v = self.coss_func(dt, time, st_twd97, obs_vector, poissons_ratio, shear_modulus, theta)

        if v < self.gbest_F:
            self.gbest_X = P.copy()
            self.gbest_F = v
          
        # replace worst particle
        elif v<self.F.max():
            idx = self.F.argmax()
            self.X[idx] = P.copy()
            self.F[idx] = v    












        