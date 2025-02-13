# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:38:26 2024

@author: cdrg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import math

from coordinate import WGS84_2_TWD97, TWD97_2_WGS84
from constants import STUDY_REGION, DATA_POINTS, DATA_NUM, MOGI_NUM 
from Apso import Apso
from MogiKalman import MogiKalman
from Kalman import Kalman
#====================================================================
# data init
PIC_OUTPUT_PATH = 'D:/RA_all/kalman_mogi_apso/output/pic/'
OBS_PATH = "D:/RA_all/kalman_mogi_apso/output/obs/"
data_filename_list = os.listdir(OBS_PATH) 

dt =  1/365.25
poissons_ratio = 0.25
shear_modulus = 30E9

# true value
mogi_1_ENU = {'lat':23.45, 'lon':121.15, 'altitude':-4e3, 'radius':2e2}
mogi_2_ENU = {'lat':23.55, 'lon':121.25, 'altitude':-10e3, 'radius':5e2}
#====================================================================
# func
def import_data(data_path, filename):
    with open(data_path + filename) as file:
        data = []
        for line in file:
            data.append(line.rstrip().split())
            
    return data

#====================================================================
# apso setting
particle_num = 30
apso_itr = 20
dim = 13

#apso_para = ["mogi_TWD", "q_mogi", "q_annual", "q_semi_annual", "r_var"]
# lower bound
apso_mogi_TWD_lb = list(WGS84_2_TWD97(STUDY_REGION["lat_lb"], STUDY_REGION["lon_lb"], -15e3)) * MOGI_NUM
apso_q_mogi_lb = [1e0 for _ in range(MOGI_NUM)]

apso_q_sesonal_lb = [1e-16, 1e-17]
apso_r_var_lb = [1e-8, 1e-8, 1e-8]

apso_para_lb = np.array(apso_mogi_TWD_lb + apso_q_mogi_lb + apso_q_sesonal_lb + apso_r_var_lb)

# upper bound
apso_mogi_TWD_ub =  list(WGS84_2_TWD97(STUDY_REGION["lat_ub"], STUDY_REGION["lon_ub"], -1e-5))* MOGI_NUM
apso_q_mogi_ub = [5e13 for _ in range(MOGI_NUM)]

apso_q_sesonal_ub = [1e-11, 1e-12]
apso_r_var_ub = [1e-4, 1e-4, 1e-4]

apso_para_ub = np.array(apso_mogi_TWD_ub + apso_q_mogi_ub + apso_q_sesonal_ub + apso_r_var_ub)

del apso_mogi_TWD_lb, apso_q_mogi_lb, apso_q_sesonal_lb, apso_r_var_lb, \
    apso_mogi_TWD_ub, apso_q_mogi_ub, apso_q_sesonal_ub, apso_r_var_ub

#====================================================================
# import data
lon_list = []
lat_list = []
st_twd97_list = []
st_twd97_array = np.zeros([DATA_NUM, 3])
obs_vector = []

for i, filename in enumerate(data_filename_list):
    lon, lat = os.path.splitext(filename)[0].split("_")
    st_twd97 = WGS84_2_TWD97(lat, lon, 0)
    st_twd97_list.append(st_twd97)
    st_twd97_array[i] = st_twd97
    
    lon_list.append(round(float(lon), 3))
    lat_list.append(round(float(lat), 3))
    
    data = np.array(import_data(OBS_PATH, filename)[1::], dtype=float).T
    time, obs = data[0], data[1::]
    obs_vector.append(obs)
 
obs_vector = np.array(obs_vector).reshape([DATA_NUM*3, DATA_POINTS])
    
apso = Apso(particle_num, dim, apso_para_lb, apso_para_ub, apso_itr)
apso.opt(dt, time, st_twd97_array, obs_vector, poissons_ratio, shear_modulus)

best_history = apso.loss_curve
gbest = apso.gbest_X   
    
#====================================================================
mogi_kf = MogiKalman(time, MOGI_NUM, DATA_NUM, DATA_POINTS, poissons_ratio, shear_modulus)


mogi_1 = {"TWD":gbest[0:3], "radius":2e2, "WGS84":TWD97_2_WGS84(*gbest[0:3])}
mogi_2 = {"TWD":gbest[3:6], "radius":5e2, "WGS84":TWD97_2_WGS84(*gbest[3:6])}

q_mogi_dp = gbest[6:8]
q_annaul = gbest[8]
q_semi_annual = gbest[9]
r_var = gbest[10::]

'''
aa = gbest.copy()

aa[0:3] = WGS84_2_TWD97(mogi_1_ENU['lat'], mogi_1_ENU['lon'], mogi_1_ENU['altitude'])
aa[3:6] = WGS84_2_TWD97(mogi_2_ENU['lat'], mogi_2_ENU['lon'], mogi_2_ENU['altitude'])

aa[-5] = (0.005*1e-3)**2
aa[-4] = (0.001*1e-3)**2
aa[-3] = 1.5e-3**2
aa[-2] = 1.5e-3**2
aa[-1] = 5e-3**2

mogi_1 = {"TWD":aa[0:3], "radius":2e2, "WGS84":TWD97_2_WGS84(*aa[0:3])}
mogi_2 = {"TWD":aa[3:6], "radius":5e2, "WGS84":TWD97_2_WGS84(*aa[3:6])}

q_mogi_dp = aa[6:8]
q_annaul = aa[8]
q_semi_annual = aa[9]
r_var = aa[10::]
'''

[mu0, cov0, F, Q, H, R] = mogi_kf.ssm_init(dt, st_twd97_array, mogi_1, mogi_2, \
                                           q_mogi_dp, q_annaul, q_semi_annual, r_var)
    
[filtered, mu, cov, innov] = mogi_kf.kalman(mu0, cov0, F, Q, H, R, obs_vector)    
    
kf = Kalman(mu0, cov0) 
rts_mu, rts_P, rts_y_mean = kf.rts(mu, cov, H, F, Q)    
    
    

fig = plt.figure(figsize = (12, 8))
ax0 = plt.subplot2grid((1, 1),(0, 0))

ax0.plot(time, obs_vector[0], 'o', c='k', ms=2)
ax0.plot(time, rts_y_mean[:,0,0], c='r', lw=2)    

    
       
x = np.arange(0, DATA_POINTS)
std1 = 400
std2 = 300

mean1 = 1500
mean2 = 1000

mogi_1_P = 1/(std1 * np.sqrt(2 * np.pi)) * np.exp( - (x - mean1)**2 / (2 * std1**2)) * 1.5e13
mogi_2_P = -1/(std2 * np.sqrt(2 * np.pi)) * np.exp( - (x - mean2)**2 / (2 * std2**2)) * 8e12
    
    
for i in range(0,DATA_POINTS,60):
    fig = plt.figure(figsize = (48, 48))
    ax0 = plt.subplot2grid((4, 4),(0, 0), rowspan=2, colspan=2)
    ax1 = plt.subplot2grid((4, 4),(2, 0), colspan=2)
    
    ax2 = plt.subplot2grid((4, 4),(0, 2), colspan=2)
    ax3 = plt.subplot2grid((4, 4),(1, 2), colspan=2)
    ax4 = plt.subplot2grid((4, 4),(2, 2), colspan=2)
    
    ax5 = plt.subplot2grid((4, 4),(3, 0), colspan=2)
    ax6 = plt.subplot2grid((4, 4),(3, 2), colspan=2)
    
    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
    
    #==================================================
    ax0.plot(np.array(lon_list), np.array(lat_list), '^', c='k', ms=28)
    ax0.plot(mogi_1_ENU['lon'], mogi_1_ENU['lat'], '*', c='r', markerfacecolor='gold', ms=50)   
    ax0.plot(mogi_2_ENU['lon'], mogi_2_ENU['lat'], '*', c='r', markerfacecolor='gold', ms=50) 
    
    ax0.plot(mogi_1['WGS84'][1], mogi_1['WGS84'][0], '*', c='b', markerfacecolor='w', ms=50)   
    ax0.plot(mogi_2['WGS84'][1], mogi_2['WGS84'][0], '*', c='b', markerfacecolor='w', ms=50) 
    
    ax0.quiver(np.array(lon_list), np.array(lat_list), obs_vector[::3,i], obs_vector[1::3,i], \
               scale=0.5, label = 'obs') 
    
    ax0.quiver(np.array(lon_list), np.array(lat_list), \
               rts_y_mean[i][0::3], rts_y_mean[i][1::3],\
               color='r', scale=0.5, label = 'RTS') 
    
    ax0.text(mogi_1['WGS84'][1]+0.005, mogi_1['WGS84'][0]+0.003, \
             'Estmated depth : ' + str(-round(mogi_1['WGS84'][2]/1e3,2)) + ' km', c='r',fontsize=35)
        
    ax0.text(mogi_2['WGS84'][1]+0.005, mogi_2['WGS84'][0]+0.003, \
             'Estmated depth : ' + str(-round(mogi_2['WGS84'][2]/1e3,2)) + ' km', c='r', fontsize=35)
        
    ax0.text(mogi_1_ENU['lon']+0.005, mogi_1_ENU['lat']+0.003, \
             'True depth : ' + str(-round(mogi_1_ENU['altitude']/1e3,2)) + ' km', c='k',fontsize=35)
        
    ax0.text(mogi_2_ENU['lon']+0.005, mogi_2_ENU['lat']+0.003, \
             'True depth : ' + str(-round(mogi_2_ENU['altitude']/1e3,2)) + ' km', c='k', fontsize=35)
    #==================================================
    
    ax1.plot(time[0:i+1], mogi_1_P[0:i+1]/1e9, 'o', c='k', ms=10, label='ground truth')
    ax1.plot(time[0:i+1], rts_mu[0:i+1, 0, 0]/1e9, c='r', lw=4, label='RTS')
    
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    ax5.plot(time[0:i+1], mogi_2_P[0:i+1]/1e9, 'o', c='k', ms=10, label='ground truth')
    ax5.plot(time[0:i+1], rts_mu[0:i+1, 2, 0]/1e9, c='r', lw=4, label='RTS')
    ax5.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    #==================================================
    ax2.plot(time, obs_vector[0], 'o', c='k', ms=8, label='obs')
    ax2.plot(time[0:i+1], rts_y_mean[0:i+1, 0, 0], c='r', lw=6, label='RTS')
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    ax3.plot(time, obs_vector[8], 'o', c='k', ms=8, label='obs')
    ax3.plot(time[0:i+1], rts_y_mean[0:i+1, 8, 0], c='r', lw=6, label='RTS')
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    ax4.plot(time, obs_vector[22], 'o', c='k', ms=8, label='obs')
    ax4.plot(time[0:i+1], rts_y_mean[0:i+1, 22, 0], c='r', lw=6, label='RTS')
    ax4.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    ax6.plot(time, obs_vector[30], 'o', c='k', ms=8, label='obs')
    ax6.plot(time[0:i+1], rts_y_mean[0:i+1, 30, 0], c='r', lw=6, label='RTS')
    ax6.yaxis.set_minor_locator(AutoMinorLocator(5)) 
    
    #==================================================
    [ax.xaxis.set_major_locator(MultipleLocator(1)) for ax in axes if ax!=ax0]
    [ax.xaxis.set_minor_locator(MultipleLocator(1/12)) for ax in axes if ax!=ax0]
    
    [ax.spines[a].set_linewidth(3.5) for a in ['top','bottom','left','right'] for ax in axes]
    [ax.spines[a].set_color("black") for a in ['top','bottom','left','right'] for ax in axes]
    
    [ax.tick_params(axis='both', which='major',direction='in',\
                            bottom=True,top=True,right=True,left=True,\
                            length=21, width=4.5, labelsize=28, pad=20) for ax in axes]
        
    [ax.tick_params(axis='both', which='minor',direction='in',\
                    bottom=True,top=True,right=True,left=True,\
                    length=9, width=3, labelsize=28, pad=20)  for ax in axes]
    
    ax0.set_title("Ground Motion", fontsize=60)   
    ax1.set_title(u'\u0394'+"P", fontsize=60)  
    ax5.set_title(u'\u0394'+"P", fontsize=60)
    
    ax2.set_title(str(lon_list[0]) + ", " + str(lat_list[0]), fontsize=45)   
    ax3.set_title(str(lon_list[math.ceil(8/3)]) + ", " + str(lat_list[math.ceil(8/3)]), fontsize=45)  
    ax4.set_title(str(lon_list[math.ceil(22/3)]) + ", " + str(lat_list[math.ceil(22/3)]), fontsize=45)
    ax6.set_title(str(lon_list[math.ceil(30/3)]) + ", " + str(lat_list[math.ceil(30/3)]), fontsize=45)
    
    [ax.set_xlabel('time in year', fontsize = 45, labelpad=20) for ax in axes if ax!=ax0]
    [ax.set_ylabel('displacement (m)', fontsize = 45, labelpad=20) for ax in axes if ax!=ax0]
    ax1.set_ylabel('GPa', fontsize = 45, labelpad=20)
    ax5.set_ylabel('GPa', fontsize = 45, labelpad=20)
 
    [ax.set_xlim(time[0]-0.1, time[-1]+0.1) for ax in axes if ax!=ax0]
    ax1.set_ylim(-3e9/1e9, 1.8e10/1e9)
    ax5.set_ylim(-1.5e10/1e9, 3e9/1e9)
    
    [ax.legend(fontsize=32) for ax in axes]
    plt.tight_layout() 
    
    plt.savefig(PIC_OUTPUT_PATH + str(i) + '.png', dpi=100)
    plt.close()
    
    
    
    
    
    






