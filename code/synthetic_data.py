# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:37:26 2024

@author: cdrg
"""

import numpy as np
from coordinate import WGS84_2_TWD97
from mogi import runMogi_disp
import matplotlib.pyplot as plt
from random import gauss

#==============================================================================
# dir
OUTPUT_PATH = "D:/RA_all/kalman_mogi_apso/output/"

#==============================================================================
# data init
data_points = 2000
time = [2010 + i / 365.25 for i in range(data_points)]

poissons_ratio = 0.25
shear_modulus = 30E9

#==============================================================================
# source
mogi_1_ENU = {'lat':23.45, 'lon':121.15, 'altitude':-4e3, 'radius':2e2}
mogi_2_ENU = {'lat':23.55, 'lon':121.25, 'altitude':-10e3, 'radius':5e2}

mogi_1_TWD = WGS84_2_TWD97(mogi_1_ENU['lat'], mogi_1_ENU['lon'], mogi_1_ENU['altitude'])  
mogi_2_TWD = WGS84_2_TWD97(mogi_2_ENU['lat'], mogi_2_ENU['lon'], mogi_2_ENU['altitude'])  
    
x = np.arange(0, data_points)

std1 = 400
std2 = 300
#std3 = 100

mean1 = 1500
mean2 = 1000
#mean3 = 800

mogi_1_P = 1/(std1 * np.sqrt(2 * np.pi)) * np.exp( - (x - mean1)**2 / (2 * std1**2)) * 1.5e13
mogi_2_P = -1/(std2 * np.sqrt(2 * np.pi)) * np.exp( - (x - mean2)**2 / (2 * std2**2)) * 8e12

#==============================================================================
# grid
study_region = [[121.1, 121.3], [23.4, 23.6]]

lon_data_points = 4
lat_data_points = 4

grid_lon, grid_lat = np.meshgrid(np.linspace(study_region[0][0], study_region[0][1], lon_data_points), \
                                 np.linspace(study_region[1][0], study_region[1][1], lat_data_points))
    
grid_TWD = WGS84_2_TWD97(grid_lat, grid_lon, np.full(grid_lat.shape, 0))

#==============================================================================
mogi_1_disp = []
mogi_2_disp = []

for i in range(data_points):
    Ux1, Uy1, Uz1 = runMogi_disp(grid_TWD[0], grid_TWD[1], 0, \
                              mogi_1_TWD[0], mogi_1_TWD[1], mogi_1_TWD[2], mogi_1_ENU['radius'], \
                              mogi_1_P[i], shear_modulus, poissons_ratio)
        
    Ux2, Uy2, Uz2 = runMogi_disp(grid_TWD[0], grid_TWD[1], 0, \
                              mogi_2_TWD[0], mogi_2_TWD[1], mogi_2_TWD[2], mogi_2_ENU['radius'], \
                              mogi_2_P[i], shear_modulus, poissons_ratio)
        
    mogi_1_disp.append([Ux1, Uy1, Uz1])
    mogi_2_disp.append([Ux2, Uy2, Uz2])
    
    Ux = Ux1 + Ux2 
    Uy = Uy1 + Uy2
    Uz = Uz1 + Uz2
      
    '''
    fig0 = plt.figure(figsize=(8, 18))
    ax0 = plt.subplot2grid((2, 1),(0, 0))
    ax1 = plt.subplot2grid((2, 1),(1, 0))
    
    ax0 .scatter(grid_lon, grid_lat, marker='o', color='k', s=15)   
    ax0 .scatter(mogi_1_ENU['lon'], mogi_1_ENU['lat'], marker='*', color='r', s=200)   
    ax0 .scatter(mogi_2_ENU['lon'], mogi_2_ENU['lat'], marker='*', color='r', s=200)   
    #ax0 .scatter(mogi_3_ENU['lon'], mogi_3_ENU['lat'], marker='*', color='r', s=200)   
    
    ax0.quiver(grid_lon, grid_lat, Ux, Uy, angles='xy', scale=0.2) 
    
    ax1.scatter(grid_lon, grid_lat, marker='o', color='k', s=15)   
    ax1 .scatter(mogi_1_ENU['lon'], mogi_1_ENU['lat'], marker='*', color='r', s=200)   
    ax1 .scatter(mogi_2_ENU['lon'], mogi_2_ENU['lat'], marker='*', color='r', s=200)   
    #ax1 .scatter(mogi_3_ENU['lon'], mogi_3_ENU['lat'], marker='*', color='r', s=200)   
    ax1.quiver(grid_lon, grid_lat, np.full(Uz.shape, 0), Uz, angles='xy', scale=0.2) 
    '''
  
mogi_1_disp_array = np.reshape(np.array(mogi_1_disp), [data_points, 3, lon_data_points * lat_data_points])
mogi_2_disp_array = np.reshape(np.array(mogi_2_disp), [data_points, 3, lon_data_points * lat_data_points])

grid_lon_flat = grid_lon.flatten()
grid_lat_flat = grid_lat.flatten()

for i in range(lon_data_points * lat_data_points):  
    noise_x = np.array([gauss(0, 1.5e-3) for _ in range(data_points)])
    noise_y = np.array([gauss(0, 1.5e-3) for _ in range(data_points)]) 
    noise_z = np.array([gauss(0, 5e-3) for _ in range(data_points)])
    
    annual_list = []
    semi_annual_list = []
    for com in range(3):
        s2_amp = [gauss(0, 4)*1e-3]
        #c2_amp = [gauss(0, 2)*1e-3]
        c2_amp = [gauss(0, 0)*1e-3]
        s4_amp = [gauss(0, 2)*1e-3]
        #c4_amp = [gauss(0, 1)*1e-3]
        c4_amp = [gauss(0, 0)*1e-3]
        
        for t in range(data_points-1):
            s2_amp.append(s2_amp[-1] + gauss(0, 0.005*1e-3))
            #c2_amp.append(c2_amp[-1] + gauss(0, 0.005*1e-3))
            c2_amp.append(c2_amp[-1] + gauss(0, 0.000*1e-3))
            
            s4_amp.append(s4_amp[-1] + gauss(0, 0.001*1e-3))
            #c4_amp.append(c4_amp[-1] + gauss(0, 0.001*1e-3))
            c4_amp.append(c4_amp[-1] + gauss(0, 0.000*1e-3))
        
        annual = np.array([b * np.sin(2 * np.pi * a) +  c * np.cos(2 * np.pi * a) \
                  for a, b, c in zip(time, s2_amp, c2_amp)])
                 
        semi_annual = np.array([b * np.sin(4 * np.pi * a) +  c * np.cos(4 * np.pi * a) \
                  for a, b, c in zip(time, s4_amp, c4_amp)])
            
        annual_list.append(annual)
        semi_annual_list.append(semi_annual)
        
    obs_x = mogi_1_disp_array[:, 0, i] + mogi_2_disp_array[:, 0, i] + \
            annual_list[0] + semi_annual_list[0] + noise_x
            
    obs_y = mogi_1_disp_array[:, 1, i] + mogi_2_disp_array[:, 1, i] + \
            annual_list[1] + semi_annual_list[1] + noise_y
        
    obs_z = mogi_1_disp_array[:, 2, i] + mogi_2_disp_array[:, 2, i] + \
            annual_list[2] + semi_annual_list[2] + noise_z
    
    with open(OUTPUT_PATH + "obs/" + str(grid_lon_flat[i]) + "_" +\
              str(grid_lat_flat[i]) + ".txt", 'w') as f:
        
        f.write("# time    x    y    z\n")
        for a in range(len(time)):
            f.write(str(time[a]) + "    " + str(obs_x[a]) + "    " +\
                    str(obs_y[a]) + "    " + str(obs_z[a]) + "\n")
                
    with open(OUTPUT_PATH + "ground_truth/" + str(grid_lon_flat[i]) + "_" +\
              str(grid_lat_flat[i]) + ".txt", 'w') as f:
        
        f.write("# time mogi_1_P mogi_2_P mogi1_x mogi1_y mogi1_z mogi2_x mogi1_y mogi1_z annual_x annual_y annual_z semi_annual_x semi_annual_y semi_annual_z noise_x noise_y noise_z\n")
                
        for a in range(len(time)):
            f.write(str(time[a]) + "    " + \
                str(mogi_1_P[a]) + "    " +\
                str(mogi_2_P[a]) + "    " +\
                    
                str(mogi_1_disp_array[a, 0, i]) + "    " +\
                str(mogi_1_disp_array[a, 1, i]) + "    " + \
                str(mogi_1_disp_array[a, 2, i]) + "    " + \
                    
                str(mogi_2_disp_array[a, 0, i]) + "    " + \
                str(mogi_2_disp_array[a, 1, i]) + "    " + \
                str(mogi_2_disp_array[a, 2, i]) + "    " + \
                    
                str(annual_list[0][a]) + "    " + \
                str(annual_list[1][a]) + "    " + \
                str(annual_list[2][a]) + "    " +\
                    
                str(semi_annual_list[0][a]) + "    " + \
                str(semi_annual_list[1][a]) + "    " + \
                str(semi_annual_list[2][a]) + "    " + \
                    
                str(noise_x[a])+ "    " + str(noise_y[a])+ "    " + str(noise_z[a])+ "    " + "\n")
    
    