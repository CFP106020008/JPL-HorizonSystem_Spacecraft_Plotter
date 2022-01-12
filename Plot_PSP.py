# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 21:10:11 2022

@author: juliu

Inspired by https://en.wikipedia.org/wiki/File:Velocity_of_Parker_Solar_Probe_wide.svg

## Some notes
When using the horizon system, you better use:
Ephemeris Type: Vector table
Coordinate Center: Sun (body center) # Or some better position I don't know...
Reference frame: FK4, CSV format checked,
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# To make the plot COOL
plt.style.use('dark_background')
raw_path = "PSP.txt"
goodfile_path = "PSP.csv"
finalimage_path = "PSP.png"
Figfacecolor = '#333333'
Axfacecolor = '#1c1c1c'
color_r = 'yellow'
color_v = 'skyblue'

r_E = 1.5e8

def transform_raw(raw_path):
    # Here we transform the original data from 
    # NASA JPL Horizon System to a format 
    # that is easy to read for pandas
    text = open(raw_path, encoding="utf-8").readlines()
    LS = text.index("$$SOE\n")
    LE = text.index("$$EOE\n")

    Labels = text[LS-2].split(sep=',')
    for i in range(len(Labels)):
        Labels[i] = Labels[i].strip()
    Labels = ",".join(Labels)
    
    text2 = list(Labels+'\n') + list(text[LS+1:LE])
    
    f2 = open(goodfile_path,'w')
    for line in text2:
        f2.write(line)
    f2.close()
    return 

def Plot_rv(goodfile_path):
    # Now to plot the data we want
    
    D = pd.read_csv(goodfile_path)
    fig, ax1 = plt.subplots(facecolor=Figfacecolor) # For r
    ax2 = ax1.twinx() # For v
    
    # Calculate v and r from raw data
    D['r'] = np.sqrt(D.X**2 + D.Y**2 + D.Z**2)
    D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
    
    # Add a "Mission time" column
    D['days'] = D['JDTDB'] - D['JDTDB'][0]
    
    # Plot r part
    ax1.plot(D.days, D.r, color=color_r)
    ax1.set_facecolor(Axfacecolor)
    ax1.set_xlabel("Mission time (days)")
    ax1.set_ylabel("Distance to Sun (km)")
    ax1.set_xlim([0, np.array(D['days'])[-1]])
    
    # Plot v part
    ax2.plot(D.days, D.v, color=color_v)
    ax2.set_ylabel("Velocity relative to Sun (km/s)")
    
    plt.tight_layout()
    plt.savefig(finalimage_path, dpi=300, facecolor='#333333')
    plt.show()

def Plot_phasespace(goodfile_path):
    D = pd.read_csv(goodfile_path)
    fig, ax = plt.subplots(facecolor=Figfacecolor)
    
    D['r'] = np.sqrt(D.X**2 + D.Y**2 + D.Z**2)
    D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
    
    ax.plot(D['r'], D['v'], color='yellow')
    #plt.show()
    
def Plot_xy(goodfile_path):
    D = pd.read_csv(goodfile_path)
    fig, ax = plt.subplots(facecolor=Figfacecolor)
    
    ax.plot(D['X'], D['Y'], color='yellow')
    ax.set_aspect('equal', adjustable='box')
    #plt.show()    
    
def Plot_xyz(goodfile_path, ax, i):
    D = pd.read_csv(goodfile_path)
    #fig = plt.figure(facecolor=Figfacecolor, figsize=(6,6))
    #ax = fig.add_subplot(projection='3d', facecolor=Axfacecolor)
    
    rmax = np.max((D['X']**2 + D['Y'] + D['Z']**2)**0.5)
    
    arrow_scale = 1.5
    
    ax.quiver(-arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, 0, 0, color='w', arrow_length_ratio=0.05) # x-axis
    ax.quiver(0, -arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, 0, color='w', arrow_length_ratio=0.05) # y-axis
    ax.quiver(0, 0, -arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, color='w', arrow_length_ratio=0.05) # z-axis
    
    ax.plot(D['X'][:i], D['Y'][:i], D['Z'][:i], color='yellow')
    ax.scatter(D['X'][i], D['Y'][i], D['Z'][i], color='yellow', s=10)
    
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.scatter(0,0,0,color='yellow',s=100)
    theta = np.linspace(0, 2*np.pi, int(1e3))
    ax.plot(r_E*np.cos(theta), r_E*np.sin(theta), color='blue')
    
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.set_xlim([-rmax, rmax])
    ax.set_ylim([-rmax, rmax])
    ax.set_zlim([-rmax, rmax])
    plt.tight_layout()
    #plt.show()
    return ax

def Plot_r(good_path, ax, i=0, Dot=False):
    
    D = pd.read_csv(goodfile_path)
    
    # Calculate v and r from raw data
    D['r'] = np.sqrt(D.X**2 + D.Y**2 + D.Z**2)
    D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
    
    # Add a "Mission time" column
    D['days'] = D['JDTDB'] - D['JDTDB'][0]
    
    # Plot r part
    ax.plot(D.days[:i], D.r[:i], color=color_r)
    ax.set_facecolor(Axfacecolor)
    ax.set_xlabel("Mission time (days)")
    ax.set_ylabel("Distance to Sun (km)")
    ax.set_xlim([0, np.array(D['days'])[-1]])
    
    ax.scatter(D.days[i], D.r[i], s=10, color=color_v)
    
    plt.tight_layout()
    #plt.savefig(finalimage_path, dpi=300, facecolor='#333333')
    #plt.show()
    
    return ax

def Plot_v(good_path, ax, i=0, Dot=False):
    
    D = pd.read_csv(goodfile_path)
    D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
    
    # Add a "Mission time" column
    D['days'] = D['JDTDB'] - D['JDTDB'][0]
    
    ax.set_xlim([0, np.array(D['days'])[-1]])
    
    # Plot v part
    ax.plot(D.days[:i], D.v[:i], color=color_v)
    ax.set_facecolor(Axfacecolor)
    ax.set_xlabel("Mission time (days)")
    ax.set_ylabel("Velocity relative to Sun (km/s)")
    
    ax.scatter(D.days[i], D.v[i], s=10, color=color_v)
    
    #plt.tight_layout()
    #plt.savefig(finalimage_path, dpi=300, facecolor='#333333')
    #plt.show()
    
    return ax

def Complex_1(i, Dot=True):
    fig = plt.figure(figsize=(12,6), facecolor=Figfacecolor)
    gs = GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[:,:2],projection='3d', facecolor=Axfacecolor)
    ax1 = Plot_xyz(goodfile_path, ax1, i)
    ax2 = fig.add_subplot(gs[0,2:])
    ax2 = Plot_r(goodfile_path, ax2, i)
    ax3 = fig.add_subplot(gs[1,2:])
    ax3 = Plot_v(goodfile_path, ax3, i)
    plt.tight_layout()
    plt.savefig("./{:04d}.png".format(i), dpi=300, facecolor=Figfacecolor)
    
def main():
    transform_raw(raw_path)
    D = pd.read_csv(goodfile_path)
    N = D['X'].size
    for i in range(5):
        Complex_1(i)
    #Plot_rv(goodfile_path)
    #Plot_xy(goodfile_path)
    #Plot_phasespace(goodfile_path)
    #Plot_xyz(goodfile_path)
    
    #plt.show()

main()