# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 21:33:19 2022

@author: juliu

Inspired by https://en.wikipedia.org/wiki/File:Velocity_of_Parker_Solar_Probe_wide.svg

## Some notes
When using the horizon system, you better use:
Ephemeris Type: Vector table
Coordinate Center: Earth (body center) # Or some better position I don't know...
Reference frame: ICRF, CSV format checked,
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from matplotlib.patches import Rectangle
import path
from os import walk
from multiprocessing import Process, Pool

# To make the plot COOL
plt.style.use('dark_background')
raw_path = "Orion.txt"
goodfile_path = "Orion.csv"
Moon_raw = "Moon.txt"
Moon_path = "Moon.csv"
finalimage_path = "Orion.png"
Figfacecolor = '#333333'
Axfacecolor = '#1c1c1c'
color_r = 'yellow'
color_v = 'skyblue'

def transform_raw(raw_path, goodfile_path):
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
    ax1.set_ylabel("Distance to the Earth (km)")
    ax1.set_xlim([0, np.array(D['days'])[-1]])
    
    # Plot v part
    ax2.plot(D.days, D.v, color=color_v)
    ax2.set_ylabel("Velocity relative to the Earth (km/s)")
    
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
    
def Plot_xyz(goodfile_path, ax, i, Name, Color='yellow', Axis=False, start=0):
    D = pd.read_csv(goodfile_path)
    #fig = plt.figure(facecolor=Figfacecolor, figsize=(6,6))
    #ax = fig.add_subplot(projection='3d', facecolor=Axfacecolor)
    
    #rmax = np.max((D['X'][:i]**2 + D['Y'][:i]**2 + D['Z'][:i]**2)**0.5)
    rmax = np.max((D['X']**2 + D['Y']**2 + D['Z']**2)**0.5)
    
    ax.plot(D['X'][start:i], 
            D['Y'][start:i], 
            D['Z'][start:i], 
            color=Color,
            )
    ax.scatter( D['X'][i], 
                D['Y'][i], 
                D['Z'][i], 
                color=Color, 
                s=10,
                label=Name)
    ax.legend()
    
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
    
    ax.set_box_aspect([1,1,1])
    #plt.tight_layout()
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
    #ax.set_xlabel("Mission time (days)")
    ax.set_ylabel("Distance to the Earth (km)")
    ax.set_xlim([0, np.array(D['days'])[-1]])
    ax.set_xticks([])
    ax.set_ylim([0,np.max(D['r'])*1.1])
    
    ax.scatter(D.days[i], D.r[i], s=10, color=color_v)
    
    ax.ticklabel_format(style='sci', useMathText=True, scilimits=(4,5))
    
    #plt.tight_layout()
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
    ax.set_ylabel("Velocity relative to the Earth (km/s)")
    ax.set_ylim([0,np.max(D['v'])*1.1])
    
    ax.scatter(D.days[i], D.v[i], s=10, color=color_v)
    
    ax.ticklabel_format(style='sci', useMathText=True, scilimits=(4,5))
    
    #plt.tight_layout()
    #plt.savefig(finalimage_path, dpi=300, facecolor='#333333')
    #plt.show()
    
    return ax

def decorate(ax):
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    D = pd.read_csv(goodfile_path)
    rmax = np.max((D['X']**2 + D['Y']**2 + D['Z']**2)**0.5)
    arrow_scale = 1.5
    ax.quiver(-arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, 0, 0, color='w', arrow_length_ratio=0.05, zorder=0) # x-axis
    ax.quiver(0, -arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, 0, color='w', arrow_length_ratio=0.05, zorder=0) # y-axis
    ax.quiver(0, 0, -arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, color='w', arrow_length_ratio=0.05, zorder=0) # z-axis
    '''
    N = 200
    stride = 1
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    r = 6731 # km
    x = r*np.outer(np.cos(u), np.sin(v))
    y = r*np.outer(np.sin(u), np.sin(v))
    z = r*np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=10, color='cyan',zorder=1)
    '''
    ax.scatter(0,0,0,color='cyan',s=100,zorder=10)
    return ax
    

def Complex_1(i, Dot=True):
    #print('Now working on frame {}'.format(i))
    fig = plt.figure(figsize=(32/3,6), facecolor=Figfacecolor)
    gs = GridSpec(2, 4, figure=fig,
                  left = 0.0625/1.5,
                  right = 1-0.0625/1.5,
                  top = 1-1/9/1.5,
                  bottom = 1/9/1.5,
                  wspace = 4/15,
                  hspace = 0.,
                  width_ratios=[4,4,3,3]
                  )
    ax1 = fig.add_subplot(gs[:,:2],projection='3d', facecolor=Axfacecolor)
    ax1.computed_zorder=False
    
    # This is Moon
    ax1 = Plot_xyz(Moon_path, 
                   ax1, 
                   i, 
                   'Moon',
                   Color='gray', 
                   Axis=True,
                   start=0
                   )
    
    
    # This is Orion
    ax1 = Plot_xyz(goodfile_path, 
                   ax1, 
                   i, 
                   'Orion Spacecraft',
                   start=max(0, i-100),
                   )
    ax1 = decorate(ax1)
    ax2 = fig.add_subplot(gs[0,2:])
    ax2 = Plot_r(goodfile_path, ax2, i)
    ax3 = fig.add_subplot(gs[1,2:])
    ax3 = Plot_v(goodfile_path, ax3, i)
    #plt.tight_layout()
    plt.savefig("./images/{:04d}.jpg".format(i), dpi=300, facecolor=Figfacecolor)
    plt.close()

def get_missing_frames():
    f = []
    for (dirpath, dirnames, filenames) in walk('./images/'):
        f.extend(filenames)
        break
    f.pop(-1)
    def s(fname):
        return int(fname.split('.')[0])
    f = list(map(s, f))
    f = sorted(list(set(list(np.arange(0,912))) - set(f)))
    #print(f)
    return f

def main():
    transform_raw(raw_path, goodfile_path)
    transform_raw(Moon_raw, Moon_path)
    #transform_raw("Earth.txt", "Earth.csv")
    #transform_raw("Mercury.txt", "Mercury.csv")
    D = pd.read_csv(goodfile_path)
    N = D['X'].size
    #fps= 30
    #t = 20
    n = N #fps*t
    
    # For single image:
    #Complex_1(100)
    
    # For single process:
    #for i in tqdm(np.linspace(0, N, N).astype(int)):
    #for i in tqdm(get_missing_frames()):
    #    Complex_1(i)
    
    # For multi-processing
    process_map(   Complex_1, 
                    np.arange(0,N-1).astype(int), 
                    max_workers=4)
    
    #Plot_rv(goodfile_path)
    #Plot_xy(goodfile_path)
    #Plot_phasespace(goodfile_path)
    #Plot_xyz(goodfile_path)
    
    #plt.show()

main()
