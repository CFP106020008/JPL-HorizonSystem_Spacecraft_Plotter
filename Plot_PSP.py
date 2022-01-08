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
import pandas as pd

# To make the plot COOL
plt.style.use('dark_background')
raw_path = "Gaia.txt"
goodfile_path = "Gaia.csv"
finalimage_path = "Gaia.png"
Figfacecolor = '#333333'
Axfacecolor = '#1c1c1c'
color_r = 'yellow'
color_v = 'skyblue'

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

def Plot_xy(goodfile_path):
    D = pd.read_csv(goodfile_path)
    fig, ax = plt.subplots(facecolor=Figfacecolor)
    
    ax.plot(D['X'], D['Y'], color='yellow')
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    
def Plot_xyz(goodfile_path):
    D = pd.read_csv(goodfile_path)
    fig = plt.figure(facecolor=Figfacecolor)
    ax = fig.gca(projection='3d')
    
    ax.plot(D['X'], D['Y'], D['Z'], color='yellow')
    #ax.set_aspect('equal', adjustable='box')
    plt.show()

def main():
    transform_raw(raw_path)
    #Plot_rv(goodfile_path)
    #Plot_xy(goodfile_path)
    Plot_xyz(goodfile_path)

main()