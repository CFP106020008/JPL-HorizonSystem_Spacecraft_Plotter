'''
This code is created by Lin Yen-Hsing (NTHU, 2017-2022).
It is used to process and plot the data retrived from the Horizon System of NASA/JPL.

The project is created on Sat Jan 8 21:10:11 2022
Inspired by https://en.wikipedia.org/wiki/File:Velocity_of_Parker_Solar_Probe_wide.svg

To use the code, you need to download the vector table data in csv format from NASA/JPL
and put them in the same folder as main.py. Next, you have to list path to the vector 
table file you want to visualize and pass them in the One_Scene class. 
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from classes import One_Scene

def main():
    plt.style.use('dark_background')
    CH3 = One_Scene(['CH3.txt', 'Moon.txt'])
    CH3.do_all_convertions()
    CH3.make_animation()

main()