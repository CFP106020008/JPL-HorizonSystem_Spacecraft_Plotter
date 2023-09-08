import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from tqdm import tqdm


class One_Scene:
    
    def __init__(self, source_list, 
                 Figfacecolor='#333333',
                 Axfacecolor='#1c1c1c',
                 color_r='yellow',
                 color_v='skyblue',
                 rvimage_path='rv.png'):
        '''
        source_list should be a list that contains the filename of 
        the txt files that are downloaded from JPL horizons system.
        '''
        self.source_list  = source_list
        self.Figfacecolor = Figfacecolor
        self.Axfacecolor  = Axfacecolor
        self.color_r      = color_r
        self.color_v      = color_v
        self.rvimage_path = rvimage_path
        return 
    
    def transform_raw(self, txt_path, csv_path):
        '''
        Here we transform the original data from 
        NASA JPL Horizon System to a format 
        that is easy to read for pandas
        '''
        text = open(txt_path, encoding="utf-8").readlines()
        LS = text.index("$$SOE\n")
        LE = text.index("$$EOE\n")

        Labels = text[LS-2].split(sep=',')
        for i in range(len(Labels)):
            Labels[i] = Labels[i].strip()
        Labels = ",".join(Labels)
        
        text2 = list(Labels+'\n') + list(text[LS+1:LE])
        
        f2 = open(csv_path,'w')
        for line in text2:
            f2.write(line)
        f2.close()
        return 

    def Plot_rv(self, path):
        # Now to plot the data we want
        
        D = pd.read_csv(path)
        fig, ax1 = plt.subplots(facecolor=self.Figfacecolor) # For r
        ax2 = ax1.twinx() # For v
        
        # Calculate v and r from raw data
        D['r'] = np.sqrt(D.X**2 + D.Y**2 + D.Z**2)
        D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
        
        # Add a "Mission time" column
        D['days'] = D['JDTDB'] - D['JDTDB'][0]
        
        # Plot r part
        ax1.plot(D.days, D.r, color=self.color_r)
        ax1.set_facecolor(self.Axfacecolor)
        ax1.set_xlabel("Mission time (days)")
        ax1.set_ylabel("Distance to the Earth (km)")
        ax1.set_xlim([0, np.array(D['days'])[-1]])
        
        # Plot v part
        ax2.plot(D.days, D.v, color=self.color_v)
        ax2.set_ylabel("Velocity relative to the Earth (km/s)")
        
        plt.tight_layout()
        plt.savefig(self.rvimage_path, dpi=300, facecolor='#333333')
        plt.show()

    def Plot_phasespace(self, path):
        D = pd.read_csv(path)
        fig, ax = plt.subplots(facecolor=self.Figfacecolor)
        
        D['r'] = np.sqrt(D.X**2 + D.Y**2 + D.Z**2)
        D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
        
        ax.plot(D['r'], D['v'], color='yellow')
        #plt.show()
        
    def Plot_xy(self, path):
        D = pd.read_csv(path)
        fig, ax = plt.subplots(facecolor=self.Figfacecolor)
        
        ax.plot(D['X'], D['Y'], color='yellow')
        ax.set_aspect('equal', adjustable='box')
        #plt.show()    
        
    def Plot_xyz(self, goodfile_path, ax, i, Color='yellow', Axis=False, start=0):
        D = pd.read_csv(goodfile_path)
        #fig = plt.figure(facecolor=Figfacecolor, figsize=(6,6))
        #ax = fig.add_subplot(projection='3d', facecolor=Axfacecolor)
        
        rmax = np.max((D['X']**2 + D['Y'] + D['Z']**2)**0.5)
        
        arrow_scale = 1.5
        if Axis:
            ax.quiver(-arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, 0, 0, color='w', arrow_length_ratio=0.05) # x-axis
            ax.quiver(0, -arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, 0, color='w', arrow_length_ratio=0.05) # y-axis
            ax.quiver(0, 0, -arrow_scale*rmax, 0, 0, 2*arrow_scale*rmax, color='w', arrow_length_ratio=0.05) # z-axis
        
        ax.plot(D['X'][start:i], D['Y'][start:i], D['Z'][start:i], color=Color)
        ax.scatter(D['X'][i], D['Y'][i], D['Z'][i], color=Color, s=10)
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.scatter(0,0,0,color='skyblue',s=100)
        
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        ax.set_xlim([-rmax, rmax])
        ax.set_ylim([-rmax, rmax])
        ax.set_zlim([-rmax, rmax])
        #plt.tight_layout()
        #plt.show()
        return ax

    def Plot_r(self, path, ax, i=0, Dot=False):
        
        D = pd.read_csv(path)
        
        # Calculate v and r from raw data
        D['r'] = np.sqrt(D.X**2 + D.Y**2 + D.Z**2)
        D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
        
        # Add a "Mission time" column
        D['days'] = D['JDTDB'] - D['JDTDB'][0]
        
        # Plot r part
        ax.plot(D.days[:i], D.r[:i], color=self.color_r)
        ax.set_facecolor(self.Axfacecolor)
        #ax.set_xlabel("Mission time (days)")
        ax.set_ylabel("Distance to the Earth (km)")
        ax.set_xlim([0, np.array(D['days'])[-1]])
        ax.set_xticks([])
        ax.set_ylim([0,np.max(D['r'])*1.2])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(4, 1))
        
        ax.scatter(D.days[i], D.r[i], s=10, color=self.color_v)
        
        #plt.tight_layout()
        #plt.savefig(finalimage_path, dpi=300, facecolor='#333333')
        #plt.show()
        
        return ax

    def Plot_v(self, path, ax, i=0, Dot=False):
        
        D = pd.read_csv(path)
        D['v'] = np.sqrt(D.VX**2 + D.VY**2 + D.VZ**2)
        
        # Add a "Mission time" column
        D['days'] = D['JDTDB'] - D['JDTDB'][0]
        
        ax.set_xlim([0, np.array(D['days'])[-1]])
        
        # Plot v part
        ax.plot(D.days[:i], D.v[:i], color=self.color_v)
        ax.set_facecolor(self.Axfacecolor)
        ax.set_xlabel("Mission time (days)")
        ax.set_ylabel("Velocity relative to the Earth (km/s)")
        ax.set_ylim([0,np.max(D['v']*1.2)])
        
        ax.scatter(D.days[i], D.v[i], s=10, color=self.color_v)
        
        #plt.tight_layout()
        #plt.savefig(finalimage_path, dpi=300, facecolor='#333333')
        #plt.show()
        
        return ax

    def Complex_1(self, i, num, Dot=True):
        fig = plt.figure(figsize=(12,27/4), facecolor=self.Figfacecolor)
        gs = GridSpec(2, 4, figure=fig,
                    left = 0.0625/1.5,
                    right = 1-0.0625/1.5,
                    top = 1-1/9/1.5,
                    bottom = 1/9/1.5,
                    wspace = 4/15,
                    hspace = 0.,
                    width_ratios=[4,4,3,3]
                    )
        ax1 = fig.add_subplot(gs[:,:2],projection='3d', facecolor=self.Axfacecolor)
        
        for source in self.source_list[1:]:
            ax1 = self.Plot_xyz(source, 
                        ax1, 
                        i, 
                        Color='gray', 
                        start=0)
        ax1 = self.Plot_xyz(self.source_list[0], 
                    ax1, 
                    i, 
                    Axis=True,
                    start=max(0, i-500))
        ax2 = fig.add_subplot(gs[0,2:])
        ax2 = self.Plot_r(self.source_list[0], ax2, i)
        ax3 = fig.add_subplot(gs[1,2:])
        ax3 = self.Plot_v(self.source_list[0], ax3, i)
        #plt.tight_layout()
        plt.savefig("./images/{:04d}.jpg".format(num), dpi=300, facecolor=self.Figfacecolor)
        plt.close()

    def do_all_convertions(self):
        for i, source in enumerate(self.source_list):
            self.transform_raw(source, source.split('.')[0]+'.csv')
            self.source_list[i] = source.split('.')[0]+'.csv'
    
    def make_animation(self):
        D = pd.read_csv(self.source_list[0])
        N = D['X'].size
        fps= 20
        t = 30
        n = fps*t
        #Complex_1(1800)
        for num, i in enumerate(tqdm(np.linspace(0, N-1, n).astype(int))):
            self.Complex_1(i, num)
        #Plot_rv(goodfile_path)
        #Plot_xy(goodfile_path)
        #Plot_phasespace(goodfile_path)
        #Plot_xyz(goodfile_path)
        
        #plt.show()
        return
    def make_final(self):
        D = pd.read_csv(self.source_list[0])
        self.Complex_1(D['X'].size - 1, D['X'].size - 1)