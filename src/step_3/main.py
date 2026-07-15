'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-13 08:56:43
LastEditTime: 2026-07-15 08:46:23
LastEditors: uyplayer uyplayer@outlook.com
Description: 收敛性与库朗条件（CFL条件）
FilePath: /CFDPython/src/step_3/main.py
@copyright Copyright (c) 2026 by uyplayer
'''




import os
import numpy
from matplotlib import pyplot
import time, sys
import scipy as sp
import matplotlib.animation as animation
from sympy import N


class LinearConv:
    def __init__(self, nx ):
        self.nx = nx 

        
        self.nt = 20 
        self.u_final = None
        self.u_initial = None  
        self.x = None 
    
    
    def simulate(self):
        L = 2.0
        delta_x = L / (self.nx - 1)
 
        delta_t = .025   
        c = 1
        
        self.x = numpy.linspace(0, L, self.nx)  
        u = numpy.ones(self.nx)
        u[int(.5 / delta_x) : int(1 / delta_x + 1)] = 2
        self.u_initial = u.copy() 

        for n in range(self.nt): 
            print(f"当前时间步数：{n}")
            u_copy = u.copy()
            for i in range(1, self.nx):  
                print(f"当前网格点：{i}")
                u[i] = u_copy[i] - u_copy[i] * delta_t / delta_x * (u_copy[i] - u_copy[i-1])
                
        self.u_final = u            
    
    
    def plot(self):
        if self.u_final is None or self.u_initial is None or self.x is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

     
        fig = pyplot.figure(figsize=(8, 5))
        pyplot.plot(self.x, self.u_initial, label='Initial (t=0)', linestyle='--')
        pyplot.plot(self.x, self.u_final, label=f'Final (nt={self.nt})', linewidth=2)
        pyplot.xlabel('X')
        pyplot.ylabel('Velocity u')
        pyplot.title('1D Non-Linear Convection Simulation')
        pyplot.legend()

        filename = f'convection_nx{self.nx}_nt{self.nt}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        pyplot.close(fig)
        print(f'图已保存到：{filepath}')
        
        
if __name__ == "__main__":
    nx = 41
    conv = LinearConv(nx)
    conv.simulate()
    conv.plot()
    
    
    
    nx =60
    conv = LinearConv(nx)
    conv.simulate()
    conv.plot()
    
    
    
    nx =80
    conv = LinearConv(nx)
    conv.simulate()
    conv.plot()
    
            