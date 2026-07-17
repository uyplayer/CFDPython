'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-13 08:56:43
LastEditTime: 2026-07-15 13:26:25
LastEditors: uyplayer uyplayer@outlook.com
Description: some description
FilePath: /CFDPython/src/step_4/main.py
@copyright Copyright (c) 2026 by uyplayer
'''
 
import os
import numpy
from matplotlib import pyplot
import time, sys
import scipy as sp
import matplotlib.animation as animation
from sympy import N






class  DiffusionEquationOneD:
    def __init__(self, nx ,nt ,nu = 0.3 ):
        
        
        self.nx = nx
        self.nt = nt
        self.nu = nu
        
        self.L = 2
        
        self.delta_x  = self.L / (self.nx - 1)
        
        self.sigma = 0.2 
        self.delta_t = self.sigma * self.delta_x**2 / nu
        
        self.u_final = None
        self.u_initial = None  
        self.x = None     
        
    def simulate(self):
        self.x = numpy.linspace(0, self.L, self.nx)  
        u = numpy.ones(self.nx)
        u[int(.5 / self.delta_x) : int(1 / self.delta_x + 1)] = 2
        
        self.u_initial = u.copy() 
        
        
        for n_time in range(self.nt): 
            print(f"当前时间步数：{n_time}")
            u_copy = u.copy()
            
            for n_grid in range(1, self.nx - 1):
                print(f"当前网格点：{n_grid}")
                u[n_grid] = u_copy[n_grid] + self.nu * self.delta_t / self.delta_x**2 * (u_copy[n_grid+1] - 2 * u_copy[n_grid] + u_copy[n_grid-1])
        
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
        
if __name__ == '__main__':
    diffusion_equation = DiffusionEquationOneD(nx=41, nt=20)
    diffusion_equation.simulate()
    diffusion_equation.plot()        
    
    
    
    diffusion_equation = DiffusionEquationOneD(nx=60, nt=20)
    diffusion_equation.simulate()
    diffusion_equation.plot()        
    
    
    diffusion_equation = DiffusionEquationOneD(nx=80, nt=20)

    diffusion_equation.simulate()
    diffusion_equation.plot()        