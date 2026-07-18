'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-18 12:10:49
LastEditTime: 2026-07-18 13:06:21
LastEditors: uyplayer uyplayer@outlook.com
Description:  2D Convection
FilePath: /CFDPython/src/step_6/main.py
@copyright Copyright (c) 2026 by uyplayer
'''


import os
import numpy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot, colormaps


class TwoDConvection:
    
    
    def __init__(self,nx, ny, nt, c=1, L=2.0, sigma=0.2):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.c = c
        self.L = L 

        
        self.delta_x = self.L / (self.nx - 1)
        self.delta_y = self.L / (self.ny - 1)
        
        self.sigma = sigma
        self.delta_t = self.sigma * self.delta_x
        
        
        # x 和 y grid  绘制2D 网格 
        self.x = numpy.linspace(0, self.L, self.nx)
        self.y = numpy.linspace(0, self.L, self.ny)
        self.X, self.Y = numpy.meshgrid(self.x, self.y)
        
        
        
        self.v = None           
        self.v_initial = None   
        self.v_final = None     
        
        self.u = None           
        self.u_initial = None   
        self.u_final = None     

    
    
    def simulate(self):
        
        v = numpy.ones((self.ny, self.nx)) 
        u = numpy.ones((self.ny, self.nx)) 
        
        u[int(.5 / self.delta_y):int(1 / self.delta_y + 1), int(.5 / self.delta_x):int(1 / self.delta_x + 1)] = 2
        v[int(.5 / self.delta_y):int(1 / self.delta_y + 1), int(.5 / self.delta_x):int(1 / self.delta_x + 1)] = 2
        
        
        self.v_initial = v.copy()
        self.u_initial = u.copy()
        
        for _ in range(self.nt + 1):
            vn = v.copy()
            un = u.copy()
            u[1:, 1:] = (un[1:, 1:] - 
                 (un[1:, 1:] * self.c * self.delta_t / self.delta_x * (un[1:, 1:] - un[1:, :-1])) -
                  vn[1:, 1:] * self.c * self.delta_t / self.delta_y  * (un[1:, 1:] - un[:-1, 1:]))
            v[1:, 1:] = (vn[1:, 1:] -
                        (un[1:, 1:] * self.c * self.delta_t / self.delta_x * (vn[1:, 1:] - vn[1:, :-1])) -
                        vn[1:, 1:] * self.c * self.delta_t / self.delta_y * (vn[1:, 1:] - vn[:-1, 1:]))
            
            
            v[0, :] = 1
            v[-1, :] = 1
            v[:, 0] = 1
            v[:, -1] = 1
            
            u[0, :] = 1
            u[-1, :] = 1
            u[:, 0] = 1
            u[:, -1] = 1
            
        self.v = v    
        self.u = u
        
        self.v_final = v.copy()
        self.u_final = u.copy()
        return self.v,self.u    
            
            
    
    def plot(self):
        """Plot initial and final states of u and v as 3D surfaces; save to PNG."""
        if self.u_initial is None or self.u_final is None \
                or self.v_initial is None or self.v_final is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(15, 10), dpi=100)

        # u — initial
        ax1: Axes3D = fig.add_subplot(221, projection='3d')  
        ax1.plot_surface(self.X, self.Y, self.u_initial[:], cmap=colormaps['viridis'])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title('u — Initial (t=0)')

        # u — final
        ax2: Axes3D = fig.add_subplot(222, projection='3d')   
        ax2.plot_surface(self.X, self.Y, self.u_final[:], cmap=colormaps['viridis'])
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title(f'u — Final (nt={self.nt})')

        # v — initial
        ax3: Axes3D = fig.add_subplot(223, projection='3d')  
        ax3.plot_surface(self.X, self.Y, self.v_initial[:], cmap=colormaps['viridis'])
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('v')
        ax3.set_title('v — Initial (t=0)')

        # v — final
        ax4: Axes3D = fig.add_subplot(224, projection='3d')  
        ax4.plot_surface(self.X, self.Y, self.v_final[:], cmap=colormaps['viridis'])
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('v')
        ax4.set_title(f'v — Final (nt={self.nt})')

        fig.suptitle(
            f'2D Convection (nx={self.nx}, ny={self.ny}, nt={self.nt})',
            fontsize=14,
        )
        fig.tight_layout()

        filename = f'convection2d_nx{self.nx}_nt{self.nt}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'图已保存到：{filepath}')

        pyplot.show() 


if __name__ == '__main__':
     
    convection = TwoDConvection(nx=41, ny=41, nt=100)
    v, u = convection.simulate()
    convection.plot()    
    
    
    
    convection = TwoDConvection(nx=100, ny=100, nt=100)
    v, u = convection.simulate()
    convection.plot()    