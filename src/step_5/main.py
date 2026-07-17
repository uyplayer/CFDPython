
'''
Author: uyplayer uyplayer@outlook.com
Date: 2026-07-13 08:56:43
LastEditors: uyplayer uyplayer@outlook.com
LastEditTime: 2026-07-17 15:47:55
FilePath: /CFDPython/src/step_5/main.py
Description: 2D Linear Convection
'''
import os
import numpy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot, colormaps


class TwoDLinearConvection:

    def __init__(self, nx, ny, nt, c=1, L=2.0, sigma=0.2):
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

        self.u = None           
        self.u_initial = None   
        self.u_final = None     

    def simulate(self):
        
        # 初始状态
        u = numpy.ones((self.ny, self.nx))
        # 设置边界条件
        u[int(.5 / self.delta_y):int(1 / self.delta_y + 1),
          int(.5 / self.delta_x):int(1 / self.delta_x + 1)] = 2

        # 初始值绘制用
        self.u_initial = u.copy()

        for _ in range(self.nt + 1):
            un = u.copy()
   
            u[1:, 1:] = (
                un[1:, 1:]
                - (self.c * self.delta_t / self.delta_x
                   * (un[1:, 1:] - un[1:, :-1]))
                - (self.c * self.delta_t / self.delta_y
                   * (un[1:, 1:] - un[:-1, 1:]))
            )
            # 边界条件
            u[0, :] = 1
            u[-1, :] = 1
            u[:, 0] = 1
            u[:, -1] = 1

        self.u = u
        self.u_final = u.copy()
        return self.u

    def plot(self):
        """Plot initial and final states side-by-side; save to PNG."""
        if self.u_initial is None or self.u_final is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(15, 6), dpi=100)
 
        ax1 = fig.add_subplot(121, projection='3d')  
        ax1.plot_surface(self.X, self.Y, self.u_initial[:], cmap=colormaps['viridis'])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title('Initial (t=0)')

        ax2 = fig.add_subplot(122, projection='3d')  
        ax2.plot_surface(self.X, self.Y, self.u_final[:], cmap=colormaps['viridis'])
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title(f'Final (nt={self.nt})')

        fig.suptitle(
            f'2D Linear Convection (nx={self.nx}, ny={self.ny}, nt={self.nt})',
            fontsize=14,
        )
        fig.tight_layout()

        filename = f'convection_nx{self.nx}_nt{self.nt}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        pyplot.close(fig)
        print(f'图已保存到：{filepath}')


if __name__ == '__main__':
    sim = TwoDLinearConvection(nx=81, ny=81, nt=100)
    sim.simulate()
    sim.plot()

    
    
    sim = TwoDLinearConvection(nx=81, ny=81, nt=100,sigma=0.5)
    sim.simulate()
    sim.plot()


    sim = TwoDLinearConvection(nx=160, ny=160, nt=100,sigma=0.95)
    sim.simulate()
    sim.plot()
    
    
    
    sim = TwoDLinearConvection(nx=160, ny=160, nt=200,sigma=0.95)
    sim.simulate()
    sim.plot()