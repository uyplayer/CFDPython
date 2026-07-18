'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-18 12:10:49
LastEditTime: 2026-07-18 15:19:30
LastEditors: uyplayer uyplayer@outlook.com
Description:  2D Diffusion
FilePath: /CFDPython/src/step_7/main.py
@copyright Copyright (c) 2026 by uyplayer
'''

import os
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class TwoDDiffusion:

    def __init__(self, nx, ny, nt, nu=0.05, L=2.0, sigma=0.25):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.nu = nu        
        self.L = L

        self.delta_x = self.L / (self.nx - 1)
        self.delta_y = self.L / (self.ny - 1)

       
        self.sigma = sigma
        self.delta_t = self.sigma * self.delta_x * self.delta_y / self.nu

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
        # 设置初始条件：中间 0.5<=x,y<=1 区域设为 2（hat function I.C.）
        u[int(.5 / self.delta_y):int(1 / self.delta_y + 1),
          int(.5 / self.delta_x):int(1 / self.delta_x + 1)] = 2

        # 初始值绘制用
        self.u_initial = u.copy()

        for _ in range(self.nt + 1):
            un = u.copy()

            u[1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                + self.nu * self.delta_t / self.delta_x ** 2
                * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + self.nu * self.delta_t / self.delta_y ** 2
                * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
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
        """Plot initial and final states of u as 3D surfaces; save to PNG."""
        if self.u_initial is None or self.u_final is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(15, 6), dpi=100)

        # u — initial
        ax1: Axes3D = fig.add_subplot(121, projection='3d')  # type: ignore[assignment]
        ax1.plot_surface(self.X, self.Y, self.u_initial[:], cmap=pyplot.get_cmap('viridis'))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title('Initial (t=0)')

        # u — final
        ax2: Axes3D = fig.add_subplot(122, projection='3d')  # type: ignore[assignment]
        ax2.plot_surface(self.X, self.Y, self.u_final[:], cmap=pyplot.get_cmap('viridis'))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title(f'Final (nt={self.nt})')

        fig.suptitle(
            f'2D Diffusion (nx={self.nx}, ny={self.ny}, nt={self.nt}, nu={self.nu})',
            fontsize=14,
        )
        fig.tight_layout()

        filename = f'diffusion2d_nx{self.nx}_nt{self.nt}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'图已保存到：{filepath}')

        pyplot.show()


if __name__ == '__main__':

    sim = TwoDDiffusion(nx=31, ny=31, nt=10, nu=0.05)
    sim.simulate()
    sim.plot()

    sim = TwoDDiffusion(nx=31, ny=31, nt=14, nu=0.05)
    sim.simulate()
    sim.plot()

    sim = TwoDDiffusion(nx=31, ny=31, nt=50, nu=0.05)
    sim.simulate()
    sim.plot()