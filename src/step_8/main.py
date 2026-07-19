'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-18 12:10:49
LastEditTime: 2026-07-19 10:18:14
LastEditors: uyplayer uyplayer@outlook.com
Description:  2D Burgers' Equation (convection + diffusion)
FilePath: /CFDPython/src/step_8/main.py
@copyright Copyright (c) 2026 by uyplayer
'''

import os
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class BurgersEquation2D:

    def __init__(self, nx=41, ny=41, nt=120, nu=0.01, L=2.0, sigma=0.0009):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.nu = nu          
        self.L = L

        self.delta_x = self.L / (self.nx - 1)
        self.delta_y = self.L / (self.ny - 1)

        
        self.sigma = sigma
        self.delta_t = self.sigma * self.delta_x * self.delta_y / self.nu

        # x 和 y grid
        self.x = numpy.linspace(0, self.L, self.nx)
        self.y = numpy.linspace(0, self.L, self.ny)
        self.X, self.Y = numpy.meshgrid(self.x, self.y)

        self.u = None
        self.v = None
        self.u_initial = None
        self.v_initial = None
        self.u_final = None
        self.v_final = None

    def simulate(self):
        """Solve 2D Burgers' equation with backward-difference convection
        + central-difference diffusion."""

      
        u = numpy.ones((self.ny, self.nx))
        v = numpy.ones((self.ny, self.nx))
        u[int(.5 / self.delta_y):int(1 / self.delta_y + 1),
          int(.5 / self.delta_x):int(1 / self.delta_x + 1)] = 2
        v[int(.5 / self.delta_y):int(1 / self.delta_y + 1),
          int(.5 / self.delta_x):int(1 / self.delta_x + 1)] = 2

        self.u_initial = u.copy()
        self.v_initial = v.copy()

        for _ in range(self.nt + 1):
            un = u.copy()
            vn = v.copy()

            # u 更新：对流（向后差分）+ 扩散（中心二阶）
            u[1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                - self.delta_t / self.delta_x * un[1:-1, 1:-1]
                  * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
                - self.delta_t / self.delta_y * vn[1:-1, 1:-1]
                  * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
                + self.nu * self.delta_t / self.delta_x ** 2
                  * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + self.nu * self.delta_t / self.delta_y ** 2
                  * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )

            # v 更新：完全对称
            v[1:-1, 1:-1] = (
                vn[1:-1, 1:-1]
                - self.delta_t / self.delta_x * un[1:-1, 1:-1]
                  * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
                - self.delta_t / self.delta_y * vn[1:-1, 1:-1]
                  * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
                + self.nu * self.delta_t / self.delta_x ** 2
                  * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                + self.nu * self.delta_t / self.delta_y ** 2
                  * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
            )

            # 边界条件
            u[0, :] = 1
            u[-1, :] = 1
            u[:, 0] = 1
            u[:, -1] = 1
            v[0, :] = 1
            v[-1, :] = 1
            v[:, 0] = 1
            v[:, -1] = 1

        self.u = u
        self.v = v
        self.u_final = u.copy()
        self.v_final = v.copy()
        return self.u, self.v

    def plot(self):
        """Plot initial and final states of u and v as 3D surfaces; save to PNG."""
        if self.u_initial is None or self.u_final is None \
                or self.v_initial is None or self.v_final is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(15, 10), dpi=100)

        # u — initial
        ax1: Axes3D = fig.add_subplot(221, projection='3d')  
        ax1.plot_surface(self.X, self.Y, self.u_initial[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title('u — Initial (t=0)')

        # u — final
        ax2: Axes3D = fig.add_subplot(222, projection='3d')  
        ax2.plot_surface(self.X, self.Y, self.u_final[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title(f'u — Final (nt={self.nt})')

        # v — initial
        ax3: Axes3D = fig.add_subplot(223, projection='3d')  
        ax3.plot_surface(self.X, self.Y, self.v_initial[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('v')
        ax3.set_title('v — Initial (t=0)')

        # v — final
        ax4: Axes3D = fig.add_subplot(224, projection='3d')  
        ax4.plot_surface(self.X, self.Y, self.v_final[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('v')
        ax4.set_title(f'v — Final (nt={self.nt})')

        fig.suptitle(
            f'2D Burgers (nx={self.nx}, ny={self.ny}, nt={self.nt}, '
            f'ν={self.nu}, σ={self.sigma})',
            fontsize=14,
        )
        fig.tight_layout()

        filename = f'burgers2d_nx{self.nx}_nt{self.nt}_nu{self.nu}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'图已保存到：{filepath}')

        pyplot.show()


if __name__ == '__main__':
   
    sim = BurgersEquation2D(nx=41, ny=41, nt=120, nu=0.01, sigma=0.0009)
    sim.simulate()
    print(f'u: min={sim.u.min():.4f}, max={sim.u.max():.4f}')
    print(f'v: min={sim.v.min():.4f}, max={sim.v.max():.4f}')
    sim.plot()
 
    sim = BurgersEquation2D(nx=41, ny=41, nt=120, nu=0.1, sigma=0.0009)
    sim.simulate()
    sim.plot()

 
    sim = BurgersEquation2D(nx=41, ny=41, nt=120, nu=0.001, sigma=0.0009)
    sim.simulate()
    sim.plot()