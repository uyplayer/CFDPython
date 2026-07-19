'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-18 12:10:49
LastEditors: uyplayer uyplayer@outlook.com
LastEditTime: 2026-07-19 10:55:01
FilePath: /CFDPython/src/step_9/main.py
Description:  2D Laplace Equation  (∇²p = 0)
@copyright Copyright (c) 2026 by uyplayer
'''

import os
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class Laplace2D:
    """
    2D Laplace equation: ∂²p/∂x² + ∂²p/∂y² = 0

    核心思想：每个网格点的值 = 周围 4 个邻居的加权平均
        p[i, j] = (Δy²·(p[i+1,j] + p[i-1,j]) + Δx²·(p[i,j+1] + p[i,j-1])) / (2·(Δx² + Δy²))

    如果 Δx = Δy，简化为最朴素的"四邻居平均"：
        p[i, j] = (左 + 右 + 上 + 下) / 4
    """

    def __init__(self, nx=31, ny=31, Lx=2.0, Ly=1.0, l1norm_target=1e-4,
                 max_iter=10000):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx                   
        self.Ly = Ly                  
        self.l1norm_target = l1norm_target  
        self.max_iter = max_iter       

        self.delta_x = self.Lx / (self.nx - 1)
        self.delta_y = self.Ly / (self.ny - 1)
 
        self.x = numpy.linspace(0, self.Lx, self.nx)
        self.y = numpy.linspace(0, self.Ly, self.ny)
        self.X, self.Y = numpy.meshgrid(self.x, self.y)

        self.p = None
        self.p_initial = None
        self.p_final = None
        self.iterations = 0

    def _apply_boundary_conditions(self, p):
        """每次迭代后重新施加边界条件。"""
        
        p[:, 0] = 0
        p[:, -1] = self.y

        
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        return p

    def simulate(self):
        """
        迭代求解 Laplace 方程，直到 L1 范数变化小于阈值。

        关键步骤：
          1) 初始化 p = 0（内部初值不重要）
          2) 反复做"四邻居加权平均"
          3) 每轮重新设定边界
          4) 检查两次迭代之间的相对变化量
        """
     
        p = numpy.zeros((self.ny, self.nx))
        p[:, -1] = self.y   
        self.p_initial = p.copy()

        l1norm = 1.0
        iteration = 0

        while l1norm > self.l1norm_target and iteration < self.max_iter:
            pn = p.copy()

        
            p[1:-1, 1:-1] = (
                (self.delta_y ** 2
                 * (pn[1:-1, 2:] + pn[1:-1, 0:-2])
                 + self.delta_x ** 2
                 * (pn[2:, 1:-1] + pn[0:-2, 1:-1]))
                / (2.0 * (self.delta_x ** 2 + self.delta_y ** 2))
            )

       
            p = self._apply_boundary_conditions(p)

          
            l1norm = (numpy.sum(numpy.abs(p) - numpy.abs(pn))
                      / numpy.sum(numpy.abs(pn)))

            iteration += 1

        self.iterations = iteration
        self.p = p
        self.p_final = p.copy()
        print(f'收敛：迭代 {iteration} 次后 L1 norm = {l1norm:.6e}')
        return self.p

    def plot(self):
        """Plot initial and final states of p as 3D surfaces; save to PNG."""
        if self.p_initial is None or self.p_final is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(15, 6), dpi=100)

   
        ax1: Axes3D = fig.add_subplot(121, projection='3d')  
        ax1.plot_surface(self.X, self.Y, self.p_initial[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('p')
        ax1.set_title('Initial (only BCs set)')

   
        ax2: Axes3D = fig.add_subplot(122, projection='3d')  
        ax2.plot_surface(self.X, self.Y, self.p_final[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('p')
        ax2.set_title(f'Final (converged in {self.iterations} iters)')

        fig.suptitle(
            f'2D Laplace (nx={self.nx}, ny={self.ny}, '
            f'Lx={self.Lx}, Ly={self.Ly})',
            fontsize=14,
        )
        fig.tight_layout()

        filename = f'laplace2d_nx{self.nx}_ny{self.ny}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'图已保存到：{filepath}')

        pyplot.show()


if __name__ == '__main__':
   
    sim = Laplace2D(nx=31, ny=31, Lx=2.0, Ly=1.0, l1norm_target=1e-4)
    sim.simulate()
    sim.plot()

 
    sim = Laplace2D(nx=31, ny=31, Lx=2.0, Ly=1.0, l1norm_target=1e-6)
    sim.simulate()
    sim.plot()

 
    sim = Laplace2D(nx=61, ny=61, Lx=2.0, Ly=1.0, l1norm_target=1e-4)
    sim.simulate()
    sim.plot()