'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-18 12:10:49
LastEditors: uyplayer uyplayer@outlook.com
LastEditTime: 2026-07-19 11:10:20
FilePath: /CFDPython/src/step_10/main.py
Description:  2D Poisson Equation  (∇²p = b)
@copyright Copyright (c) 2026 by uyplayer
'''

import os
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class Poisson2D:
    """
    2D Poisson equation: ∂²p/∂x² + ∂²p/∂y² = b

    与 Laplace 方程（∇²p = 0）的唯一区别：右边多了源项 b。
    离散公式：
        p[i,j] = (Δy²·(p[i+1,j] + p[i-1,j])
                  + Δx²·(p[i,j+1] + p[i,j-1])
                  - b[i,j]·Δx²·Δy²)
                 / (2·(Δx² + Δy²))

    b 的符号决定了该点 p 的"凹凸"方向：
      b > 0 → -b·dx²dy² < 0 → p 出现下凹（负峰）
      b < 0 → -b·dx²dy² > 0 → p 出现上凸（正峰）
    """

    def __init__(self, nx=50, ny=50, n_iter=100,
                 xmin=0.0, xmax=2.0, ymin=0.0, ymax=1.0):
        self.nx = nx
        self.ny = ny
        # 注意：这里不是物理时间步！而是 "伪时间" 松弛迭代次数。
        # Poisson 方程本身没有时间项；每轮迭代相当于做一次 Jacobi 扫描。
        self.n_iter = n_iter

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.delta_x = (self.xmax - self.xmin) / (self.nx - 1)
        self.delta_y = (self.ymax - self.ymin) / (self.ny - 1)

        # x / y grid
        self.x = numpy.linspace(self.xmin, self.xmax, self.nx)
        self.y = numpy.linspace(self.ymin, self.ymax, self.ny)
        self.X, self.Y = numpy.meshgrid(self.x, self.y)

        self.p = None
        self.b = None
        self.p_initial = None
        self.p_final = None
        self.iterations = 0

    def _set_source(self, b=None):
        """
        设置源项 b。默认使用 notebook 配置：左下 +100、右上 -100 两个点源。
        """
        if b is not None:
            self.b = b.copy()
            return self.b

        b = numpy.zeros((self.ny, self.nx))
        b[int(self.ny / 4), int(self.nx / 4)] = 100              # 左下：正源
        b[int(3 * self.ny / 4), int(3 * self.nx / 4)] = -100    # 右上：负源
        self.b = b
        return self.b

    def _apply_boundary_conditions(self, p):
        """四面 Dirichlet 边界：p = 0。"""
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0
        return p

    def simulate(self, b=None):
        """
        迭代求解 Poisson 方程。

        关键步骤：
          1) 初始化 p = 0；设置源项 b
          2) 反复做"四邻居加权平均 - 源项修正"
          3) 每轮重新设定边界
        """
        # 源项
        if b is None:
            self._set_source()
        else:
            self.b = b.copy()

        # 初值：内部全是 0
        p = numpy.zeros((self.ny, self.nx))
        self.p_initial = p.copy()

        for it in range(self.n_iter):
            pd = p.copy()

            # 核心：五点模板 + 源项修正
            p[1:-1, 1:-1] = (
                (self.delta_y ** 2 * (pd[1:-1, 2:] + pd[1:-1, 0:-2])
                 + self.delta_x ** 2 * (pd[2:, 1:-1] + pd[0:-2, 1:-1])
                 - self.b[1:-1, 1:-1] * self.delta_x ** 2 * self.delta_y ** 2)
                / (2.0 * (self.delta_x ** 2 + self.delta_y ** 2))
            )

            # 边界条件（每轮都要重新设！）
            p = self._apply_boundary_conditions(p)

        self.iterations = self.n_iter
        self.p = p
        self.p_final = p.copy()
        return self.p

    def plot(self):
        """Plot initial and final states of p as 3D surfaces; save to PNG."""
        if self.p_initial is None or self.p_final is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(15, 6), dpi=100)

        # p — initial（全 0，只看到边界）
        ax1: Axes3D = fig.add_subplot(121, projection='3d')  
        ax1.plot_surface(self.X, self.Y, self.p_initial[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('p')
        ax1.set_title('Initial (p=0, sources set)')

        # p — final（迭代后的"鼓包+凹陷"形态）
        ax2: Axes3D = fig.add_subplot(122, projection='3d')  
        ax2.plot_surface(self.X, self.Y, self.p_final[:],
                         cmap=pyplot.get_cmap('viridis'))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('p')
        ax2.set_title(f'Final (after {self.iterations} pseudo-timesteps)')

        fig.suptitle(
            f'2D Poisson (nx={self.nx}, ny={self.ny}, n_iter={self.n_iter})',
            fontsize=14,
        )
        fig.tight_layout()

        filename = f'poisson2d_nx{self.nx}_ny{self.ny}_iter{self.n_iter}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'图已保存到：{filepath}')

        pyplot.show()


if __name__ == '__main__':

    sim = Poisson2D(nx=50, ny=50, n_iter=100, xmin=0, xmax=2, ymin=0, ymax=1)
    sim.simulate()
    sim.plot()


    sim = Poisson2D(nx=50, ny=50, n_iter=500, xmin=0, xmax=2, ymin=0, ymax=1)
    sim.simulate()
    sim.plot()


    b_custom = numpy.zeros((50, 50))
    b_custom[25, 25] = 200
    sim = Poisson2D(nx=50, ny=50, n_iter=200, xmin=0, xmax=2, ymin=0, ymax=1)
    sim.simulate(b=b_custom)
    sim.plot()