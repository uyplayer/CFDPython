'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-18 12:10:49
LastEditors: uyplayer uyplayer@outlook.com
LastEditTime: 2026-07-19 11:30:00
FilePath: /CFDPython/src/step_11/main.py
Description:  Lid-Driven Cavity Flow — Full 2D Navier-Stokes
@copyright Copyright (c) 2026 by uyplayer
'''

import os
import numpy
from matplotlib import pyplot


class NavierStokes2DCavity:
    """
    完整 Navier-Stokes 2D 求解器：腔体驱动流（Lid-driven cavity flow）。

    三个方程：
      1. u 动量方程 (x方向)
      2. v 动量方程 (y方向)
      3. 压力 Poisson 方程（保证 ∇·v = 0）

    时间推进：每个 nt 步里都做：
      a) 用当前 u, v 算压力 Poisson 的源项 b
      b) 迭代 nit 次解压力 Poisson 方程
      c) 用新压力梯度修正 u, v
      d) 施加速度边界条件
    """

    def __init__(self, nx=41, ny=41, nt=500, nit=50,
                 rho=1.0, nu=0.1, dt=0.001, L=2.0):
        self.nx = nx
        self.ny = ny
        self.nt = nt                  # 物理时间步数
        self.nit = nit                # 压力 Poisson 子迭代次数（伪时间）

        self.rho = rho
        self.nu = nu
        self.dt = dt                  # 真实物理时间步长
        self.L = L

        self.delta_x = self.L / (self.nx - 1)
        self.delta_y = self.L / (self.ny - 1)

        # x / y grid
        self.x = numpy.linspace(0, self.L, self.nx)
        self.y = numpy.linspace(0, self.L, self.ny)
        self.X, self.Y = numpy.meshgrid(self.x, self.y)

        # 状态变量
        self.u = None
        self.v = None
        self.p = None
        self.b = None

    def _build_up_b(self, b):
        """
        构造压力 Poisson 方程的源项 b：
          b = ρ * [ (1/dt)·∇·u - (∂u/∂x)² - 2·(∂u/∂y)(∂v/∂x) - (∂v/∂y)² ]

        所有空间导数都用中心差分。
        """
        u = self.u
        v = self.v
        dx = self.delta_x
        dy = self.delta_y
        dt = self.dt
        rho = self.rho

        b[1:-1, 1:-1] = (
            rho * (
                1.0 / dt *
                ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx)
                 + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy))
                - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx)) ** 2
                - 2.0 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2.0 * dy)
                       * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * dx))
                - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy)) ** 2
            )
        )
        return b

    def _pressure_poisson(self, p, b):
        """
        迭代解压力 Poisson 方程：
          ∇²p = b
        整理得：
          p[i,j] = [(Δy²·(p右+p左) + Δx²·(p上+p下)) - dx²·dy²·b] / [2·(dx²+dy²)]

        注意：这里调用的是 step_10 的 Poisson 思路，但 b 由动量方程给出。
        """
        dx = self.delta_x
        dy = self.delta_y
        pn = numpy.empty_like(p)

        for _ in range(self.nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2
                 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2)
                / (2.0 * (dx ** 2 + dy ** 2))
                - dx ** 2 * dy ** 2 / (2.0 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
            )

            # 压力边界条件
            p[:, -1] = p[:, -2]    # ∂p/∂x = 0 @ x = L
            p[0, :] = p[1, :]      # ∂p/∂y = 0 @ y = 0
            p[:, 0] = p[:, 1]      # ∂p/∂x = 0 @ x = 0
            p[-1, :] = 0           # p = 0 @ y = L（参考压力）
        return p

    def simulate(self):
        """
        时间推进求解腔体流。每个 nt 步：
          1) 算压力 Poisson 源项 b
          2) 解压力 Poisson 方程
          3) 用新压力梯度更新 u, v
          4) 施加速度边界条件
        """
        # 初始化：u = v = p = 0
        u = numpy.zeros((self.ny, self.nx))
        v = numpy.zeros((self.ny, self.nx))
        p = numpy.zeros((self.ny, self.nx))
        b = numpy.zeros((self.ny, self.nx))

        self.u = u
        self.v = v
        self.p = p
        self.b = b

        for n in range(self.nt):
            un = u.copy()
            vn = v.copy()

            # 1) 构造压力源项
            b = self._build_up_b(b)

            # 2) 解压力 Poisson（内部迭代 nit 次）
            p = self._pressure_poisson(p, b)

            # 3) 用压力梯度更新 u（动量方程）
            u[1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                - un[1:-1, 1:-1] * self.dt / self.delta_x
                  * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
                - vn[1:-1, 1:-1] * self.dt / self.delta_y
                  * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
                - self.dt / (2.0 * self.rho * self.delta_x)
                  * (p[1:-1, 2:] - p[1:-1, 0:-2])
                + self.nu * (
                    self.dt / self.delta_x ** 2
                    * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                    + self.dt / self.delta_y ** 2
                    * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
                )
            )

            # 4) 用压力梯度更新 v（动量方程）
            v[1:-1, 1:-1] = (
                vn[1:-1, 1:-1]
                - un[1:-1, 1:-1] * self.dt / self.delta_x
                  * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
                - vn[1:-1, 1:-1] * self.dt / self.delta_y
                  * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
                - self.dt / (2.0 * self.rho * self.delta_y)
                  * (p[2:, 1:-1] - p[0:-2, 1:-1])
                + self.nu * (
                    self.dt / self.delta_x ** 2
                    * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                    + self.dt / self.delta_y ** 2
                    * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
                )
            )

            # 5) 速度边界条件
            u[0, :]  = 0
            u[:, 0]  = 0
            u[:, -1] = 0
            u[-1, :] = 1.0          # ⭐ 关键！顶盖（lid）以速度 1 推动流体
            v[0, :]  = 0
            v[-1, :] = 0
            v[:, 0]  = 0
            v[:, -1] = 0

        self.u = u
        self.v = v
        self.p = p
        return self.u, self.v, self.p

    def plot(self, mode='quiver', nt_show=None):
        """
        可视化流场：
          - 等高线填充（contourf）+ 速度箭头（quiver）或 流线（streamplot）

        参数：
          mode: 'quiver' 或 'streamplot'
          nt_show: 用于标题显示的步数（默认 self.nt）
        """
        if self.u is None or self.p is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(11, 7), dpi=100)

        # 压力场：填充等高线 + 等高线
        pyplot.contourf(self.X, self.Y, self.p, alpha=0.5,
                        cmap=pyplot.get_cmap('viridis'))
        pyplot.colorbar()
        pyplot.contour(self.X, self.Y, self.p,
                       cmap=pyplot.get_cmap('viridis'))

        # 速度场：quiver（箭头）或 streamplot（流线）
        if mode == 'streamplot':
            pyplot.streamplot(self.X, self.Y, self.u, self.v,
                              color='black', density=1.2)
            title_extra = 'streamplot'
        else:
            pyplot.quiver(self.X[::2, ::2], self.Y[::2, ::2],
                          self.u[::2, ::2], self.v[::2, ::2],
                          color='black')
            title_extra = 'quiver'

        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title(f'Cavity Flow (nt={nt_show or self.nt}, '
                     f'ν={self.nu}, ρ={self.rho}, {title_extra})')

        filename = (f'cavity_nt{nt_show or self.nt}_'
                    f'nu{self.nu}_{title_extra}.png')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'图已保存到：{filepath}')
        pyplot.show()


if __name__ == '__main__':
    # 1) 与 notebook 默认 nt=100 对应：腔体流开始形成（局部旋涡）
    sim = NavierStokes2DCavity(nx=41, ny=41, nt=100, nit=50,
                               rho=1.0, nu=0.1, dt=0.001)
    sim.simulate()
    sim.plot(mode='quiver')

    # 2) nt=700：稳态螺旋（最终教科书图）
    sim = NavierStokes2DCavity(nx=41, ny=41, nt=700, nit=50,
                               rho=1.0, nu=0.1, dt=0.001)
    sim.simulate()
    sim.plot(mode='streamplot')

    # 3) 高粘度 ν=0.5：粘性主导，旋涡弱、流动平缓
    sim = NavierStokes2DCavity(nx=41, ny=41, nt=300, nit=50,
                               rho=1.0, nu=0.5, dt=0.001)
    sim.simulate()
    sim.plot(mode='streamplot')