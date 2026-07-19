'''
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2026-07-18 12:10:49
LastEditors: uyplayer uyplayer@outlook.com
LastEditTime: 2026-07-19 11:45:00
FilePath: /CFDPython/src/step_12/main.py
Description:  Pressure-driven Channel Flow — 2D Navier-Stokes (periodic in x)
@copyright Copyright (c) 2026 by uyplayer
'''

import os
import numpy
from matplotlib import pyplot


class NavierStokes2DChannel:
    """
    压力驱动的通道流（Poiseuille-like flow）。

    与 Step 11 的差异：
      1. u 动量方程多了一项 F·Δt（常数压力梯度驱动）
      2. x 方向使用周期性边界（无限长通道）
      3. y 方向上下都是壁面（u=v=0）
      4. 用 while udiff > tolerance 自适应停止（不是固定 nt 步）

    物理结果：抛物线速度剖面（中间快，两边慢）。
    """

    def __init__(self, nx=41, ny=41, rho=1.0, nu=0.1, F=1.0, dt=0.01,
                 Lx=2.0, Ly=2.0, udiff_target=1e-3, max_iter=10000):
        self.nx = nx
        self.ny = ny

        self.rho = rho
        self.nu = nu
        self.F = F                      # ⭐ 压力梯度（常数源项）
        self.dt = dt

        self.Lx = Lx
        self.Ly = Ly
        self.delta_x = self.Lx / (self.nx - 1)
        self.delta_y = self.Ly / (self.ny - 1)

        self.udiff_target = udiff_target
        self.max_iter = max_iter

        # grid
        self.x = numpy.linspace(0, self.Lx, self.nx)
        self.y = numpy.linspace(0, self.Ly, self.ny)
        self.X, self.Y = numpy.meshgrid(self.x, self.y)

        # 状态变量
        self.u = None
        self.v = None
        self.p = None
        self.b = None
        self.nit = 50                   # Poisson 子迭代次数（每步内部）
        self.steps = 0

    # ---------- 1) 压力源项 b（带周期性边界） ----------
    def _build_up_b(self, b, u, v):
        """构造压力 Poisson 方程的源项 b（含周期性边界）。"""
        dx, dy, dt, rho = self.delta_x, self.delta_y, self.dt, self.rho
        b[1:-1, 1:-1] = (
            rho * (
                1.0 / dt * (
                    (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx)
                    + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy)
                )
                - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx)) ** 2
                - 2.0 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2.0 * dy)
                       * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * dx))
                - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy)) ** 2
            )
        )
        # Periodic BC for b @ x = Lx ：用 x=0 的"邻居"替代 x=Lx+1
        b[1:-1, -1] = (
            rho * (
                1.0 / dt * (
                    (u[1:-1, 0] - u[1:-1, -2]) / (2.0 * dx)
                    + (v[2:, -1] - v[0:-2, -1]) / (2.0 * dy)
                )
                - ((u[1:-1, 0] - u[1:-1, -2]) / (2.0 * dx)) ** 2
                - 2.0 * ((u[2:, -1] - u[0:-2, -1]) / (2.0 * dy)
                       * (v[1:-1, 0] - v[1:-1, -2]) / (2.0 * dx))
                - ((v[2:, -1] - v[0:-2, -1]) / (2.0 * dy)) ** 2
            )
        )
        # Periodic BC for b @ x = 0 ：用 x=Lx-1（即 -1）替代 x=-1
        b[1:-1, 0] = (
            rho * (
                1.0 / dt * (
                    (u[1:-1, 1] - u[1:-1, -1]) / (2.0 * dx)
                    + (v[2:, 0] - v[0:-2, 0]) / (2.0 * dy)
                )
                - ((u[1:-1, 1] - u[1:-1, -1]) / (2.0 * dx)) ** 2
                - 2.0 * ((u[2:, 0] - u[0:-2, 0]) / (2.0 * dy)
                       * (v[1:-1, 1] - v[1:-1, -1]) / (2.0 * dx))
                - ((v[2:, 0] - v[0:-2, 0]) / (2.0 * dy)) ** 2
            )
        )
        return b

    # ---------- 2) 压力 Poisson 求解（带周期性边界） ----------
    def _pressure_poisson_periodic(self, p, b):
        dx, dy = self.delta_x, self.delta_y
        pn = numpy.empty_like(p)
        for _ in range(self.nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2
                 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2)
                / (2.0 * (dx ** 2 + dy ** 2))
                - dx ** 2 * dy ** 2 / (2.0 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
            )
            # 周期性 x=2 ：用 x=0 的邻居
            p[1:-1, -1] = (
                ((pn[1:-1, 0] + pn[1:-1, -2]) * dy ** 2
                 + (pn[2:, -1] + pn[0:-2, -1]) * dx ** 2)
                / (2.0 * (dx ** 2 + dy ** 2))
                - dx ** 2 * dy ** 2 / (2.0 * (dx ** 2 + dy ** 2)) * b[1:-1, -1]
            )
            # 周期性 x=0 ：用 x=Lx-1（-1）替代 -1
            p[1:-1, 0] = (
                ((pn[1:-1, 1] + pn[1:-1, -1]) * dy ** 2
                 + (pn[2:, 0] + pn[0:-2, 0]) * dx ** 2)
                / (2.0 * (dx ** 2 + dy ** 2))
                - dx ** 2 * dy ** 2 / (2.0 * (dx ** 2 + dy ** 2)) * b[1:-1, 0]
            )
            # 上下壁面：∂p/∂y=0
            p[-1, :] = p[-2, :]
            p[0, :] = p[1, :]
        return p

    # ---------- 3) 内部 u 更新 ----------
    def _update_u_internal(self, u, v, p, un):
        return (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * self.dt / self.delta_x
              * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - v[1:-1, 1:-1] * self.dt / self.delta_y
              * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - self.dt / (2.0 * self.rho * self.delta_x)
              * (p[1:-1, 2:] - p[1:-1, 0:-2])
            + self.nu * (
                self.dt / self.delta_x ** 2
                * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + self.dt / self.delta_y ** 2
                * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
            + self.F * self.dt          # ⭐ 唯一的关键差别！驱动项
        )

    # ---------- 4) 内部 v 更新 ----------
    def _update_v_internal(self, u, v, p, vn):
        return (
            vn[1:-1, 1:-1]
            - u[1:-1, 1:-1] * self.dt / self.delta_x
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

    # ---------- 主循环：while 残差 ----------
    def simulate(self):
        """
        时间推进直到 u 的相对变化量 < udiff_target。

        每步：
          1) 算压力源项 b（含周期性）
          2) 解压力 Poisson（含周期性）
          3) 更新 u（含 F 源项）和 v
          4) 周期性边界 u/v @ x=0, Lx
          5) 壁面边界 u/v=0 @ y=0, Ly
          6) 计算 udiff；判定是否收敛
        """
        u = numpy.zeros((self.ny, self.nx))
        v = numpy.zeros((self.ny, self.nx))
        p = numpy.ones((self.ny, self.nx))   # notebook 用 p=ones 初值
        b = numpy.zeros((self.ny, self.nx))

        udiff = 1.0
        self.steps = 0
        un = numpy.empty_like(u)
        vn = numpy.empty_like(v)

        while udiff > self.udiff_target and self.steps < self.max_iter:
            un = u.copy()
            vn = v.copy()

            # 1) 算 b
            b = self._build_up_b(b, u, v)

            # 2) 压力 Poisson（含周期性）
            p = self._pressure_poisson_periodic(p, b)

            # 3) 内部 u, v 更新
            u[1:-1, 1:-1] = self._update_u_internal(u, v, p, un)
            v[1:-1, 1:-1] = self._update_v_internal(u, v, p, vn)

            # 4) 周期性边界：x=Lx（用 x=0 邻居）
            u[1:-1, -1] = (
                un[1:-1, -1]
                - un[1:-1, -1] * self.dt / self.delta_x
                  * (un[1:-1, -1] - un[1:-1, -2])
                - v[1:-1, -1] * self.dt / self.delta_y
                  * (un[1:-1, -1] - un[0:-2, -1])
                - self.dt / (2.0 * self.rho * self.delta_x)
                  * (p[1:-1, 0] - p[1:-1, -2])
                + self.nu * (
                    self.dt / self.delta_x ** 2
                    * (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1, -2])
                    + self.dt / self.delta_y ** 2
                    * (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])
                )
                + self.F * self.dt
            )
            v[1:-1, -1] = (
                vn[1:-1, -1]
                - u[1:-1, -1] * self.dt / self.delta_x
                  * (vn[1:-1, -1] - vn[1:-1, -2])
                - vn[1:-1, -1] * self.dt / self.delta_y
                  * (vn[1:-1, -1] - vn[0:-2, -1])
                - self.dt / (2.0 * self.rho * self.delta_y)
                  * (p[2:, -1] - p[0:-2, -1])
                + self.nu * (
                    self.dt / self.delta_x ** 2
                    * (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2])
                    + self.dt / self.delta_y ** 2
                    * (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])
                )
            )

            # 4) 周期性边界：x=0（用 x=Lx-1 替代 x=-1）
            u[1:-1, 0] = (
                un[1:-1, 0]
                - un[1:-1, 0] * self.dt / self.delta_x
                  * (un[1:-1, 0] - un[1:-1, -1])
                - v[1:-1, 0] * self.dt / self.delta_y
                  * (un[1:-1, 0] - un[0:-2, 0])
                - self.dt / (2.0 * self.rho * self.delta_x)
                  * (p[1:-1, 1] - p[1:-1, -1])
                + self.nu * (
                    self.dt / self.delta_x ** 2
                    * (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1])
                    + self.dt / self.delta_y ** 2
                    * (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])
                )
                + self.F * self.dt
            )
            v[1:-1, 0] = (
                vn[1:-1, 0]
                - u[1:-1, 0] * self.dt / self.delta_x
                  * (vn[1:-1, 0] - vn[1:-1, -1])
                - vn[1:-1, 0] * self.dt / self.delta_y
                  * (vn[1:-1, 0] - vn[0:-2, 0])
                - self.dt / (2.0 * self.rho * self.delta_y)
                  * (p[2:, 0] - p[0:-2, 0])
                + self.nu * (
                    self.dt / self.delta_x ** 2
                    * (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1])
                    + self.dt / self.delta_y ** 2
                    * (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])
                )
            )

            # 5) 上下壁面：u=v=0
            u[0, :] = 0
            u[-1, :] = 0
            v[0, :] = 0
            v[-1, :] = 0

            # 6) 收敛判据：u 的总变化率
            udiff = (numpy.sum(u) - numpy.sum(un)) / numpy.sum(u)
            self.steps += 1

        self.u = u
        self.v = v
        self.p = p
        print(f'收敛：步数 = {self.steps}, udiff = {udiff:.6e}')
        return self.u, self.v, self.p

    # ---------- 可视化：quiver 显示抛物线剖面 ----------
    def plot(self, subsample=3, mode='quiver'):
        """绘制 u, v 流场（quiver 箭头）。"""
        if self.u is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

        fig = pyplot.figure(figsize=(11, 7), dpi=100)
        if mode == 'streamplot':
            pyplot.streamplot(self.X, self.Y, self.u, self.v,
                              color='black', density=1.2)
            title_extra = 'streamplot'
        else:
            pyplot.quiver(self.X[::subsample, ::subsample],
                          self.Y[::subsample, ::subsample],
                          self.u[::subsample, ::subsample],
                          self.v[::subsample, ::subsample],
                          color='black')
            title_extra = f'quiver (every {subsample})'

        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title(f'Channel Flow (steps={self.steps}, '
                     f'ν={self.nu}, F={self.F}, {title_extra})')

        filename = f'channel_steps{self.steps}_nu{self.nu}_F{self.F}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'图已保存到：{filepath}')
        pyplot.show()

    # ---------- 额外：可绘制速度剖面（验证 Poiseuille） ----------
    def plot_u_profile(self):
        """绘制 y 方向上 u 的中线剖面，应该是抛物线。"""
        if self.u is None:
            print("请先调用 simulate() 方法进行仿真！")
            return

        # 取 x 中线
        u_mid = self.u[:, self.nx // 2]
        fig = pyplot.figure(figsize=(8, 5), dpi=100)
        pyplot.plot(u_mid, self.y, 'b-', linewidth=2)
        pyplot.xlabel('u')
        pyplot.ylabel('y')
        pyplot.title(f'u-profile at x=L/2 (steps={self.steps}, '
                     f'F={self.F}, ν={self.nu})')
        pyplot.grid(True)

        filename = f'channel_uprof_steps{self.steps}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f'剖面图已保存到：{filepath}')
        pyplot.show()


if __name__ == '__main__':
    # 1) 与 notebook 默认参数一致（F=1, nu=0.1, dt=0.01）
    sim = NavierStokes2DChannel(nx=41, ny=41, rho=1.0, nu=0.1, F=1.0,
                                dt=0.01)
    sim.simulate()
    sim.plot(subsample=3, mode='quiver')
    sim.plot_u_profile()

    # 2) 大压力梯度（F=5 → 更快更明显的剖面）
    sim = NavierStokes2DChannel(nx=41, ny=41, rho=1.0, nu=0.1, F=5.0,
                                dt=0.01, udiff_target=1e-3)
    sim.simulate()
    sim.plot_u_profile()

    # 3) 低粘度 → 更"陡"的剖面（中央更快）
    sim = NavierStokes2DChannel(nx=41, ny=41, rho=1.0, nu=0.01, F=1.0,
                                dt=0.005, udiff_target=1e-3)
    sim.simulate()
    sim.plot(subsample=3, mode='streamplot')