'''
Author: uyplayer uyplayer@outlook.com
Date: 2026-07-13 08:56:43
LastEditors: uyplayer uyplayer@outlook.com
LastEditTime: 2026-07-17 09:35:34
FilePath: /CFDPython/src/step_5/main.py
Description: 生成遇到问题，请再试试
'''
import numpy
import os
import sympy
from typing import Optional
from sympy.utilities.lambdify import lambdify
from matplotlib import pyplot


class BurgersEquation:

    def __init__(self, nx, nt, nu=0.07, L=2*numpy.pi, dt=None):
        self.nx = nx
        self.nt = nt
        self.nu = nu
        self.L = L                          
        self.dx = L / (nx - 1)              
        
        self.x:         Optional[numpy.ndarray] = None
        self.u_initial: Optional[numpy.ndarray] = None
        self.u_final:   Optional[numpy.ndarray] = None
        
        u_max = 5.0                         
        if dt is None:
            self.dt = min(self.dx / u_max, 0.5 * self.dx**2 / self.nu)
        else:
            self.dt = dt

 
    def build_initial_condition(self):
 
        x  = sympy.Symbol('x')
        nu = sympy.Symbol('nu')
        t  = sympy.Symbol('t')
     
        phi = (sympy.exp(-(x - 4*t)**2 / (4*nu*(t + 1))) +
               sympy.exp(-(x - 4*t - 2*sympy.pi)**2 / (4*nu*(t + 1))))
        phiprime = phi.diff(x)
 
        u_expr = -2*nu*(phiprime/phi) + 4
        self.ufunc = lambdify((t, x, nu), u_expr)
 
    def simulate(self):
        self.build_initial_condition()
        # 网络空间
        self.x = numpy.linspace(0, self.L, self.nx)
        t = 0.0
        # 当前时刻的解
        self.u = numpy.asarray([self.ufunc(t, x0, self.nu) for x0 in self.x])
 
        self.u_initial = self.u.copy()

        for n in range(self.nt):
            print(f"当前时间步数：{n}")
            un = self.u.copy()
            print(f't={t:.2f}')
            for i in range(1, self.nx - 1):
                print(f"当前网格点：{i}")
                self.u[i] = (un[i]
                             - un[i] * self.dt / self.dx * (un[i] - un[i-1])
                             + self.nu * self.dt / self.dx**2
                               * (un[i+1] - 2*un[i] + un[i-1]))
            # 周期性边界
            self.u[0] = (un[0]
                         - un[0] * self.dt / self.dx * (un[0] - un[-2])
                         + self.nu * self.dt / self.dx**2
                           * (un[1] - 2*un[0] + un[-2]))
            self.u[-1] = self.u[0]

        # 保存最终状态(供 plot() 使用)
        self.u_final = self.u.copy()
 
    def plot(self):    
        if self.u_final is None or self.u_initial is None or self.x is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return

     
        fig = pyplot.figure(figsize=(8, 5))
        pyplot.plot(self.x, self.u_initial, label='Initial (t=0)', linestyle='--')
        pyplot.plot(self.x, self.u_final, label=f'Final (nt={self.nt})', linewidth=2)
        pyplot.xlabel('X')
        pyplot.ylabel('Velocity u')
        pyplot.title(f"1D Burgers' Equation Simulation (nu={self.nu})")
        pyplot.legend()

        filename = f'convection_nx{self.nx}_nt{self.nt}.png'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        fig.savefig(filepath, dpi=120, bbox_inches='tight')
        pyplot.close(fig)
        print(f'图已保存到：{filepath}')


if __name__ == '__main__':
    burgers = BurgersEquation(nx=101, nt=1000)
    burgers.simulate()
    burgers.plot()