import numpy                   
from matplotlib import pyplot      
import time, sys 
import scipy as sp
import matplotlib.animation as animation 
from sympy import N 


class OneDLinearConvection:
    def __init__(self, nx, nt, c):
        self.nx = nx  # grid 数量
        self.nt = nt  # 时间步数
        self.c = c   # 波速度
        self.L = 2.0
        self.delta_t = 0.02  # 修复了 dt = 0.02 的多余赋值
        self.u_final = None
        self.u_initial = None # 预先定义初始状态
        self.x = None         # 预先定义网格空间

    def simulate(self):
        delta_x = self.L / self.nx
 
        self.x = numpy.linspace(0, self.L, self.nx)
        
        u = numpy.ones(self.nx)
        # 人工初始化
        u[int(.5 / delta_x):int(1 / delta_x + 1)] = 2 
        
    
        self.u_initial = u.copy()
        
        for n in range(0, self.nt):
            print(f"当前时间步数：{n}")
            u_copy = u.copy()
            for i in range(1, self.nx):
                print(f"当前网格点：{i}")
                u[i] = u_copy[i] - self.c * self.delta_t / delta_x * (u_copy[i] - u_copy[i-1])
        
        self.u_final = u    
        
    def plot(self):
       
        if self.u_final is None or self.u_initial is None or self.x is None:
            print("请先调用 simulate() 方法进行仿真，再进行绘图！")
            return
            
 
        pyplot.figure(figsize=(8, 5))
        pyplot.plot(self.x, self.u_initial, label='Initial (t=0)', linestyle='--')
        pyplot.plot(self.x, self.u_final, label=f'Final (nt={self.nt})', linewidth=2)
        pyplot.xlabel('X')
        pyplot.ylabel('Velocity u')
        pyplot.title('1D Linear Convection Simulation')
        pyplot.legend()
        pyplot.show()
        
        
if __name__ == '__main__':
 
    sim = OneDLinearConvection(nx=100, nt=20, c=1.0)
    
 
    sim.simulate()
     
    sim.plot()