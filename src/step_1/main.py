"""Step 1 — 1D Linear Convection

Source notebook: lessons/01_Step_1.ipynb
This script is auto-generated from the notebook code cells.
IPython-only constructs (magics, YouTubeVideo, HTML) are removed
or commented out so the file runs as a plain Python module.
"""

from __future__ import annotations


# --- from notebook code cell 6 ---
# Remember: comments in python are denoted by the pound sign
import numpy                       #here we load numpy
from matplotlib import pyplot      #here we load matplotlib
import time, sys                   #and load some utilities

# (notebook cell 7 skipped — IPython-only: '#this makes matplotlib plots appear in the notebook (instead of a separate window)')

# --- from notebook code cell 9 ---
nx = 500  # try changing this number from 41 to 81 and Run All ... what happens?
dx = 10 / (nx-1)
nt = 100    #nt is the number of timesteps we want to calculate
dt = .025  #dt is the amount of time each timestep covers (delta t)
c = 1      #assume wavespeed of c = 1

# --- from notebook code cell 11 ---
u = numpy.ones(nx)      #numpy function ones()
u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s
print(u)

# --- from notebook code cell 13 ---
pyplot.plot(numpy.linspace(0, 2, nx), u);

# --- from notebook code cell 16 ---
un = numpy.ones(nx) #initialize a temporary array

for n in range(nt):  #loop for values of n from 0 to nt, so it will run nt times
    un = u.copy() ##copy the existing values of u into un
    for i in range(1, nx): ## you can try commenting this line and...
    #for i in range(nx): ## ... uncommenting this line and see what happens!
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

# --- from notebook code cell 18 ---
pyplot.plot(numpy.linspace(0, 2, nx), u);

# (notebook cell 22 skipped — IPython-only: 'from IPython.display import YouTubeVideo')

# (notebook cell 23 skipped — IPython-only: "YouTubeVideo('xq9YTcv-fQg')")

# (notebook cell 25 skipped — IPython-only: "YouTubeVideo('y2WaK7_iMRI')")

# (notebook cell 28 skipped — IPython-only: 'from IPython.core.display import HTML')


def main() -> None:
    """Run the Step 1 — 1D Linear Convection example end-to-end."""
    # Re-execute the code above inside this scope.
    # The top-level statements already ran on import; this entry
    # point is here so you can `python -m step_<N>.main` or
    # `python step_<N>/main.py` and reach a single hook.
    pass  # no extra orchestration needed for this step


if __name__ == "__main__":
    main()
