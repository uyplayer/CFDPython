"""Step 4 — 1D Diffusion Equation

Source notebook: lessons/05_Step_4.ipynb
This script is auto-generated from the notebook code cells.
IPython-only constructs (magics, YouTubeVideo, HTML) are removed
or commented out so the file runs as a plain Python module.
"""

from __future__ import annotations


# --- from notebook code cell 8 ---
import numpy
import sympy

# --- from notebook code cell 10 ---
from sympy import init_printing
init_printing(use_latex=True)

# --- from notebook code cell 12 ---
x, nu, t = sympy.symbols('x nu t')
phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
       sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
phi

# --- from notebook code cell 14 ---
phiprime = phi.diff(x)
phiprime

# --- from notebook code cell 16 ---
print(phiprime)

# --- from notebook code cell 18 ---
from sympy.utilities.lambdify import lambdify

u = -2 * nu * (phiprime / phi) + 4
print(u)

# --- from notebook code cell 20 ---
ufunc = lambdify((t, x, nu), u)
print(ufunc(1, 4, 3))

# (notebook cell 22 skipped — IPython-only: 'from matplotlib import pyplot')

# --- from notebook code cell 23 ---
pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.plot(x, u, marker='o', lw=2)
pyplot.xlim([0, 2 * numpy.pi])
pyplot.ylim([0, 10]);

# --- from notebook code cell 26 ---
for n in range(nt):
    un = u.copy()
    for i in range(1, nx-1):
        u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 *\
                (un[i+1] - 2 * un[i] + un[i-1])
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 *\
                (un[1] - 2 * un[0] + un[-2])
    u[-1] = u[0]
        
u_analytical = numpy.asarray([ufunc(nt * dt, xi, nu) for xi in x])

# --- from notebook code cell 27 ---
pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.plot(x,u, marker='o', lw=2, label='Computational')
pyplot.plot(x, u_analytical, label='Analytical')
pyplot.xlim([0, 2 * numpy.pi])
pyplot.ylim([0, 10])
pyplot.legend();

# (notebook cell 29 skipped — IPython-only: 'from IPython.core.display import HTML')


def main() -> None:
    """Run the Step 4 — 1D Diffusion Equation example end-to-end."""
    # Re-execute the code above inside this scope.
    # The top-level statements already ran on import; this entry
    # point is here so you can `python -m step_<N>.main` or
    # `python step_<N>/main.py` and reach a single hook.
    pass  # no extra orchestration needed for this step


if __name__ == "__main__":
    main()
