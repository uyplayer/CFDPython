"""Step 7 — 1D Poisson Equation

Source notebook: lessons/09_Step_7.ipynb
This script is auto-generated from the notebook code cells.
IPython-only constructs (magics, YouTubeVideo, HTML) are removed
or commented out so the file runs as a plain Python module.
"""

from __future__ import annotations


# (notebook cell 8 skipped — IPython-only: 'import numpy')

# --- from notebook code cell 9 ---
###variable declarations
nx = 31
ny = 31
nt = 17
nu = .05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

u = numpy.ones((ny, nx))  # create a 1xn vector of 1's
un = numpy.ones((ny, nx))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2

# --- from notebook code cell 10 ---
fig = pyplot.figure()
ax = fig.gca(projection='3d')
X, Y = numpy.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(1, 2.5)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');

# --- from notebook code cell 12 ---
###Run through nt timesteps
def diffuse(nt):
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2  
    
    for n in range(nt + 1): 
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1,1:-1] + 
                        nu * dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy**2 * 
                        (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=True)
    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');

# --- from notebook code cell 13 ---
diffuse(10)

# --- from notebook code cell 14 ---
diffuse(14)

# --- from notebook code cell 15 ---
diffuse(50)

# (notebook cell 18 skipped — IPython-only: 'from IPython.display import YouTubeVideo')

# (notebook cell 19 skipped — IPython-only: 'from IPython.core.display import HTML')


def main() -> None:
    """Run the Step 7 — 1D Poisson Equation example end-to-end."""
    # Re-execute the code above inside this scope.
    # The top-level statements already ran on import; this entry
    # point is here so you can `python -m step_<N>.main` or
    # `python step_<N>/main.py` and reach a single hook.
    pass  # no extra orchestration needed for this step


if __name__ == "__main__":
    main()
