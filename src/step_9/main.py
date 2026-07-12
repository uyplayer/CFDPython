"""Step 9 — 2D Convection (refactored)

Source notebook: lessons/12_Step_9.ipynb
This script is auto-generated from the notebook code cells.
IPython-only constructs (magics, YouTubeVideo, HTML) are removed
or commented out so the file runs as a plain Python module.
"""

from __future__ import annotations


# (notebook cell 11 skipped — IPython-only: 'import numpy')

# --- from notebook code cell 12 ---
def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = numpy.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

# --- from notebook code cell 15 ---
def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1
    pn = numpy.empty_like(p)

    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))
            
        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
        l1norm = (numpy.sum(numpy.abs(p[:]) - numpy.abs(pn[:])) /
                numpy.sum(numpy.abs(pn[:])))
     
    return p

# --- from notebook code cell 17 ---
##variable declarations
nx = 31
ny = 31
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)


##initial conditions
p = numpy.zeros((ny, nx))  # create a XxY vector of 0's


##plotting aids
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 1, ny)

##boundary conditions
p[:, 0] = 0  # p = 0 @ x = 0
p[:, -1] = y  # p = y @ x = 2
p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1

# --- from notebook code cell 19 ---
plot2D(x, y, p)

# --- from notebook code cell 21 ---
p = laplace2d(p, y, dx, dy, 1e-4)

# --- from notebook code cell 23 ---
plot2D(x, y, p)

# (notebook cell 27 skipped — IPython-only: 'from IPython.display import YouTubeVideo')

# (notebook cell 29 skipped — IPython-only: 'from IPython.display import YouTubeVideo')

# (notebook cell 30 skipped — IPython-only: 'from IPython.core.display import HTML')


def main() -> None:
    """Run the Step 9 — 2D Convection (refactored) example end-to-end."""
    # Re-execute the code above inside this scope.
    # The top-level statements already ran on import; this entry
    # point is here so you can `python -m step_<N>.main` or
    # `python step_<N>/main.py` and reach a single hook.
    pass  # no extra orchestration needed for this step


if __name__ == "__main__":
    main()
