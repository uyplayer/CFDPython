"""Step 2 — 1D Nonlinear Convection

Source notebook: lessons/02_Step_2.ipynb
This script is auto-generated from the notebook code cells.
IPython-only constructs (magics, YouTubeVideo, HTML) are removed
or commented out so the file runs as a plain Python module.
"""

from __future__ import annotations


# (notebook cell 7 skipped — IPython-only: "import numpy                 # we're importing numpy ")

# --- from notebook code cell 9 ---
for n in range(nt):  #iterate through time
    un = u.copy() ##copy the existing values of u into un
    for i in range(1, nx):  ##now we'll iterate through the u array
    
     ###This is the line from Step 1, copied exactly.  Edit it for our new equation.
     ###then uncomment it and run the cell to evaluate Step 2   
      
           ###u[i] = un[i] - c * dt / dx * (un[i] - un[i-1]) 

        
pyplot.plot(numpy.linspace(0, 2, nx), u) ##Plot the results

# (notebook cell 13 skipped — IPython-only: 'from IPython.display import YouTubeVideo')

# (notebook cell 14 skipped — IPython-only: 'from IPython.core.display import HTML')


def main() -> None:
    """Run the Step 2 — 1D Nonlinear Convection example end-to-end."""
    # Re-execute the code above inside this scope.
    # The top-level statements already ran on import; this entry
    # point is here so you can `python -m step_<N>.main` or
    # `python step_<N>/main.py` and reach a single hook.
    pass  # no extra orchestration needed for this step


if __name__ == "__main__":
    main()
