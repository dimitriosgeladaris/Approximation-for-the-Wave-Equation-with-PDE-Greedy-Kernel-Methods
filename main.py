"""
Template for solving your own wave problem with kernel collocation.

This is a starting point for a custom linear wave problem

    u_tt - c^2 * Laplace(u) = f      in the space-time interior,
                          u = g      on the spatial boundary,
                     u(0, .) = u0     (initial displacement),
                  u_t(0, .) = v0     (initial velocity).

Edit the settings block and the four data functions below, then run:

    python main.py

The thesis experiments live in ``examples/`` (one runnable file per
experiment). The main file is only a blank template for your own problems. It uses
the same model as the examples, so there is no duplicated solve/plot
code. If you have a closed-form solution you can return it from ``reference`` to
get an error report; otherwise set ``reference_func = None`` and only the
predicted solution is plotted.
"""

import os
import sys

import numpy as np

# Path to use needed functions.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))
import model

# ============================== settings =================================
method = "greedy"        # "greedy" or "full"
kernel_time = "Gaussian"   # time kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6"
kernel_space = "Gaussian"  # space kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6" 
ep_time = 4.0            # time-kernel shape parameter
ep_space = 2.5           # space-kernel shape parameter
m = 70                   # build collocation functional pool (i.e., N, see thesis)
nGreedy = 400            # number of maximum greedy expansion size
N_pred = 120             # prediction-grid resolution

# Optional: show animations of the selected collocation points (greedy) and of the solution over time.
make_animation = False

# Optional: select (ep_time, ep_space) by hold-out validation (greedy only).
optimize_parameters = False
ep_time_candidates = np.linspace(1.5, 10, 15)
ep_space_candidates = np.linspace(1.5, 10, 15)
# =========================================================================

c = 1.0 # wave speed
dim_space = 1 
domain = dict(t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0)
# For a 2D problem, set dim_space = 2 and add y_min / y_max to domain.

# 2d examples can be found in the examples folder, e.g., ``examples/wave_2d.py``.
def make_data():
    """Return (f_func, g_func, u0_func, v0_func) for your problem.

    Each takes an array X of shape (m, 1 + dim_space) whose columns are
    (t, x[, y]) and returns an (m, 1) column vector.
    """
    x_min, x_max = domain["x_min"], domain["x_max"]
    L = x_max - x_min

    def f_func(X):
        return np.zeros((X.shape[0], 1))

    def g_func(X):
        return np.zeros((X.shape[0], 1))

    def u0_func(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x = X[:, -1:]
        return np.sin(np.pi * (x - x_min) / L)

    def v0_func(X):
        return np.zeros((X.shape[0], 1))

    return f_func, g_func, u0_func, v0_func


# Optional: return a closed-form reference solution to get an error report,
# or set reference_func = None below to skip it.
def analytical_solution(t_pred, x_pred):
    x_min, x_max = domain["x_min"], domain["x_max"]
    L = x_max - x_min
    T, X = np.meshgrid(t_pred, x_pred, indexing="ij")
    return np.sin(np.pi * (X - x_min) / L) * np.cos(np.pi * c * T / L)


reference_func = analytical_solution   # set to None if you have no analytical solution


if __name__ == "__main__":
    if optimize_parameters:
        best_eps, _ = model.search_parameters(
            kernel_time=kernel_time, kernel_space=kernel_space, c=c, domain=domain, dim_space=dim_space, n=m,
            data=make_data(), ep_time_candidates=ep_time_candidates,
            ep_space_candidates=ep_space_candidates, n_greedy=nGreedy,
        )
        if best_eps is not None:
            ep_time, ep_space = best_eps
            print(f"Selected (ep_time, ep_space) = ({ep_time:.6g}, {ep_space:.6g})")

    wave_model = model.solve(
        method=method, kernel_time=kernel_time, kernel_space=kernel_space, ep_time=ep_time, ep_space=ep_space,
        c=c, domain=domain, dim_space=dim_space, n=m, data=make_data(),
        n_greedy=nGreedy, plot_selection=(method == "greedy"),
        make_animation=make_animation,
    )
    model.evaluate(wave_model, domain=domain, dim_space=dim_space,
                   reference=reference_func, pred_N=N_pred, c=c, make_animation=make_animation)
