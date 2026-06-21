"""
Experiment: 2D travelling sine wave.

    u_tt - c^2 (u_xx + u_yy) = 0,   (x,y) in [-1,1]^2,  t in [0,1],
    boundary, initial displacement and initial velocity taken from
        u(t,x,y) = sin(kx pi (x - x_min)/Lx + ky pi (y - y_min)/Ly - omega t),
    with omega = c sqrt((kx pi/Lx)^2 + (ky pi/Ly)^2).

Reference solution: the travelling wave itself.

Run with:  python 2d_travelling_sine.py
This experiment was run with the greedy solver in the thesis.
"""

import numpy as np
import model

# ============================== settings =================================
method = "greedy"        # "greedy" (full collocation was not used here)
kernel_time = "Gaussian"   # time kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6"
kernel_space = "Gaussian"  # space kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6" (set different to mix kernels)

# Optimized Kernel shape parameters (time / space). 
#   IMQ:      (1.5, 1.3)
ep_time = 1.9444444444444444
ep_space = 2.1666666666666665

m = 30                   # collocation points per class (per axis)
kx = 3                   # wave number in x
ky = 3                   # wave number in y
nGreedy = 2000           # number of greedy points
N_pred = 50              # prediction-grid resolution

# Optional: show animations of the selected collocation points (greedy)
# and of the solution over time.
make_animation = False

# Optional: select (ep_time, ep_space) by hold-out validation (greedy only).
# If True, the values above are overwritten by the search result.
optimize_parameters = False
ep_time_candidates = np.linspace(1.5, 2.5, 10)
ep_space_candidates = np.linspace(1.5, 2.5, 10)
# =========================================================================

c = 1.0 # wave speed
domain = dict(t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0,
              y_min=-1.0, y_max=1.0)


def _omega():
    Lx = domain["x_max"] - domain["x_min"]
    Ly = domain["y_max"] - domain["y_min"]
    return c * np.sqrt((kx * np.pi / Lx) ** 2 + (ky * np.pi / Ly) ** 2)


def make_data():
    x_min, x_max = domain["x_min"], domain["x_max"]
    y_min, y_max = domain["y_min"], domain["y_max"]
    Lx, Ly = x_max - x_min, y_max - y_min
    omega = _omega()

    def f_func(X):
        return np.zeros((X.shape[0], 1))

    def g_func(X):
        t = X[:, 0:1]
        x = X[:, 1:2]
        y = X[:, 2:3]
        return np.sin(kx * np.pi * (x - x_min) / Lx
                      + ky * np.pi * (y - y_min) / Ly - omega * t)

    def u0_func(X):
        x = X[:, 1:2]
        y = X[:, 2:3]
        return np.sin(kx * np.pi * (x - x_min) / Lx
                      + ky * np.pi * (y - y_min) / Ly)

    def v0_func(X):
        x = X[:, 1:2]
        y = X[:, 2:3]
        return -omega * np.cos(kx * np.pi * (x - x_min) / Lx
                               + ky * np.pi * (y - y_min) / Ly)

    return f_func, g_func, u0_func, v0_func


def analytical_solution(t_pred, x_pred, y_pred):
    x_min, x_max = domain["x_min"], domain["x_max"]
    y_min, y_max = domain["y_min"], domain["y_max"]
    Lx, Ly = x_max - x_min, y_max - y_min
    omega = _omega()
    T, X, Y = np.meshgrid(t_pred, x_pred, y_pred, indexing="ij")
    return np.sin(kx * np.pi * (X - x_min) / Lx
                  + ky * np.pi * (Y - y_min) / Ly - omega * T)


if __name__ == "__main__":
    if optimize_parameters:
        best_eps, _ = model.search_parameters(
            kernel_time=kernel_time, kernel_space=kernel_space, c=c, domain=domain, dim_space=2, n=m, data=make_data(),
            ep_time_candidates=ep_time_candidates,
            ep_space_candidates=ep_space_candidates, n_greedy=nGreedy,
        )
        if best_eps is not None:
            ep_time, ep_space = best_eps
            print(f"Selected (ep_time, ep_space) = ({ep_time:.6g}, {ep_space:.6g})")

    wave_model = model.solve(
        method=method, kernel_time=kernel_time, kernel_space=kernel_space, ep_time=ep_time, ep_space=ep_space,
        c=c, domain=domain, dim_space=2, n=m, data=make_data(),
        n_greedy=nGreedy, make_animation=make_animation,
    )
    model.evaluate(wave_model, domain=domain, dim_space=2,
                   reference=analytical_solution, pred_N=N_pred, c=c, make_animation=make_animation)
