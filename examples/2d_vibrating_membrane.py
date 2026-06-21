"""
Experiment: 2D vibrating membrane (standing wave / eigenmode).

    u_tt - c^2 (u_xx + u_yy) = 0,   (x,y) in [-1,1]^2,  t in [0,1],
    u = 0 on the boundary,  u_t(0,.) = 0,
    u(0,x,y) = sin(kx pi (x - x_min)/Lx) sin(ky pi (y - y_min)/Ly).

Reference solution: standing wave with
    omega = c pi sqrt(kx^2/Lx^2 + ky^2/Ly^2).

Run with:  python 2d_vibrating_membrane.py
This experiment was run with the greedy solver in the thesis.
"""

import numpy as np
import model

# ============================== settings =================================
method = "greedy"        # "greedy" 
kernel_time = "Gaussian"   # time kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6"
kernel_space = "Gaussian"  # space kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6" (set different to mix kernels)

# Optimized Kernel shape parameters (time / space). Values for k = 3, greedy:
#   Gaussian: (2.0, 2.0)
#   IMQ:      (1.5, 1.5)
ep_time = 2.0
ep_space = 2.0

m = 30                   # collocation points per class (per axis)
kx = 3                   # wave number in x
ky = 3                   # wave number in y
nGreedy = 2000           # number of greedy points
N_pred = 40              # prediction-grid resolution

# Optional: show animations of the selected collocation points (greedy)
# and of the solution over time.
make_animation = False

# Optional: select (ep_time, ep_space) by hold-out validation (greedy only).
# If True, the values above are overwritten by the search result.
optimize_parameters = False
ep_time_candidates = np.linspace(1.5, 10, 10)
ep_space_candidates = np.linspace(1.5, 10, 4)
# =========================================================================

c = 1.0
domain = dict(t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0,
              y_min=-1.0, y_max=1.0)


def make_data():
    x_min, x_max = domain["x_min"], domain["x_max"]
    y_min, y_max = domain["y_min"], domain["y_max"]
    Lx, Ly = x_max - x_min, y_max - y_min

    def f_func(X):
        return np.zeros((X.shape[0], 1))

    def g_func(X):
        return np.zeros((X.shape[0], 1))

    def v0_func(X):
        return np.zeros((X.shape[0], 1))

    def u0_func(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x = X[:, -2]
        y = X[:, -1]
        return (np.sin(kx * np.pi * (x - x_min) / Lx)
                * np.sin(ky * np.pi * (y - y_min) / Ly))

    return f_func, g_func, u0_func, v0_func


def analytical_solution(t_pred, x_pred, y_pred):
    x_min, x_max = domain["x_min"], domain["x_max"]
    y_min, y_max = domain["y_min"], domain["y_max"]
    Lx, Ly = x_max - x_min, y_max - y_min
    T, X, Y = np.meshgrid(t_pred, x_pred, y_pred, indexing="ij")
    omega = c * np.pi * np.sqrt(kx ** 2 / Lx ** 2 + ky ** 2 / Ly ** 2)
    return (np.sin(kx * np.pi * (X - x_min) / Lx)
            * np.sin(ky * np.pi * (Y - y_min) / Ly)
            * np.cos(omega * T))


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
