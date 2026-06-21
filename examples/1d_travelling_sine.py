"""
Experiment: 1D travelling sine wave.

    u_tt - c^2 u_xx = 0,   x in [-1, 1],  t in [0, 1],
    boundary, initial displacement and initial velocity taken from the
    travelling wave  u(t,x) = sin(k pi (x - x_min)/L - k pi c t / L).

Reference solution: the travelling wave itself.

Run with:  python 1d_travelling_sine.py
"""

import numpy as np
import model

# ============================== settings =================================
method = "greedy"        # "greedy" or "full"
kernel_time = "Gaussian"   # time kernel: "Gaussian", "IMQ", "Matern_7_2"
kernel_space = "Gaussian"  # space kernel: "Gaussian", "IMQ", "Matern_7_2" (set different to mix kernels)

# Optimized Kernel shape parameters (time / space). Values for k = 3, greedy:
#   IMQ:        (2.0, 2.0)
ep_time = 4.214285714285714
ep_space = 2.2857142857142856

m = 70                   # Parameter to tune collocation functional pool (see thesis for details)
k = 3                    # wave number (mode)
nGreedy = 400            # number of greedy points (greedy only)
N_pred = 120             # prediction-grid resolution

# Optional: show animations of the selected collocation points (greedy)
# and of the solution over time.
make_animation = False

# Optional: select (ep_time, ep_space) by hold-out validation (greedy only).
# If True, the values above are overwritten by the search result.
optimize_parameters = False
ep_time_candidates = np.linspace(1.5, 10, 15)
ep_space_candidates = np.linspace(1.5, 10, 15)
# =========================================================================

c = 1.0 # wave speed
domain = dict(t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0)


def make_data():
    x_min, x_max = domain["x_min"], domain["x_max"]
    L = x_max - x_min

    def f_func(X):
        return np.zeros((X.shape[0], 1))

    def g_func(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        t = X[:, 0:1]
        x = X[:, 1:2]
        return np.sin(k * np.pi * (x - x_min) / L - k * np.pi * c * t / L)

    def v0_func(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x = X[:, -1:]
        return -(k * np.pi * c / L) * np.cos(k * np.pi * (x - x_min) / L)

    def u0_func(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x = X[:, -1:]
        return np.sin(k * np.pi * (x - x_min) / L)

    return f_func, g_func, u0_func, v0_func


def reference(t_pred, x_pred):
    x_min, x_max = domain["x_min"], domain["x_max"]
    L = x_max - x_min
    T, X = np.meshgrid(t_pred, x_pred, indexing="ij")
    return np.sin(k * np.pi * (X - x_min) / L - k * np.pi * c * T / L)


if __name__ == "__main__":
    if optimize_parameters:
        best_eps, _ = model.search_parameters(
            kernel_time=kernel_time, kernel_space=kernel_space, c=c, domain=domain, dim_space=1, n=m, data=make_data(),
            ep_time_candidates=ep_time_candidates,
            ep_space_candidates=ep_space_candidates, n_greedy=nGreedy,
        )
        if best_eps is not None:
            ep_time, ep_space = best_eps
            print(f"Selected (ep_time, ep_space) = ({ep_time:.6g}, {ep_space:.6g})")

    wave_model = model.solve(
        method=method, kernel_time=kernel_time, kernel_space=kernel_space, ep_time=ep_time, ep_space=ep_space,
        c=c, domain=domain, dim_space=1, n=m, data=make_data(),
        n_greedy=nGreedy, plot_selection=True, make_animation=make_animation,
    )
    model.evaluate(wave_model, domain=domain, dim_space=1,
                   reference=reference, pred_N=N_pred, c=c, make_animation=make_animation)
