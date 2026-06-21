"""
Experiment: 1D compactly supported bump.

    u_tt - c^2 u_xx = 0,   x in [0, 4],  t in [0, 2],  c = 4,
    u = 0 on the boundary,  u_t(0,x) = 0,
    u(0,x) = compactly supported bump centred in the domain.

Reference solution: finite-difference snapshots bundled in ../data/.

Run with:  python 1d_compact_bump.py
"""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import model

# ============================== settings =================================
method = "greedy"        # "greedy" or "full"
kernel_time = "Gaussian"   # time kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6"
kernel_space = "Gaussian"  # space kernel: "Gaussian", "IMQ", "Matern_7_2", "Wendland_C6" (set different to mix kernels)

# Optimized Kernel shape parameters (time / space). Thesis values (n = 70), greedy:
#   Gaussian:    (13.99301785242179,  6.830293179941049)
#   IMQ:         (9.985646114961668,  3.146390118797252)
#   Matern_7_2:  (10.659572440270338, 4.017981501720107)
#   Wendland_C6: (1.013414695595158,  0.4923166109937793)
ep_time = 13.99301785242179
ep_space = 6.830293179941049

m = 70                   # Parameter to tune collocation functional pool (see thesis for details).
nGreedy = 5000           # number of greedy points (greedy only)
N_pred = 120             # prediction-grid resolution

# Optional: show animations of the selected collocation points (greedy)
# and of the solution over time.
make_animation = False

# Optional: select (ep_time, ep_space) by hold-out validation (greedy only).
# If True, the values above are overwritten by the search result.
optimize_parameters = False
ep_time_candidates = np.linspace(0.5, 50, 5)
ep_space_candidates = np.linspace(0.5, 50, 5)

# Optional: Bayesian optimization of (ep_time, ep_space) with scikit-optimize,
# minimizing the relative error against the FD reference. If True, the values
# above are overwritten by the optimizer's result. Requires scikit-optimize
# (pip install scikit-optimize).
bayesian_optimize = False
bayes_bounds = (0.5, 10.0)   # (low, high) used for both ep_time and ep_space
bayes_n_calls = 30
bayes_n_initial_points = 10
# =========================================================================

c = 4.0 # wave speed
domain = dict(t_min=0.0, t_max=2.0, x_min=0.0, x_max=4.0)

# Path to the bundled finite-difference reference solution.
_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "wave1d_full_snapshots_compact_L=4.npz",
)


def make_data():
    x_min, x_max = domain["x_min"], domain["x_max"]

    def f_func(X):
        return np.zeros((X.shape[0], 1))

    def g_func(X):
        return np.zeros((X.shape[0], 1))

    def v0_func(X):
        return np.zeros((X.shape[0], 1))

    def u0_func(X, center=None, width=2):
        """u(0,x) = exp(1) exp(-1 / (1 - ((x - x0)/(width/2))^2)) on the bump."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x = X[:, -1:]
        if center is None:
            center = 0.5 * (x_min + x_max)
        u = np.zeros_like(x)
        mask = np.abs(x - center) < (width / 2)
        u[mask] = np.exp(1) * np.exp(
            -1 / (1 - ((x[mask] - center) / (width / 2)) ** 2)
        )
        return u

    return f_func, g_func, u0_func, v0_func


def reference_solution(t_pred, x_pred):
    data = np.load(_DATA, allow_pickle=True)
    snapshots = [np.array(snap) for snap in data["snapshots"]]
    sol_fd = snapshots[0]  # first simulation as reference

    L = data["L"]
    endT = data["endT"]
    N_fd = data["N"]
    Nt_fd = sol_fd.shape[1]
    x_fd = np.linspace(0, L, N_fd)
    t_fd = np.linspace(0, endT, Nt_fd)
    U_fd_full = sol_fd[:N_fd, :].T  # (Nt_fd, N_fd)

    T, X = np.meshgrid(t_pred, x_pred, indexing="ij")
    points = np.column_stack((T.ravel(), X.ravel()))
    interp_fd = RegularGridInterpolator((t_fd, x_fd), U_fd_full, bounds_error=True)
    return interp_fd(points).reshape((len(t_pred), len(x_pred)))


def run_bayesian_optimization():
    """Bayesian optimization of the shape parameters via scikit-optimize.

    Minimizes the relative error against the FD reference. Returns the best
    (ep_time, ep_space).
    """
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    space = [
        Real(bayes_bounds[0], bayes_bounds[1], name="ep_time"),
        Real(bayes_bounds[0], bayes_bounds[1], name="ep_space"),
    ]

    @use_named_args(space)
    def objective(ep_time, ep_space):
        wave_model = model.solve(
            method="greedy", kernel_time=kernel_time, kernel_space=kernel_space, ep_time=ep_time, ep_space=ep_space,
            c=c, domain=domain, dim_space=1, n=m, data=make_data(),
            n_greedy=nGreedy,
        )
        err = model.evaluate(wave_model, domain=domain, dim_space=1,
                             reference=reference_solution, pred_N=N_pred, c=c, plot=False)
        if err is None or np.isnan(err) or np.isinf(err):
            return 1e6  # penalize invalid solutions heavily
        return err

    result = gp_minimize(objective, space, n_calls=bayes_n_calls,
                         n_initial_points=bayes_n_initial_points)
    print(f"Bayesian optimum: ep_time = {result.x[0]:.6g}, "
          f"ep_space = {result.x[1]:.6g}, error = {result.fun:.6e}")
    return result.x[0], result.x[1] 


if __name__ == "__main__":
    if bayesian_optimize:
        ep_time, ep_space = run_bayesian_optimization()
    elif optimize_parameters:
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
                   reference=reference_solution, pred_N=N_pred, c=c, make_animation=make_animation)
