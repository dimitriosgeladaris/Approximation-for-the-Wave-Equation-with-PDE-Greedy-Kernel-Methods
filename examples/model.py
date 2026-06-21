"""
Shared driver for the example scripts.

Every example file in this folder is a thin, editable script: it fixes the
problem (domain, wave speed, data functions, reference solution) and a small
block of settings (method, kernel, shape parameters, point counts), then calls
the functions in this module:

    wave_model = model.solve(...)        # build kernels + grid, run the solver
    model.evaluate(wave_model, ...)      # predict on a grid, report error, plot

Optionally, the kernel shape parameters can be selected by the residual-based
validation procedure with

    best_eps, _ = model.search_parameters(...)

This file implements all shared and needed functionality.
"""

import os
import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt

# Make the core package in ../src importable from any example script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import kernels  # noqa: E402
import grids  # noqa: E402
from greedy import PDEGreedyWave  # noqa: E402
from collocation import WaveCollocation  # noqa: E402


KERNEL_CLASSES = {
    "Gaussian": kernels.Gaussian,
    "IMQ": kernels.IMQ,
    "Matern_7_2": kernels.Matern_7_2,
    "Wendland_C6": kernels.Wendland_C6,
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_kernels(kernel_time, ep_time, ep_space, dim_space, kernel_space=None):
    """Instantiate the time (1D) and space (dim_space) kernels by name.

    ``kernel_time`` selects the time kernel. ``kernel_space`` selects the space
    kernel; if None, the same type as ``kernel_time`` is used (the default,
    which reproduces the original product kernel with a single kernel type).
    """
    if kernel_space is None:
        kernel_space = kernel_time
    for name in (kernel_time, kernel_space):
        if name not in KERNEL_CLASSES:
            raise ValueError(
                f"Unknown kernel '{name}'. Available: {', '.join(KERNEL_CLASSES)}"
            )
    KTime = KERNEL_CLASSES[kernel_time]
    KSpace = KERNEL_CLASSES[kernel_space]
    return KTime(ep=ep_time, dim=1), KSpace(ep=ep_space, dim=dim_space)


def build_grid(domain, dim_space, n):
    """Create the collocation point sets for the domain (n points per class)."""
    d = domain
    if dim_space == 1:
        return grids.create_time_space_grid_1d(
            d["t_min"], d["t_max"], d["x_min"], d["x_max"], n, n, n, n
        )
    return grids.create_time_space_grid_2d(
        d["t_min"], d["t_max"], d["x_min"], d["x_max"],
        d["y_min"], d["y_max"], n, n, n, n
    )


def prediction_grid(domain, dim_space, pred_N):
    """Build the uniform prediction grid used to evaluate the error."""
    d = domain
    t_pred = np.linspace(d["t_min"], d["t_max"], pred_N)
    x_pred = np.linspace(d["x_min"], d["x_max"], pred_N)
    if dim_space == 1:
        return t_pred, x_pred, None
    y_pred = np.linspace(d["y_min"], d["y_max"], pred_N)
    return t_pred, x_pred, y_pred


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
def solve(method, kernel_time, ep_time, ep_space, c, domain, dim_space, n, data,
          n_greedy=None, plot_selection=False, make_animation=False,
          kernel_space=None):
    """Build kernels and grid, run the chosen solver, and return the model.

    Parameters
    ----------
    method : "greedy" for f-greedy or "full" for full collocation matrix.
    kernel_time : time-kernel name (Gaussian, IMQ, Matern_7_2, Wendland_C6).
    ep_time, ep_space : kernel shape parameters in time and space.
    c : wave speed.
    domain : dict with t_min/t_max/x_min/x_max (+ y_min/y_max in 2D).
    dim_space : 1 or 2.
    n : number of collocation points per class (per axis).
    data : tuple (f_func, g_func, u0_func, v0_func).
    n_greedy : number of greedy points (required for method="greedy").
    plot_selection : if True and greedy, show the point-selection / residual plots.
    make_animation : if True and greedy, animate the order in which the
                     collocation points were selected (greedy only).
    kernel_space : space-kernel name; if None, the same type as ``kernel_time``
                   is used (so a single kernel type is applied to time and space).
    """
    k_time, k_space = build_kernels(kernel_time, ep_time, ep_space, dim_space,
                                    kernel_space=kernel_space)
    f_func, g_func, u0_func, v0_func = data
    (interior_points, boundary_points,
     initial_displacement_points, initial_velocity_points) = build_grid(
        domain, dim_space, n
    )

    if method == "greedy":
        if n_greedy is None:
            raise ValueError("n_greedy must be set for method='greedy'.")
        model = PDEGreedyWave(
            k_time, k_space, f_func, g_func, u0_func, v0_func,
            interior_points, boundary_points,
            initial_displacement_points, initial_velocity_points, c=c,
        )
        start = time()
        model.fgreedy(nMax=n_greedy)
        print(f"Training via PDE-f-greedy of {n_greedy} points "
              f"took {time() - start:.2f} seconds.")
        if plot_selection:
            model.plot_selected_points_and_residuals(makeAnimation=make_animation)

    elif method == "full":
        model = WaveCollocation(
            k_time, k_space, f_func, g_func, u0_func, v0_func,
            interior_points, boundary_points,
            initial_displacement_points, initial_velocity_points, c=c,
        )
        start = time()
        model.fit()
        print(f"Full collocation solve took {time() - start:.2f} seconds.")

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'greedy' or 'full'.")

    return model


def search_parameters(kernel_time, c, domain, dim_space, n, data,
                      ep_time_candidates, ep_space_candidates, n_greedy,
                      kernel_space=None):
    """Optional: hold-out search over kernel shape parameters (greedy only).

    Returns (best_eps, best_model) from the greedy model's own
    ``select_kernel_parameters``. Useful for reproducing the parameter search;
    the example scripts otherwise use the fixed values reported in the thesis.
    ``kernel_space`` selects the space kernel; if None, the same type as
    ``kernel_time`` is used.
    """
    k_time, k_space = build_kernels(kernel_time, ep_time_candidates[0],
                                    ep_space_candidates[0], dim_space,
                                    kernel_space=kernel_space)
    f_func, g_func, u0_func, v0_func = data
    (interior_points, boundary_points,
     initial_displacement_points, initial_velocity_points) = build_grid(
        domain, dim_space, n
    )
    model = PDEGreedyWave(
        k_time, k_space, f_func, g_func, u0_func, v0_func,
        interior_points, boundary_points,
        initial_displacement_points, initial_velocity_points, c=c,
    )
    return model.select_kernel_parameters(
        ep_time_candidates, ep_space_candidates, nMax=n_greedy
    )


# ---------------------------------------------------------------------------
# Evaluate + plot
# ---------------------------------------------------------------------------
def evaluate(model, domain, dim_space, reference=None, pred_N=120, c=1.0, plot=True,
             make_animation=False):
    """Predict on the grid, report the error and show plots.

    If ``reference`` is None (e.g. a custom problem without a closed-form
    solution), only the predicted solution is plotted and no error is computed.
    ``reference`` must be a callable: reference(t_pred, x_pred[, y_pred]) -> U_ref.
    If ``plot`` is False, no figures are drawn (useful inside an optimization
    loop); the error is still returned.
    If ``make_animation`` is True, an animation of the solution over time is
    shown (prediction vs. reference when a reference is available).
    """
    t_pred, x_pred, y_pred = prediction_grid(domain, dim_space, pred_N)

    if dim_space == 1:
        T, X = np.meshgrid(t_pred, x_pred, indexing="ij")
        X_pred = np.column_stack((T.ravel(), X.ravel()))
        U = model.predict_s(X_pred).ravel().reshape((pred_N, pred_N))

        if reference is None:
            if plot:
                _plot_1d_solution(t_pred, x_pred, U, c)
            if make_animation:
                _animate_1d(t_pred, x_pred, U, None)
            return None

        U_ref = reference(t_pred, x_pred)
        error = np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref)
        if plot:
            print(f"Relative L2 error vs. reference solution: {error:.6e}")
            _plot_1d(t_pred, x_pred, U, U_ref, c)
        if make_animation:
            _animate_1d(t_pred, x_pred, U, U_ref)
        return error

    # 2D
    T, X, Y = np.meshgrid(t_pred, x_pred, y_pred, indexing="ij")
    X_pred = np.column_stack((T.ravel(), X.ravel(), Y.ravel()))
    U = model.predict_s(X_pred).ravel().reshape((pred_N, pred_N, pred_N))

    if reference is None:
        if plot:
            _plot_2d_solution(t_pred, x_pred, y_pred, U)
        if make_animation:
            _animate_2d(t_pred, x_pred, y_pred, U, None)
        return None

    U_ref = reference(t_pred, x_pred, y_pred)
    error = np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref)
    if plot:
        print(f"Relative L2 error vs. reference solution: {error:.6e}")
        _plot_2d(t_pred, x_pred, y_pred, U, U_ref)
    if make_animation:
        _animate_2d(t_pred, x_pred, y_pred, U, U_ref)
    return error


# ----------------------------- 1D plots -----------------------------------
def _plot_1d(t_vals, x_vals, U, U_ref, c):
    times_to_show = [0, len(t_vals) // 4, len(t_vals) // 2, -1]

    plt.figure(figsize=(8, 6))
    for ti in times_to_show:
        plt.plot(x_vals, U[ti], label=f"Prediction t = {t_vals[ti]:.3f}")
        plt.plot(x_vals, U_ref[ti], "--", label=f"Reference t = {t_vals[ti]:.3f}")
    plt.title(f"Wave equation solution at different times (c={c})")
    plt.xlabel("x")
    plt.ylabel("u(t,x)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    for ti in times_to_show:
        plt.plot(x_vals, np.abs(U[ti] - U_ref[ti]),
                 label=f"Error t = {t_vals[ti]:.3f}")
    plt.title("Absolute error between prediction and reference")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    dx = x_vals[1] - x_vals[0]
    spatial_error_abs = np.linalg.norm(U - U_ref, axis=1) * np.sqrt(dx)
    plt.figure(figsize=(8, 5))
    plt.semilogy(t_vals, spatial_error_abs, "-o", markersize=4)
    plt.xlabel("Time t")
    plt.ylabel(r"Absolute $\ell^2$ error over space")
    plt.title(r"Spatial $\ell^2$-error over time")
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.show()


def _plot_1d_solution(t_vals, x_vals, U, c):
    times_to_show = [0, len(t_vals) // 4, len(t_vals) // 2, -1]
    plt.figure(figsize=(8, 6))
    for ti in times_to_show:
        plt.plot(x_vals, U[ti], label=f"t = {t_vals[ti]:.3f}")
    plt.title(f"Predicted solution (c={c})")
    plt.xlabel("x")
    plt.ylabel("u(t,x)")
    plt.legend()
    plt.grid()
    plt.show()


# ----------------------------- 2D plots -----------------------------------
def _plot_2d(t_vals, x_vals, y_vals, U, U_ref):
    from matplotlib.colors import LogNorm

    snapshot_times = [t_vals[0], t_vals[len(t_vals) // 2], t_vals[-1]]
    vmin, vmax = U.min(), U.max()

    abs_err_all = np.abs(U - U_ref)
    err_max = abs_err_all.max()
    nonzero = abs_err_all[abs_err_all > 0]
    err_floor = max(nonzero.min() if nonzero.size else 1e-12, 1e-12)

    for t_target in snapshot_times:
        frame = int(np.argmin(np.abs(t_vals - t_target)))
        t = t_vals[frame]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].pcolormesh(x_vals, y_vals, U[frame].T, shading="auto",
                                 cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Predicted solution at t = {t:.3f}")
        fig.colorbar(im0, ax=axes[0], label="u(t,x,y)")

        im1 = axes[1].pcolormesh(x_vals, y_vals, U_ref[frame].T, shading="auto",
                                 cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title(f"Reference solution at t = {t:.3f}")
        fig.colorbar(im1, ax=axes[1], label="u(t,x,y)")

        err_frame = np.maximum(np.abs(U[frame] - U_ref[frame]).T, err_floor)
        im2 = axes[2].pcolormesh(x_vals, y_vals, err_frame, shading="auto",
                                 cmap="inferno",
                                 norm=LogNorm(vmin=err_floor, vmax=err_max))
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title(f"Absolute error at t = {t:.3f}")
        fig.colorbar(im2, ax=axes[2], label="Absolute error")

        plt.tight_layout()
        plt.show()

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    quad_weight = np.sqrt(dx * dy)

    spatial_error_abs = np.linalg.norm(U - U_ref, axis=(1, 2)) * quad_weight
    plt.figure(figsize=(8, 5))
    plt.semilogy(t_vals, spatial_error_abs, "-o", markersize=4)
    plt.xlabel("Time t")
    plt.ylabel(r"Absolute $\ell^2$ error over space")
    plt.title(r"Spatial $\ell^2$-error over time")
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.show()

def _plot_2d_solution(t_vals, x_vals, y_vals, U):
    snapshot_times = [t_vals[0], t_vals[len(t_vals) // 2], t_vals[-1]]
    vmin, vmax = U.min(), U.max()
    for t_target in snapshot_times:
        frame = int(np.argmin(np.abs(t_vals - t_target)))
        t = t_vals[frame]
        plt.figure(figsize=(7, 5))
        im = plt.pcolormesh(x_vals, y_vals, U[frame].T, shading="auto",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="u(t,x,y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Predicted solution at t = {t:.3f}")
        plt.tight_layout()
        plt.show()


# ----------------------------- animations ---------------------------------
# Keep references to created animations so they are not garbage-collected
# before the figure is shown (matplotlib only holds a weak reference).
_ANIMATIONS = []


def _animate_1d(t_vals, x_vals, U, U_ref=None):
    """Animate the 1D solution over time (prediction, and reference if given)."""
    from matplotlib.animation import FuncAnimation

    if U_ref is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        line_pred, = ax1.plot([], [], label="Prediction", linewidth=2)
        line_ref, = ax1.plot([], [], "--", label="Reference", linewidth=2)
        line_err, = ax2.plot([], [], color="red", linewidth=2)

        ax1.set_xlim(x_vals.min(), x_vals.max())
        ax1.set_ylim(min(U.min(), U_ref.min()), max(U.max(), U_ref.max()))
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(t,x)")
        ax1.set_title("Wave equation solution")
        ax1.legend()
        ax1.grid()

        ax2.set_xlim(x_vals.min(), x_vals.max())
        ax2.set_ylim(np.abs(U - U_ref).min(), np.abs(U - U_ref).max())
        ax2.set_xlabel("x")
        ax2.set_ylabel("Error")
        ax2.set_title("Absolute error")
        ax2.grid()

        time_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes,
                             verticalalignment="top")

        def animate(frame):
            line_pred.set_data(x_vals, U[frame])
            line_ref.set_data(x_vals, U_ref[frame])
            line_err.set_data(x_vals, np.abs(U[frame] - U_ref[frame]))
            time_text.set_text(f"t = {t_vals[frame]:.3f}")
            return line_pred, line_ref, line_err, time_text
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        line_pred, = ax1.plot([], [], label="Prediction", linewidth=2)
        ax1.set_xlim(x_vals.min(), x_vals.max())
        ax1.set_ylim(U.min(), U.max())
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(t,x)")
        ax1.set_title("Wave equation solution")
        ax1.legend()
        ax1.grid()
        time_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes,
                             verticalalignment="top")

        def animate(frame):
            line_pred.set_data(x_vals, U[frame])
            time_text.set_text(f"t = {t_vals[frame]:.3f}")
            return line_pred, time_text

    anim = FuncAnimation(fig, animate, frames=len(t_vals), interval=50, blit=True)
    _ANIMATIONS.append(anim)
    plt.tight_layout()
    plt.show()
    return anim


def _animate_2d(t_vals, x_vals, y_vals, U, U_ref=None):
    """Animate the 2D solution over time (prediction, reference, error)."""
    from matplotlib.animation import FuncAnimation

    if U_ref is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].pcolormesh(x_vals, y_vals, U[0].T, shading="auto",
                                 cmap="viridis", vmin=U.min(), vmax=U.max())
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Predicted solution")
        fig.colorbar(im0, ax=axes[0], label="u(t,x,y)")

        im1 = axes[1].pcolormesh(x_vals, y_vals, U_ref[0].T, shading="auto",
                                 cmap="viridis", vmin=U_ref.min(), vmax=U_ref.max())
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Reference solution")
        fig.colorbar(im1, ax=axes[1], label="u(t,x,y)")

        im2 = axes[2].pcolormesh(x_vals, y_vals, np.abs(U[0] - U_ref[0]).T,
                                 shading="auto", cmap="inferno",
                                 vmin=0, vmax=np.abs(U - U_ref).max())
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Absolute error")
        fig.colorbar(im2, ax=axes[2], label="Absolute error")

        time_text = axes[0].text(0.05, 0.95, f"t = {t_vals[0]:.3f}",
                                transform=axes[0].transAxes,
                                verticalalignment="top", color="white", fontsize=12)

        def animate(frame):
            im0.set_array(U[frame].T.ravel())
            im1.set_array(U_ref[frame].T.ravel())
            im2.set_array(np.abs(U[frame] - U_ref[frame]).T.ravel())
            time_text.set_text(f"t = {t_vals[frame]:.3f}")
            return im0, im1, im2, time_text
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        im0 = ax.pcolormesh(x_vals, y_vals, U[0].T, shading="auto",
                            cmap="viridis", vmin=U.min(), vmax=U.max())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Predicted solution")
        fig.colorbar(im0, ax=ax, label="u(t,x,y)")
        time_text = ax.text(0.05, 0.95, f"t = {t_vals[0]:.3f}",
                           transform=ax.transAxes, verticalalignment="top",
                           color="white", fontsize=12)

        def animate(frame):
            im0.set_array(U[frame].T.ravel())
            time_text.set_text(f"t = {t_vals[frame]:.3f}")
            return im0, time_text

    anim = FuncAnimation(fig, animate, frames=len(t_vals), interval=100,
                        blit=False, repeat=True)
    _ANIMATIONS.append(anim)
    plt.tight_layout()
    plt.show()
    return anim
