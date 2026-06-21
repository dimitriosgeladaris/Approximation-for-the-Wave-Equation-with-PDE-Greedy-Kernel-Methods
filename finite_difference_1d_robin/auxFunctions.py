import numpy as np
from scipy.optimize import fsolve


def implicit_midpoint_step(x_i, JgradH, dt):
    def equations(vars):
        x_mid = 0.5 * (x_i + vars)
        eq1 = vars - x_i - dt * JgradH(x_mid)
        return eq1

    initial_guess = x_i
    x_new = fsolve(equations, initial_guess)
    return x_new


def implicit_midpoint(x0, JgradH, dt, endT):
    n_steps = int(np.floor(endT / dt))
    X = np.zeros((x0.shape[0], n_steps + 1))
    X[:, 0] = x0

    for i in range(n_steps):
        X[:, i + 1] = implicit_midpoint_step(X[:, i], JgradH, dt)

    tspan = np.linspace(0, n_steps * dt, n_steps + 1)
    return X, tspan


def symb_Euler_step(x_i, JgradH, dt):
    n = x_i.shape[0] // 2

    def equations(vars):
        x_mid = np.c_[x_i[0:n], vars[n:]].T
        eq1 = (
            np.atleast_2d(vars).T
            - np.atleast_2d(x_i).T
            - dt * JgradH(x_mid)
        )
        return eq1[:, 0]

    initial_guess = x_i
    x_new = fsolve(equations, initial_guess, tol=1e-6)
    return x_new


def symb_Euler(x0, JgradH, dt, endT):
    n_steps = int(np.floor(endT / dt))
    X = np.zeros((len(x0), n_steps + 1))
    X[:, 0] = x0

    for i in range(n_steps):
        X[:, i + 1] = symb_Euler_step(X[:, i], JgradH, dt)

    tspan = np.linspace(0, n_steps * dt, n_steps + 1)
    return X, tspan
