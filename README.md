# Symmetric Kernel Collocation for the Wave Equation

Symmetric kernel collocation for the linear hyperbolic wave equation, using the
target dependent PDE-f-greedy algorithm together with product RBF
kernels that separate the temporal and spatial components. This repository
accompanies the thesis and reproduces its numerical experiments.

The method approximates the solution of

```
u_tt - c^2 * Laplace(u) = f      in the space-time interior,
                      u = g      on the spatial boundary,
                 u(0, .) = u0     (initial displacement),
              u_t(0, .) = v0     (initial velocity),
```

by symmetric kernel collocation/generalized interpolation.

## Repository layout

```
wave_kernel_greedy/
├── README.md
├── requirements.txt
├── main.py                       # template: solve your own f, g, u0, v0
├── src/                          # core implementation (the heart of the code)
│   ├── kernels.py                # product kernel + RBFs and its derivatives (Gaussian, IMQ, Matern_7_2, Wendland_C6)
│   ├── collocation.py            # full symmetric collocation matrix solver 
│   ├── greedy.py                 # PDE-f-greedy solver
│   └── grids.py                  # space-time collocation grids (1D and 2D)
├── examples/                     # one runnable script per thesis experiment
│   ├── model.py                  # required functions for all the scripts to avoid duplicate code
│   ├── 1d_vibrating_string.py
│   ├── 1d_travelling_sine.py
│   ├── 1d_compact_bump.py
│   ├── 2d_vibrating_membrane.py
│   └── 2d_travelling_sine.py
└── data/
    └── wave1d_full_snapshots_compact_L=4.npz   # FD reference for the compact bump example
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3 with NumPy, SciPy, Matplotlib and scikit-optimize.

## Running the experiments

There is one runnable file per experiment in `examples/`. Each file has a
clearly marked settings block at the top where you can change the solver, the
kernel, its shape parameters, the number of collocation points (functionals), the wave number,
and the number of greedy points (functionals). To run an experiment with the settings used in
the thesis:

```bash
cd examples
python 1d_vibrating_string.py
python 1d_travelling_sine.py
python 1d_compact_bump.py
python 2d_vibrating_membrane.py
python 2d_travelling_sine.py
```

Each script prints the relative error against the reference solution E_ref and shows a
few diagnostic plots (solution snapshots, absolute error, spatial error over
time).

### Adjusting parameters

Open any example and edit the settings block, for example to use a different
kernel on the compact bump:

```python
method = "greedy"        # "greedy" or "full"
time_kernel = "Wendland_C6"   # Gaussian, IMQ, Matern_7_2, Wendland_C6
space_kernel = "Wendland_C6"
ep_time = 1.013414695595158 # Kernel parameters for space and time
ep_space = 0.4923166109937793
m = 70 # Create collocation functional pool via parameter m, see thesis
nGreedy = 5000 # maximum expansion size n_max
```

The thesis values for every kernel are listed in the comments of each file. The
1D experiments support both the greedy and the full-collocation solver
(`method = "greedy"` / `"full"`); the 2D experiments were run with the greedy
solver only.

### Selecting the kernel parameters automatically

Every example (and `main.py`) can select the shape parameters
`(ep_time, ep_space)` by the hold-out validation procedure described in the
thesis, instead of using the fixed values. Set in the settings block:

```python
optimize_parameters = True
ep_time_candidates = np.linspace(1.5, 10, 15)
ep_space_candidates = np.linspace(1.5, 10, 15)
```

The greedy model is then trained on a training split for every candidate pair,
scored on a held-out validation set, and the best pair is used for the final
run. This applies to the greedy solver.

In addition, the compact-bump experiment offers a Bayesian optimization of the
shape parameters (via `scikit-optimize`), minimizing the relative error against
the finite-difference reference. In `examples/1d_compact_bump.py`:

```python
bayesian_optimize = True
bayes_bounds = (0.5, 10.0)
bayes_n_calls = 30
bayes_n_initial_points = 10
```

This requires `scikit-optimize` (`pip install scikit-optimize`).

### Available experiments

| Script                       | Methods       | Description                                  |
|------------------------------|---------------|----------------------------------------------|
| `1d_vibrating_string.py`     | greedy, full  | 1D vibrating string                          |
| `1d_travelling_sine.py`      | greedy, full  | 1D traveling sine wave                       |
| `1d_compact_bump.py`         | greedy, full  | 1D compactly supported bump (FD reference)   |
| `2d_vibrating_membrane.py`   | greedy        | 2D vibrating membrane                        |
| `2d_travelling_sine.py`      | greedy        | 2D traveling sine wave                       |

## Solving your own problem

`main.py` is a template that uses the same model file as the
examples. Edit its settings block and the four data functions
`f_func, g_func, u0_func, v0_func`, optionally provide a closed-form `analytical_solution`
(or set `reference_func = None`), then run:

```bash
python main.py
```

## Solvers

* **`PDEGreedyWave`** (`src/greedy.py`) — builds the approximant incrementally
  with the PDE-f-greedy rule, cycling through the interior, boundary, initial
  displacement and initial velocity functionals and selecting, within the chosen
  class, the point of maximal residual. Includes an incremental Cholesky update
  and an optional hold-out validation routine for selecting the kernel shape
  parameters (`model.search_parameters`).
* **`WaveCollocation`** (`src/collocation.py`) — assembles and solves the full
  symmetric collocation system on all points at once.
