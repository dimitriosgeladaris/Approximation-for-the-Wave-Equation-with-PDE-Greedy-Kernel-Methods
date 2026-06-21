import kernels
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PDEGreedyWave:
    def __init__(self, k_time: kernels.Kernel, k_space: kernels.Kernel,
                f_func, g_func, u0_func, v0_func,
                 interior_points, boundary_points, initial_displacement_points, initial_velocity_points, c=1.0):
        
        # store kernel CLASSES, not only instances
        self.kernel_time_class = type(k_time)
        self.kernel_space_class = type(k_space)

        # store eps for possible reuse
        self.eps_time = k_time.ep
        self.eps_space = k_space.ep

        self.kernel_time = k_time
        self.kernel_space = k_space
        self.c = c
        self.kernel = kernels.ProductKernel(k_time, k_space, c=c)
        self.f_func = f_func
        self.g_func = g_func
        self.u0_func = u0_func
        self.v0_func = v0_func

        self.interior_points = interior_points  # shape (N_interior, dim_space + 1)
        self.boundary_points = boundary_points  # shape (N_boundary, dim_space + 1)
        self.initial_displacement_points = initial_displacement_points    # shape (N_initial_displacement, dim_space + 1)
        self.initial_velocity_points = initial_velocity_points  # shape (N_initial_velocity, dim_space + 1)


        self.N_interior = interior_points.shape[0]
        self.N_boundary = boundary_points.shape[0]
        self.N_initial_displacement = initial_displacement_points.shape[0]
        self.N_initial_velocity = initial_velocity_points.shape[0]
        self.N = (self.N_interior + self.N_boundary +
                        self.N_initial_displacement + self.N_initial_velocity)
        
        self.output_int = None  # Right-hand side values at interior points
        self.output_bnd = None  # Boundary condition values at boundary points
        self.output_init_disp = None  # Initial displacement values
        self.output_init_vel = None  # Initial velocity values
        
        self.nMax = None  # Maximum number of basis functions/Selected greedy points
        self.selected_points_int = None  # List of selected collocation points and their operators
        self.selected_points_bnd = None  # List of selected collocation points and their operators
        self.selected_points_init_disp = None  # List of selected collocation points and their operators
        self.selected_points_init_vel = None  # List of selected collocation points and their operators
        self.selected_points = None  # Combined list of selected collocation points

        self.selected_operators_all = None
        self.alpha = None  # Coefficients of the kernel expansion
        self.C = None # Cholesky factor
        self.maxRes = None  # Residuals at each iteration
        self.maxResInt = None  # Residuals at interior points
        self.maxResBnd = None  # Residuals at boundary points
        self.maxResInitDisp = None  # Residuals at initial displacement points
        self.maxResInitVel = None  # Residuals at initial velocity points

    def fgreedy(self, nMax):
        """
        f-greedy method to solve the wave equation using symmetric kernel collocation.
        Functionals are selected in the order:
        PDE - Boundary - PDE - Initial Displacement - PDE - Initial Velocity - repeat
        """
        self.nMax = nMax

        # Get output data at all collocation points
        f_interior, f_boundary, f_initial_displacement, f_initial_velocity = self.get_output_data()
        self.output_int = f_interior
        self.output_bnd = f_boundary
        self.output_init_disp = f_initial_displacement
        self.output_init_vel = f_initial_velocity

        f_all = np.vstack((f_interior, f_boundary, f_initial_displacement, f_initial_velocity))

        # Set boundary types
        INTERIOR = 0
        BOUNDARY = 1
        INITIAL_DISPLACEMENT = 2
        INITIAL_VELOCITY = 3

        # Start with the initial displacement functional
        idx = np.argmax(np.abs(f_initial_displacement))
        selected_point_init = self.initial_displacement_points[idx]  # shape (dim,)
        self.selected_points_init_disp = [selected_point_init]

        self.selected_points_int = []
        self.selected_points_bnd = []
        self.selected_points_init_vel = []

        iteration_count_int = []
        iteration_count_bnd = []
        iteration_count_init_disp = [1]
        iteration_count_init_vel = []

        self.selected_points = [selected_point_init]  # list of points
        selected_operators_all = [INITIAL_DISPLACEMENT]

        rhs = f_initial_displacement[idx].reshape(-1, 1)  # ensure 2D column vector

        G = np.zeros((self.N, nMax))
        eps = 0 # Regularization parameter to avoid division by zero, for value bigger than 0 we do regression -> More duplicates possible!
        self.maxRes = np.ones(nMax)
        self.maxResInt = np.ones(nMax)
        self.maxResBnd = np.ones(nMax)
        self.maxResInitDisp = np.ones(nMax)
        self.maxResInitVel = np.ones(nMax)

        # Kernel at the first point
        K0 = self.kernel_evaluation(
            selected_point_init, INITIAL_DISPLACEMENT,
            selected_point_init, INITIAL_DISPLACEMENT
        )[0, 0]
        self.C = np.atleast_2d(1 / np.sqrt(eps + K0))
        self.alpha = np.atleast_2d(rhs / K0)

        order = [INTERIOR, BOUNDARY, INTERIOR, INITIAL_DISPLACEMENT, INTERIOR,INITIAL_VELOCITY] # Order of functionals to select, this is one "cycle"
        for i in range(nMax):
            # Compute approximation on all collocation points using the last selected point
            g = self.orderedGeneralizedInterpolantVec(
                    selected_point=self.selected_points[-1],
                    operator_selected=selected_operators_all[-1]
                )
            G[:, i] = g[:, 0]
            snTemp = G[:, :(i + 1)] @ self.alpha

            # Compute and store residuals
            residual = f_all - snTemp
            self.maxRes[i] = np.max(np.abs(residual))
            self.maxResInt[i] = np.max(np.abs(residual[:self.N_interior]))
            self.maxResBnd[i] = np.max(np.abs(residual[self.N_interior:self.N_interior + self.N_boundary]))
            self.maxResInitDisp[i] = np.max(np.abs(residual[self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement]))
            self.maxResInitVel[i] = np.max(np.abs(residual[self.N_interior + self.N_boundary + self.N_initial_displacement:]))

            
            ## Determine next operator type to select, we skip types that are below tolerance \eta and check the cycle. If all operators under res, we stop.
            tol_residual = 1e-6
            residual_map = {
                INTERIOR: self.maxResInt[i],
                BOUNDARY: self.maxResBnd[i],
                INITIAL_DISPLACEMENT: self.maxResInitDisp[i],
                INITIAL_VELOCITY: self.maxResInitVel[i],
            }

            # Cyclic scan starting from prescribed order
            start_idx = i % len(order)
            operator_type = None

            # Determine next candidate operator type based on maximum residual
            for k in range(len(order)):
                candidate = order[(start_idx + k) % len(order)]
                if residual_map[candidate] >= tol_residual:
                    operator_type = candidate
                    break

            # If all possible operators to select are below tolerance, stop
            if operator_type is None:
                print(f"All residuals below tolerance {tol_residual:.1e} at iteration {i}.")
                # Truncate arrays to actual length
                self.maxRes = self.maxRes[:i+1]
                self.maxResInt = self.maxResInt[:i+1]
                self.maxResBnd = self.maxResBnd[:i+1]
                self.maxResInitDisp = self.maxResInitDisp[:i+1]
                self.maxResInitVel = self.maxResInitVel[:i+1]
                break


            selected_operators_all.append(operator_type)

            ## Select new point based on operator (via f-greedy)
            if operator_type == INTERIOR:
                errors = self.output_int - snTemp[:self.N_interior]
                idx_new = np.argmax(np.abs(errors))
                new_point = self.interior_points[idx_new]
                self.selected_points_int.append(new_point)
                iteration_count_int.append(i + 2)
            elif operator_type == BOUNDARY:
                errors = self.output_bnd - snTemp[self.N_interior:self.N_interior + self.N_boundary]
                idx_new = np.argmax(np.abs(errors))
                new_point = self.boundary_points[idx_new]
                self.selected_points_bnd.append(new_point)
                iteration_count_bnd.append(i + 2)
            elif operator_type == INITIAL_DISPLACEMENT:
                errors = self.output_init_disp - snTemp[self.N_interior+self.N_boundary:
                                                        self.N_interior+self.N_boundary+self.N_initial_displacement]
                idx_new = np.argmax(np.abs(errors))
                new_point = self.initial_displacement_points[idx_new]
                self.selected_points_init_disp.append(new_point)
                iteration_count_init_disp.append(i + 2)
            elif operator_type == INITIAL_VELOCITY:
                errors = self.output_init_vel - snTemp[self.N_interior+self.N_boundary+self.N_initial_displacement:]
                idx_new = np.argmax(np.abs(errors))
                new_point = self.initial_velocity_points[idx_new]
                self.selected_points_init_vel.append(new_point)
                iteration_count_init_vel.append(i + 2)
            else:
                raise ValueError("Invalid operator type selected.")

            # Update selected points
            self.selected_points.append(new_point)

            # Kernel column between new point and previous functionals
            KX = self.greedyGeneralizedInterpolantVec(
                    selected_points=self.selected_points[:-1],
                    selected_operators=selected_operators_all[:-1],
                    new_point=new_point,
                    new_functional=operator_type
                )

            # Right-hand side value for new point
            if operator_type == INTERIOR:
                rhs_val = f_interior[idx_new]
            elif operator_type == BOUNDARY:
                rhs_val = f_boundary[idx_new]
            elif operator_type == INITIAL_DISPLACEMENT:
                rhs_val = f_initial_displacement[idx_new]
            elif operator_type == INITIAL_VELOCITY:
                rhs_val = f_initial_velocity[idx_new]

            rhs = np.vstack([rhs, rhs_val.reshape(1, 1)])  # Reshape to (1, 1) for proper stacking

            # Kernel self-interaction
            KXX_val = self.kernel_evaluation(new_point, operator_type, new_point, operator_type)
            if isinstance(KXX_val, np.ndarray):
                KXX = KXX_val.item() + eps
            else:
                KXX = KXX_val + eps

            # ---- Cholesky update with breakdown detection ----
            dm     = self.C @ KX.reshape(-1, 1)
            dmm_sq = KXX - (dm.T @ dm)[0, 0]

            # Ensure dmm_sq is positive and above a small tolerance to avoid numerical issues in Cholesky factorization. If not, we stop the greedy selection to prevent breakdown.
            dmm_tol = 1e-12
            if (not np.isfinite(dmm_sq)) or dmm_sq <= dmm_tol:
                print(f"Cholesky breakdown at iteration {i}: "
                      f"dmm^2 = {dmm_sq:.3e}. "
                      f"Stopping with {i+1} basis functions; "
                      f"last valid maxRes = {self.maxRes[i]:.3e}.")

                # If breakdown occurs, we remove the last selected point and operator to keep only valid ones.
                selected_operators_all.pop()
                self.selected_points.pop()
                if operator_type == INTERIOR:
                    self.selected_points_int.pop()
                elif operator_type == BOUNDARY:
                    self.selected_points_bnd.pop()
                elif operator_type == INITIAL_DISPLACEMENT:
                    self.selected_points_init_disp.pop()
                elif operator_type == INITIAL_VELOCITY:
                    self.selected_points_init_vel.pop()
                self.selected_operators_all = selected_operators_all

                # residual[i] was computed at the top of this iteration
                # with the still-valid alpha, so keep up to and including i.
                self.maxRes         = self.maxRes[:i+1]
                self.maxResInt      = self.maxResInt[:i+1]
                self.maxResBnd      = self.maxResBnd[:i+1]
                self.maxResInitDisp = self.maxResInitDisp[:i+1]
                self.maxResInitVel  = self.maxResInitVel[:i+1]
                break

            dmm = np.sqrt(dmm_sq)
            cmm = 1.0 / dmm

            cm = -(self.C.T @ dm) * cmm
            cm = cm.reshape(-1, 1)

            self.C = np.c_[
                np.r_[self.C, cm.T],
                np.r_[np.zeros((cm.shape[0], 1)), np.array([[cmm]])]
            ]

            self.alpha = self.C.T @ (self.C @ rhs)
            self.selected_operators_all = selected_operators_all

        # Print number of selected points of each type
        num_int = len(self.selected_points_int)
        num_bnd = len(self.selected_points_bnd)
        num_init_disp = len(self.selected_points_init_disp)
        num_init_vel = len(self.selected_points_init_vel)
        print(f"Selected functionals: Interior (PDE): {num_int}, Boundary: {num_bnd}, Initial Displacement: {num_init_disp}, Initial Velocity: {num_init_vel}")
        # Check uniqueness of selected points
        unique_functionals = set((tuple(p), op) for p, op in zip(self.selected_points, self.selected_operators_all))
        if len(unique_functionals) < len(self.selected_points):
            n_dup = len(self.selected_points) - len(unique_functionals)
            print(f"Warning: {n_dup} TRUE duplicate (point, operator) functionals "
                f"were selected — this would imply rank-deficient Gram matrix.")
        return self

    def predict_s(self, X_pred):
        """
        Vectorized prediction of u(t,x) at new points X_pred.
        X_pred: (n_pred, dim_total) where dim_total = 1 (time) + 1 (space) = 2 for 1D
        """
        n_pred = X_pred.shape[0]
        n_basis = len(self.selected_points)
        G_pred = np.zeros((n_pred, n_basis))

        # Convert selected points and operators to arrays
        selected_points = np.array(self.selected_points)
        selected_operators = np.array(self.selected_operators_all)

        # Group indices by operator type
        for op in (0, 1, 2, 3):
            idxs = np.where(selected_operators == op)[0]
            if len(idxs) == 0:
                continue

            pts = selected_points[idxs]

            # Evaluate kernel once per group
            if op == 0:  # Interior (PDE)
                G_pred[:, idxs] = self.kernel.h2_eval(X_pred, pts)
            elif op in (1, 2):  # Boundary or Initial displacement
                G_pred[:, idxs] = self.kernel.eval(X_pred, pts)
            elif op == 3:  # Initial velocity
                G_pred[:, idxs] = self.kernel.dt2_eval(X_pred, pts)

        # Multiply by coefficients
        return G_pred @ self.alpha
    
    def predict_Hs(self, X_pred):
        """
        Vectorized prediction of H[u](t,x) at new points X_pred.
        X_pred: (n_pred, dim_total) where dim_total = 1 (time) + 1 (space) = 2 for 1D
        """
        n_pred = X_pred.shape[0]
        n_basis = len(self.selected_points)
        G_pred = np.zeros((n_pred, n_basis))

        # Convert selected points and operators to arrays
        selected_points = np.array(self.selected_points)
        selected_operators = np.array(self.selected_operators_all)

        # Group indices by operator type
        for op in (0, 1, 2, 3):
            idxs = np.where(selected_operators == op)[0]
            if len(idxs) == 0:
                continue

            pts = selected_points[idxs]

            # Evaluate kernel once per group
            if op == 0:  # Interior (PDE)
                G_pred[:, idxs] = self.kernel.hh_eval(X_pred, pts)
            elif op in (1, 2):  # Boundary or Initial displacement
                G_pred[:, idxs] = self.kernel.h1_eval(X_pred, pts)
            elif op == 3:  # Initial velocity
                G_pred[:, idxs] = self.kernel.h1dt2_eval(X_pred, pts)

        # Multiply by coefficients
        return G_pred @ self.alpha
    
    def predict_dt_s(self, X_pred):
        """
        Vectorized prediction of u_t(t,x) at new points X_pred.
        Mirrors predict_s but applies an extra ∂_t at the evaluation point.
        """
        n_pred  = X_pred.shape[0]
        n_basis = len(self.selected_points)
        G_pred  = np.zeros((n_pred, n_basis))

        selected_points    = np.array(self.selected_points)
        selected_operators = np.array(self.selected_operators_all)

        for op in (0, 1, 2, 3):
            idxs = np.where(selected_operators == op)[0]
            if len(idxs) == 0:
                continue
            pts = selected_points[idxs]

            if op == 0:                      # basis H_y K   →  ∂_{t,x} H_y K
                G_pred[:, idxs] = self.kernel.h2dt1_eval(X_pred, pts)
            elif op in (1, 2):               # basis K        →  ∂_{t,x} K
                G_pred[:, idxs] = self.kernel.dt1_eval(X_pred, pts)
            elif op == 3:                    # basis ∂_{t,y}K → ∂_{t,x} ∂_{t,y} K
                G_pred[:, idxs] = self.kernel.dtdt_eval(X_pred, pts)

        return G_pred @ self.alpha



    def orderedGeneralizedInterpolantVec(self, selected_point, operator_selected):
        if operator_selected == 0:  # Interior point (PDE functional)
            interior_part = self.kernel.hh_eval(self.interior_points, selected_point)
            boundary_part = self.kernel.h2_eval(self.boundary_points, selected_point)
            initial_disp_part = self.kernel.h2_eval(self.initial_displacement_points, selected_point)
            initial_vel_part = self.kernel.h2dt1_eval(self.initial_velocity_points, selected_point)
            return np.vstack((interior_part, boundary_part, initial_disp_part, initial_vel_part))
        elif operator_selected == 1 or operator_selected == 2:  # Boundary point or Initial displacement point
            interior_part = self.kernel.h1_eval(self.interior_points, selected_point)
            boundary_part = self.kernel.eval(self.boundary_points, selected_point)
            initial_disp_part = self.kernel.eval(self.initial_displacement_points, selected_point)
            initial_vel_part = self.kernel.dt1_eval(self.initial_velocity_points, selected_point)
            return np.vstack((interior_part, boundary_part, initial_disp_part, initial_vel_part))
        elif operator_selected == 3:  # Initial velocity point
            interior_part = self.kernel.h1dt2_eval(self.interior_points, selected_point)
            boundary_part = self.kernel.dt2_eval(self.boundary_points, selected_point)
            initial_disp_part = self.kernel.dt2_eval(self.initial_displacement_points, selected_point)
            initial_vel_part = self.kernel.dtdt_eval(self.initial_velocity_points, selected_point)
            return np.vstack((interior_part, boundary_part, initial_disp_part, initial_vel_part))
        else:
            raise ValueError("Invalid operator selected.")

            
    def greedyGeneralizedInterpolantVec(self, selected_points, selected_operators, new_point, new_functional):
        selected_points = np.asarray(selected_points)
        selected_operators = np.asarray(selected_operators)

        n = len(selected_points)
        KX = np.empty(n)

        for op in (0, 1, 2, 3):
            mask = selected_operators == op
            if not np.any(mask):
                continue

            pts = selected_points[mask]

            if op == 0:
                if new_functional == 0:
                    KX[mask] = self.kernel.hh_eval(pts, new_point).ravel()
                elif new_functional in (1, 2):
                    KX[mask] = self.kernel.h1_eval(pts, new_point).ravel()
                elif new_functional == 3:
                    KX[mask] = self.kernel.h1dt2_eval(pts, new_point).ravel()

            elif op in (1, 2):
                if new_functional == 0:
                    KX[mask] = self.kernel.h2_eval(pts, new_point).ravel()
                elif new_functional in (1, 2):
                    KX[mask] = self.kernel.eval(pts, new_point).ravel()
                elif new_functional == 3:
                    KX[mask] = self.kernel.dt2_eval(pts, new_point).ravel()

            elif op == 3:
                if new_functional == 0:
                    KX[mask] = self.kernel.h2dt1_eval(pts, new_point).ravel()
                elif new_functional in (1, 2):
                    KX[mask] = self.kernel.dt1_eval(pts, new_point).ravel()
                elif new_functional == 3:
                    KX[mask] = self.kernel.dtdt_eval(pts, new_point).ravel()

        return KX





    def kernel_evaluation(self, point1, operator1, point2, operator2):
        if operator1 == 0:  # Interior point (PDE functional)
            if operator2 == 0:
                return self.kernel.hh_eval(point1, point2)
            elif operator2 == 1 or operator2 == 2:
                return self.kernel.h1_eval(point1, point2)
            elif operator2 == 3:
                return self.kernel.h1dt2_eval(point1, point2)
        elif operator1 == 1 or operator1 == 2:  # Boundary point or Initial displacement point
            if operator2 == 0:
                return self.kernel.h2_eval(point1, point2)
            elif operator2 == 1 or operator2 == 2:
                return self.kernel.eval(point1, point2)
            elif operator2 == 3:
                return self.kernel.dt2_eval(point1, point2)
        elif operator1 == 3:  # Initial velocity point
            if operator2 == 0:
                return self.kernel.h2dt1_eval(point1, point2)
            elif operator2 == 1 or operator2 == 2:
                return self.kernel.dt1_eval(point1, point2)
            elif operator2 == 3:
                return self.kernel.dtdt_eval(point1, point2)
        else:
            raise ValueError("Invalid operator selected.")
    

    def get_output_data(self):
        X_interior = self.interior_points
        N_interior= self.N_interior
        f_interior = self.f_func(X_interior)
        f_interior = np.atleast_2d(f_interior).reshape(N_interior, 1)

        X_boundary = self.boundary_points
        N_boundary = self.N_boundary
        f_boundary = self.g_func(X_boundary)
        f_boundary = np.atleast_2d(f_boundary).reshape(N_boundary, 1)

        X_initial_displacement = self.initial_displacement_points   # shape (N, dim_time+dim_space)
        N_initial_displacement  = self.N_initial_displacement
        f_initial_displacement = self.u0_func(X_initial_displacement)  
        f_initial_displacement = np.atleast_2d(f_initial_displacement).reshape(N_initial_displacement, 1)

        X_initial_velocity = self.initial_velocity_points
        N_initial_velocity = self.N_initial_velocity
        f_initial_velocity = self.v0_func(X_initial_velocity) 
        f_initial_velocity = np.atleast_2d(f_initial_velocity).reshape(N_initial_velocity, 1)
        return f_interior, f_boundary, f_initial_displacement, f_initial_velocity
    
    def split_points(self, X, val_fraction=0.2, seed=0):
        np.random.seed(seed)
        N = X.shape[0]
        perm = np.random.permutation(N)
        n_val = int(val_fraction * N)

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        return X[train_idx], X[val_idx]
    
    def select_kernel_parameters(self, eps_candidates_time, eps_candidates_space, nMax, top_k=3, val_fraction=0.3, seed=0):
        best_eps   = None
        best_score = np.inf
        best_model = None
        results    = {}

        int_train,  int_val  = self.split_points(self.interior_points, val_fraction, seed)
        bnd_train,  bnd_val  = self.split_points(self.boundary_points, val_fraction, seed)
        disp_train, disp_val = self.split_points(self.initial_displacement_points, val_fraction, seed)
        vel_train,  vel_val  = self.split_points(self.initial_velocity_points, val_fraction, seed)

        for eps_t in eps_candidates_time:
            for eps_s in eps_candidates_space:
                print(f"Testing eps_time={eps_t:.3f}, eps_space={eps_s:.3f}")

                k_time  = self.kernel_time_class(ep=float(eps_t), dim=self.kernel_time.dim)
                k_space = self.kernel_space_class(ep=float(eps_s), dim=self.kernel_space.dim)

                model = PDEGreedyWave(
                    k_time, k_space,
                    self.f_func, self.g_func, self.u0_func, self.v0_func,
                    int_train, bnd_train, disp_train, vel_train,
                    c=self.c
                )
                model.fgreedy(nMax)

                # Did the model complete all nMax iterations?
                n_completed = len(model.maxRes)
                reached_full = (n_completed >= nMax)

                val_score   = self._val_score(model, int_val, bnd_val, disp_val, vel_val)
                train_score = model.maxRes[-1]

                status = "OK" if reached_full else f"STOPPED@{n_completed}/{nMax}"
                print(f"  → train residual = {train_score:.3e}, "
                    f"val score = {val_score:.3e}  [{status}]")

                results[(float(eps_t), float(eps_s))] = (
                    val_score if reached_full else np.inf,
                    model.maxRes.copy()
                )

                # Only consider models that reached the full nMax.
                if reached_full and np.isfinite(val_score) and val_score < best_score:
                    best_score = val_score
                    best_eps   = (float(eps_t), float(eps_s))
                    best_model = model

        if best_eps is None:
            print("\nNo model reached the full nMax. Consider lowering nMax or widening the eps grid.")
        else:
            print(f"\nBest eps: {best_eps},  val score: {best_score:.3e}")

        self._plot_parameter_search(results, eps_candidates_time, eps_candidates_space, top_k)
        return best_eps, best_model
    

    def _val_score(self, model, int_val, bnd_val, disp_val, vel_val):
        """
        Validation score for kernel-parameter selection.

        RMSE of the held-out residual, pooled over all four functional classes:
        - Interior PDE residual on the held-out interior subset int_val
          (truth: f, operator applied via predict_Hs).
        - Boundary / IC displacement: held-out subsets (truth: g / u0).
        - IC velocity: held-out subset (truth: v0, uses predict_dt_s).
        No zero-truth guard — a solver that predicts nonzero where truth is
        zero deserves to be penalized.
        Sanity gate rejects runs with blown-up Cholesky coefficients.
        """
        sq_err = 0.0
        n_pts  = 0

        # ---- Interior PDE residual on held-out interior functionals ----
        if int_val is not None and int_val.shape[0] > 0:
            f_true  = self.f_func(int_val).reshape(-1, 1)
            Hs_pred = model.predict_Hs(int_val)
            sq_err += np.sum((f_true - Hs_pred) ** 2)
            n_pts  += int_val.shape[0]

        # ---- Boundary condition (Dirichlet trace == g) ----
        if bnd_val is not None and bnd_val.shape[0] > 0:
            f_true = self.g_func(bnd_val).reshape(-1, 1)
            sq_err += np.sum((f_true - model.predict_s(bnd_val)) ** 2)
            n_pts  += bnd_val.shape[0]

        # ---- Initial displacement (u(0,·) == u0) ----
        if disp_val is not None and disp_val.shape[0] > 0:
            f_true = self.u0_func(disp_val).reshape(-1, 1)
            sq_err += np.sum((f_true - model.predict_s(disp_val)) ** 2)
            n_pts  += disp_val.shape[0]

        # ---- Initial velocity (u_t(0,·) == v0) — uses predict_dt_s ----
        if vel_val is not None and vel_val.shape[0] > 0:
            f_true = self.v0_func(vel_val).reshape(-1, 1)
            sq_err += np.sum((f_true - model.predict_dt_s(vel_val)) ** 2)
            n_pts  += vel_val.shape[0]

        if n_pts == 0:
            return np.inf

        rmse = np.sqrt(sq_err / n_pts)

        # ---- Sanity gate: reject blown-up coefficients ----
        if (not np.isfinite(rmse)) or np.max(np.abs(model.alpha)) > 1e8:
            return np.inf

        return rmse


    def _plot_parameter_search(self, results, eps_time_list, eps_space_list, top_k=3):
        nt = len(eps_time_list)
        ns = len(eps_space_list)

        grid = np.full((nt, ns), np.nan)
        for i, et in enumerate(eps_time_list):
            for j, es in enumerate(eps_space_list):
                score, _ = results.get((float(et), float(es)), (np.nan, None))
                grid[i, j] = np.log10(score) if (score is not None and np.isfinite(score) and score > 0) else np.nan

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im = axes[0].imshow(
            grid, origin='lower', aspect='auto',
            extent=[eps_space_list[0], eps_space_list[-1],
                    eps_time_list[0],  eps_time_list[-1]],
            cmap='viridis_r'
        )
        plt.colorbar(im, ax=axes[0], label='log10(val score)')
        axes[0].set_xlabel('eps_space')
        axes[0].set_ylabel('eps_time')
        axes[0].set_title('Validation Score Heatmap')

        # Only mark best among finite scores
        finite_results = {k: v for k, v in results.items() if np.isfinite(v[0])}
        if finite_results:
            best_key = min(finite_results, key=lambda k: finite_results[k][0])
            axes[0].plot(best_key[1], best_key[0], 'r*', markersize=14, label='best')
            axes[0].legend()

        # Top-k convergence curves — skip nan entries
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if np.isfinite(v[0])],
            key=lambda x: x[1][0]
        )
        for (et, es), (score, maxRes) in sorted_results[:top_k]:
            axes[1].semilogy(
                np.arange(1, len(maxRes) + 1), maxRes,
                label=f'et={et:.1f}, es={es:.1f}  (val={score:.1e})',
                linewidth=1.8
            )

        axes[1].set_xlabel('Iteration n')
        axes[1].set_ylabel('Train Max Residual')
        axes[1].set_title(f'Convergence (train): Top {top_k} by val score')
        axes[1].legend(fontsize=8, frameon=True)
        axes[1].grid(True, which='both', linestyle=':', linewidth=0.8)

        plt.tight_layout()
        plt.show()

    def plot_selected_points_and_residuals(self, makeAnimation=False):

        # Plot number of selected functionals over iterations
        plt.figure(figsize=(10, 6))

        cumulative_int = np.cumsum([1 if op == 0 else 0 for op in self.selected_operators_all])
        cumulative_bnd = np.cumsum([1 if op == 1 else 0 for op in self.selected_operators_all])
        cumulative_init_disp = np.cumsum([1 if op == 2 else 0 for op in self.selected_operators_all])
        cumulative_init_vel = np.cumsum([1 if op == 3 else 0 for op in self.selected_operators_all])

        iters = np.arange(1, len(self.selected_operators_all) + 1)

        plt.plot(iters, cumulative_int, label='Interior (PDE)', linewidth=1.5)
        plt.plot(iters, cumulative_bnd, label='Boundary', linewidth=1.5)
        plt.plot(iters, cumulative_init_disp, label='Initial Displacement', linewidth=1.5)
        plt.plot(iters, cumulative_init_vel, label='Initial Velocity', linewidth=1.5)

        plt.xlabel('Iteration')
        plt.ylabel('Number of Selected Functionals')
        plt.title('Cumulative Functional Selection over Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Plot selected collocation points in time-space
        if not makeAnimation:

            if self.kernel_space.dim == 1:
                times_int = np.array([p[0] for p in self.selected_points_int])
                spaces_int = np.array([p[1] for p in self.selected_points_int])

                times_bnd = np.array([p[0] for p in self.selected_points_bnd])
                spaces_bnd = np.array([p[1] for p in self.selected_points_bnd])

                times_init_disp = np.array([p[0] for p in self.selected_points_init_disp])
                spaces_init_disp = np.array([p[1] for p in self.selected_points_init_disp])

                times_init_vel = np.array([p[0] for p in self.selected_points_init_vel])
                spaces_init_vel = np.array([p[1] for p in self.selected_points_init_vel])

                plt.figure(figsize=(8, 6))
                plt.scatter(spaces_int, times_int, c='blue', label='Interior Points (PDE)', marker='o')
                plt.scatter(spaces_bnd, times_bnd, c='red', label='Boundary Points', marker='s')
                plt.scatter(spaces_init_disp, times_init_disp, c='green', label='Initial Displacement Points', marker='^')
                plt.scatter(spaces_init_vel, times_init_vel, c='orange', label='Initial Velocity Points', marker='x')
                plt.xlabel('Space')
                plt.ylabel('Time')
                plt.title('Selected Collocation Points by f-Greedy')
                plt.legend()
                plt.grid()
                plt.show()
            else:
                # Each point is now (t, x, y)
                times_int = np.array([p[0] for p in self.selected_points_int])
                x_int = np.array([p[1] for p in self.selected_points_int])
                y_int = np.array([p[2] for p in self.selected_points_int])

                times_bnd = np.array([p[0] for p in self.selected_points_bnd])
                x_bnd = np.array([p[1] for p in self.selected_points_bnd])
                y_bnd = np.array([p[2] for p in self.selected_points_bnd])

                times_init_disp = np.array([p[0] for p in self.selected_points_init_disp])
                x_init_disp = np.array([p[1] for p in self.selected_points_init_disp])
                y_init_disp = np.array([p[2] for p in self.selected_points_init_disp])

                times_init_vel = np.array([p[0] for p in self.selected_points_init_vel])
                x_init_vel = np.array([p[1] for p in self.selected_points_init_vel])
                y_init_vel = np.array([p[2] for p in self.selected_points_init_vel])

                # --- 3D Scatter Plot ---
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(x_int, y_int, times_int, c='blue', label='Interior Points (PDE)', marker='o')
                ax.scatter(x_bnd, y_bnd, times_bnd, c='red', label='Boundary Points', marker='s')
                ax.scatter(x_init_disp, y_init_disp, times_init_disp, c='green', label='Initial Displacement Points', marker='^')
                ax.scatter(x_init_vel, y_init_vel, times_init_vel, c='orange', label='Initial Velocity Points', marker='x')

                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('Time t')
                ax.set_title('Selected Collocation Points by f-Greedy (2D space)')
                ax.legend()
                plt.show()
        else:  # Create animation by selection order
            if self.kernel_space.dim == 1:
                # Map operator types to colors/markers
                op_color = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}
                op_marker = {0: 'o', 1: 's', 2: '^', 3: 'x'}
                op_label = {0: 'Interior Points (PDE)',
                            1: 'Boundary Points',
                            2: 'Initial Displacement Points',
                            3: 'Initial Velocity Points'}

                fig, ax = plt.subplots(figsize=(8, 6))

                ax.set_xlabel('Space')
                ax.set_ylabel('Time')
                ax.set_title('Selected Collocation Points by f-Greedy')
                ax.grid(True)

                # Axis limits
                all_points = np.array(self.selected_points)
                ax.set_xlim(np.min(all_points[:, 1]), np.max(all_points[:, 1]))
                ax.set_ylim(np.min(all_points[:, 0]), np.max(all_points[:, 0]))

                # Create empty scatter plots for each type
                scatters = {}
                for op in (0, 1, 2, 3):
                    scatters[op] = ax.scatter([], [], c=op_color[op], marker=op_marker[op], label=op_label[op])

                ax.legend()

                def update(frame):
                    # Add points in selection order up to current frame
                    for op in (0, 1, 2, 3):
                        points = [p for j, p in enumerate(self.selected_points[:frame+1])
                                if self.selected_operators_all[j] == op]
                        if points:
                            pts = np.array(points)
                            scatters[op].set_offsets(pts[:, [1, 0]])  # space vs time
                    return tuple(scatters.values())

                anim = FuncAnimation(
                    fig,
                    update,
                    frames=len(self.selected_points),
                    interval=200,
                    blit=True
                )

                plt.show()
            else:

                # Map operator types to colors/markers
                op_color = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}
                op_marker = {0: 'o', 1: 's', 2: '^', 3: 'x'}
                op_label = {0: 'Interior Points (PDE)',
                            1: 'Boundary Points',
                            2: 'Initial Displacement Points',
                            3: 'Initial Velocity Points'}

                # 3D figure
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('Time t')
                ax.set_title('Selected Collocation Points by f-Greedy (2D space)')

                # Axis limits
                all_points = np.array(self.selected_points)  # shape (N, 3): [t, x, y]
                ax.set_xlim(np.min(all_points[:, 1]), np.max(all_points[:, 1]))
                ax.set_ylim(np.min(all_points[:, 2]), np.max(all_points[:, 2]))
                ax.set_zlim(np.min(all_points[:, 0]), np.max(all_points[:, 0]))

                # Create empty scatter plots for each type
                scatters = {}
                for op in (0, 1, 2, 3):
                    scatters[op] = ax.scatter([], [], [], c=op_color[op], marker=op_marker[op], label=op_label[op])

                ax.legend()

                # Animation update function
                def update(frame):
                    for op in (0, 1, 2, 3):
                        # Select points of type `op` up to current frame
                        points = [p for j, p in enumerate(self.selected_points[:frame+1])
                                if self.selected_operators_all[j] == op]
                        if points:
                            pts = np.array(points)
                            scatters[op]._offsets3d = (pts[:, 1], pts[:, 2], pts[:, 0])  # x, y, t
                    return tuple(scatters.values())

                # Create animation
                anim = FuncAnimation(
                    fig,
                    update,
                    frames=len(self.selected_points),
                    interval=100,
                    blit=False  # blit=True doesn't work with 3D scatters
                )

                plt.show()

        # Plot total and functional residuals (clean, publication-quality)
        plt.figure(figsize=(8, 6))

        iters = np.arange(1, len(self.maxRes) + 1)

        plt.semilogy(iters, self.maxResInt,
                    linewidth=2,
                    alpha=0.9,
                    label='Interior (PDE) Residual')

        plt.semilogy(iters, self.maxResBnd,
                    linestyle='-.', linewidth=2,
                    alpha=0.9,
                    label='Boundary Residual')

        plt.semilogy(iters, self.maxResInitDisp,
                    linestyle=':', linewidth=2.2,
                    alpha=0.9,
                    label='Initial Displacement Residual')

        plt.semilogy(iters, self.maxResInitVel,
                    linestyle=(0, (5, 2)), linewidth=2,
                    alpha=0.9,
                    label='Initial Velocity Residual')

        plt.xlabel('Expansion size n')
        plt.ylabel(r'Maximum Residual $\max_{\lambda \in \Lambda_N} |\lambda(u - s^{(n)})|$')
        plt.title('Residual Convergence in f-Greedy Algorithm')

        plt.legend(frameon=True)
        plt.grid(True, which='both', linestyle=':', linewidth=0.8)

        plt.tight_layout()
        plt.show()


