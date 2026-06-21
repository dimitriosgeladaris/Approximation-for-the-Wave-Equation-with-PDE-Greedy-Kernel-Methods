import kernels
import numpy as np


r"""
Represents a PDE problem of the form
    u_tt - c^2 \Delta u(x) = f(x),    x ∈ Ω
    B u(x) = g(x),    x ∈ ∂Ω
    with initial conditions u(0,x) = u0(x), u_t(0,x) = 0,
solved using symmetric kernel collocation and inverting full system matrix.
"""

class WaveCollocation:
    def __init__(self, k_time: kernels.Kernel, k_space: kernels.Kernel,
                f_func, g_func, u0_func, v0_func,
                 interior_points, boundary_points, initial_displacement_points, initial_velocity_points, c=1.0):
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
        
        self.A = None  # Collocation matrix
        self.b = None  # Right-hand side vector
        self.coeffs = None  # Coefficients of the kernel expansion

    def predict_s(self, X_pred):
        """
        Predict the solution at given space-time points X_t.
        X_t: shape (M, dim_time + dim_space)
        Returns: predicted values at X_t, shape (M, 1)
        """
        if self.coeffs is None:
            raise ValueError("Model is not trained yet. Call assemble_system() first.")
        
        else:
            prediction = self.kernel.h2_eval(X_pred, self.interior_points) @ self.coeffs[:self.N_interior] 
            prediction += self.kernel.eval(X_pred, self.boundary_points) @ self.coeffs[self.N_interior:self.N_interior + self.N_boundary]
            prediction += self.kernel.eval(X_pred, self.initial_displacement_points) @ self.coeffs[self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement]
            prediction += self.kernel.dt2_eval(X_pred, self.initial_velocity_points) @ self.coeffs[self.N_interior + self.N_boundary + self.N_initial_displacement:]
            return prediction
    
    def predict_Ls(self, X_pred):
        """
        Predict the application of the differential operator (d^2/dt^2 - d^2/dx^2) to the solution at given space-time points X_t.
        X_t: shape (M, dim_time + dim_space)
        Returns: predicted values at X_t, shape (M, 1)
        """
        if self.coeffs is None:
            raise ValueError("Model is not trained yet. Call assemble_system() first.")
        
        else:
            prediction = self.kernel.hh_eval(X_pred, self.interior_points) @ self.coeffs[:self.N_interior] 
            prediction += self.kernel.h1_eval(X_pred, self.boundary_points) @ self.coeffs[self.N_interior:self.N_interior + self.N_boundary]
            prediction += self.kernel.h1_eval(X_pred, self.initial_displacement_points) @ self.coeffs[self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement]
            prediction += self.kernel.h1dt2_eval(X_pred, self.initial_velocity_points) @ self.coeffs[self.N_interior + self.N_boundary + self.N_initial_displacement:]
            return prediction

    def fit(self):
        # Build and solve the collocation system, first upper part of matrix including diagonal blocks
        # Build diagonal blocks
        A_interior, b_interior = self.build_interior_system()
        A_boundary, b_boundary = self.build_boundary_system()
        A_initial_displacement, b_initial_displacement = self.build_initial_displacement_system()
        A_initial_velocity, b_initial_velocity = self.build_initial_velocity_system()

        # Build mixed blocks
        A_inner_boundary = self.build_inner_boundary_system()
        A_inner_initial_displacement = self.build_inner_initial_displacement_system()
        A_inner_initial_velocity = self.build_inner_initial_velocity_system()
        A_boundary_initial_displacement = self.build_boundary_initial_displacement_system()
        A_boundary_initial_velocity = self.build_boundary_initial_velocity_system()
        A_initial_displacement_initial_velocity = self.build_initial_displacement_initial_velocity_system()

        # Assemble full matrix A and right-hand side b
        N = self.N
        A = np.zeros((N, N))
        b = np.zeros((N, 1))

        # Diagonal blocks
        A[0:self.N_interior, 0:self.N_interior] = A_interior
        b[0:self.N_interior, 0:1] = b_interior
        A[self.N_interior:self.N_interior + self.N_boundary, self.N_interior:self.N_interior + self.N_boundary] = A_boundary
        b[self.N_interior:self.N_interior + self.N_boundary, 0:1] = b_boundary
        A[self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement, self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement] = A_initial_displacement
        b[self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement, 0:1] = b_initial_displacement
        A[self.N_interior + self.N_boundary + self.N_initial_displacement:, self.N_interior + self.N_boundary + self.N_initial_displacement:] = A_initial_velocity
        b[self.N_interior + self.N_boundary + self.N_initial_displacement:, 0:1] = b_initial_velocity

        # Off-diagonal blocks
        A[0:self.N_interior, self.N_interior:self.N_interior + self.N_boundary] = A_inner_boundary
        A[0:self.N_interior, self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement] = A_inner_initial_displacement
        A[0:self.N_interior, self.N_interior + self.N_boundary + self.N_initial_displacement:] = A_inner_initial_velocity
        A[self.N_interior:self.N_interior + self.N_boundary, self.N_interior + self.N_boundary + self.N_initial_displacement:] = A_boundary_initial_velocity
        A[self.N_interior:self.N_interior + self.N_boundary, self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement] = A_boundary_initial_displacement
        A[self.N_interior + self.N_boundary:self.N_interior + self.N_boundary + self.N_initial_displacement, self.N_interior + self.N_boundary + self.N_initial_displacement:] = A_initial_displacement_initial_velocity    

        # Since A is symmetric, fill in the lower triangular part
        A = np.triu(A) + np.triu(A, 1).T
        
        self.A = A
        self.b = b

        # Solve the linear system
        self.coeffs = np.linalg.solve(self.A, self.b)
        # self.coeffs = np.linalg.lstsq(self.A, self.b, rcond=None)[0]

        return self

    def build_interior_system(self):
        # Block for (u_tt - Δu)(x_i) = f(x_i), x_i ∈ interior points
        X = self.interior_points
        N = self.N_interior

        A_interior = self.kernel.hh_eval(X, X)
        b_interior = self.f_func(X)
        b_interior = np.atleast_2d(b_interior).reshape(N, 1)
        return A_interior, b_interior
    
    def build_boundary_system(self):
        # Block for u(x_i) = g(x_i), x_i ∈ boundary points
        X = self.boundary_points
        N = self.N_boundary

        A_boundary = self.kernel.eval(X, X)
        b_boundary = self.g_func(X)
        b_boundary = np.atleast_2d(b_boundary).reshape(N, 1)
        return A_boundary, b_boundary
    
    def build_initial_displacement_system(self):
        X = self.initial_displacement_points   # shape (N, dim_time+dim_space)
        N = self.N_initial_displacement

        A_initial_displacement = self.kernel.eval(X, X)
        b_initial_displacement = self.u0_func(X)  
        b_initial_displacement = np.atleast_2d(b_initial_displacement).reshape(N, 1)
        return A_initial_displacement, b_initial_displacement

    
    def build_initial_velocity_system(self):
        # Block for u_t(0,x_i) = v0(x_i), x_i ∈ initial velocity points
        X = self.initial_velocity_points
        N = self.N_initial_velocity

        A_initial_velocity = self.kernel.dtdt_eval(X, X)
        b_initial_velocity = self.v0_func(X) 
        b_initial_velocity = np.atleast_2d(b_initial_velocity).reshape(N, 1)
        return A_initial_velocity, b_initial_velocity
    
    # Now build mixed blocks
    def build_inner_boundary_system(self):
        # Block for h^[1]^B^[2] k((t,x), (t',x'))
        X_interior = self.interior_points
        X_boundary = self.boundary_points
        A_inner_boundary = self.kernel.h1_eval(X_interior, X_boundary)
        return A_inner_boundary
    
    def build_inner_initial_displacement_system(self):
        X_interior = self.interior_points
        X_initial_displacement = self.initial_displacement_points
        A_inner_initial_displacement = self.kernel.h1_eval(X_interior, X_initial_displacement)
        return A_inner_initial_displacement
    
    def build_inner_initial_velocity_system(self):
        X_interior = self.interior_points
        X_initial_velocity = self.initial_velocity_points
        A_inner_initial_velocity = self.kernel.h1dt2_eval(X_interior, X_initial_velocity)
        return A_inner_initial_velocity

    def build_boundary_initial_displacement_system(self):
        X_boundary = self.boundary_points
        X_initial_displacement = self.initial_displacement_points
        A_boundary_initial_displacement = self.kernel.eval(X_boundary, X_initial_displacement)
        return A_boundary_initial_displacement
    
    def build_boundary_initial_velocity_system(self):
        X_boundary = self.boundary_points
        X_initial_velocity = self.initial_velocity_points
        A_boundary_initial_velocity = self.kernel.dt2_eval(X_boundary, X_initial_velocity)
        return A_boundary_initial_velocity
    
    def build_initial_displacement_initial_velocity_system(self):
        X_initial_displacement = self.initial_displacement_points
        X_initial_velocity = self.initial_velocity_points
        A_initial_displacement_initial_velocity = self.kernel.dt2_eval(X_initial_displacement, X_initial_velocity)
        return A_initial_displacement_initial_velocity