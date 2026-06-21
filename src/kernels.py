# Kernel class edited and extended for PDE applications, original by Gabriele Santin.
# d1_eval: -Laplace_x k(*,y)|_{x} implemented for -Laplace -> signs in multiplication kernel flips.
# Dimitrios Geladaris 12/2025

from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix
import numpy as np


# Abstract kernel
class Kernel(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, x, y):
        pass

    def eval_prod(self, x, y, v, batch_size=100):
        N = x.shape[0]
        num_batches = int(np.ceil(N / batch_size))
        mat_vec_prod = np.zeros((N, 1))
        for idx in range(num_batches):
            idx_begin = idx * batch_size
            idx_end = (idx + 1) * batch_size
            A = self.eval(x[idx_begin:idx_end, :], y)
            mat_vec_prod[idx_begin:idx_end] = A @ v
        return mat_vec_prod

    @abstractmethod
    def diagonal(self, X):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

class ProductKernel(Kernel):
    """
    Product kernel: k((t,x), (t',x')) = k1(t,t') * k2(x,x').
    Wave/D'Alembert operator is defined as h^(t,x) := d^2/dt^2 - c^2*Laplace_x.
    """

    def __init__(self, k1: Kernel, k2: Kernel, c=1.0):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.c = c # wave speed
        self.dim_time = k1.dim
        self.dim_space = k2.dim
        self.name = f"Product({k1.name}, {k2.name})"

    def _split(self, X_t):
        X_t = np.atleast_2d(X_t)
        t = X_t[:, :self.dim_time]
        x = X_t[:, self.dim_time:]
        return t, x

    def eval(self, x_t, y_t):
        # Evaluation of k((t,x), (t',x')) = k1(t,t') * k2(x,x') also needed for diriclet BCs as B^[1] = B^[2] = I
        t1, x1 = self._split(x_t)
        t2, x2 = self._split(y_t)
        return self.k1.eval(t1, t2) * self.k2.eval(x1, x2)

    def diagonal(self, X_t):
        t, x = self._split(X_t)
        return self.k1.diagonal(t) * self.k2.diagonal(x)

    def set_params(self, par):
        p1, p2 = par
        self.k1.set_params(p1)
        self.k2.set_params(p2)

    def __str__(self):
        return f"{self.k1} * {self.k2}"
    
    def hh_eval(self, x_t, y_t):
        # # Evaluation of h^[1]h^[2]k(x,y) := k(x,y) * d^4/dt^4 k(t,t') + k(t,t') * (-Laplace_x)(-Laplace_x') k(x,x') - 2 * d^2/dt^2 k(t,t') * (Laplace) k(x,x')
        t1, x1 = self._split(x_t)
        t2, x2 = self._split(y_t)
        return self.k2.eval(x1, x2) * self.k1.dd_eval(t1, t2) + self.c**4 * self.k1.eval(t1, t2) * self.k2.dd_eval(x1, x2) \
               + 2 * self.c**2 * self.k1.d2t1_eval(t1, t2) * self.k2.d1_eval(x1, x2)
    
    def h1dt2_eval(self, x_t, y_t):
        # Evaluation of (d^2/dt^2 - Laplace_x)^[1] (d/dt)^[2] k((t,x), (t',x'))
        t1, x1 = self._split(x_t)
        t2, x2 = self._split(y_t)
        return -self.k2.eval(x1, x2) * self.k1.dddt1_eval(t1, t2) - self.k1.dt1_eval(t1, t2) * self.c**2 * self.k2.d1_eval(x1, x2)
    
    def h2dt1_eval(self, x_t, y_t):
        # Evaluation of (d^2/dt^2 - Laplace_x)^[2] (d/dt)^[1] k((t,x), (t',x'))
        return -self.h1dt2_eval(x_t, y_t)
    
    def h1_eval(self, x_t, y_t):
        # Evaluation of (d^2/dt^2 - c^2*Laplace_x)^[1] k(t,t') * k(x,y), i.e. h applied to the second variable
        t1, x1 = self._split(x_t)
        t2, x2 = self._split(y_t)
        return self.k2.eval(x1, x2) * self.k1.d2t1_eval(t1, t2) + self.k1.eval(t1, t2) * self.c**2 * self.k2.d2_eval(x1, x2)
    
    
    def h2_eval(self, x_t, y_t):
        # Evaluation of (d^2/dt^2 - Laplace_x)^[2] k(t,t') * k(x,y)
        return self.h1_eval(x_t, y_t)  # Since both operators are radial, the formula is the same
    
    
    def dtdt_eval(self, x_t, y_t):
        # Evaluation of (d/dt)^[1] (d/dt)^[2] k((t,x), (t',x'))
        t1, x1 = self._split(x_t)
        t2, x2 = self._split(y_t)
        return - self.k2.eval(x1, x2) * self.k1.d2t1_eval(t1, t2)
    
    def dt1_eval(self, x_t, y_t):
        # Evaluation of d/dt^[1] k((t,x), (t',x'))
        t1, x1 = self._split(x_t)
        t2, x2 = self._split(y_t)
        return self.k2.eval(x1, x2) * self.k1.dt1_eval(t1, t2)
    
    def dt2_eval(self, x_t, y_t):
        # Evaluation of d/dt' k((t,x), (t',x'))
        t1, x1 = self._split(x_t)
        t2, x2 = self._split(y_t)
        return self.k2.eval(x1, x2) * self.k1.dt2_eval(t1, t2)


# Abstract RBF for single / non-product kernels
class RBF(Kernel):
    @abstractmethod
    def __init__(self):
        super(RBF, self).__init__()

    def eval(self, x, y):
        return self.rbf(self.ep, distance_matrix(np.atleast_2d(x), np.atleast_2d(y)))

    def diagonal(self, X):
        return np.ones(X.shape[0]) * self.rbf(self.ep, 0)

    def __str__(self):
        return self.name + ' [gamma = %2.2e]' % self.ep

    def set_params(self, par):
        self.ep = par

    def dd_diagonal(self, X):
        # Evaluation of (-Laplace)^x (-Laplace)^y k(x,y) for x=y: Likely some constant

        return np.ones(X.shape[0]) * self.dd_eval(np.zeros((1, 1)), np.zeros((1, 1))).item()

    def dd_eval(self, x, y):
        # Evaluation of (-Laplace)^x (-Laplace)^y k(x,y), i.e. w.r.t both variables fourth order terms double laplacian for RBFs

        array_dists = distance_matrix(np.atleast_2d(x), np.atleast_2d(y))

        return self.double_laplacian(self.ep, array_dists) 


    def d2_eval(self, x, y):
        # Evaluation of (-Laplace^z) k(*,z)|_{x,y}, i.e. w.r.t the second variable, second derivative

        array_dists = distance_matrix(np.atleast_2d(x), np.atleast_2d(y))

        array_result = -(self.rbf2(self.ep, array_dists) + (self.dim - 1) * self.rbf1_divided_by_r(self.ep, array_dists))

        return array_result

    def d1_eval(self, x, y):
        # Evaluation of (-Laplace)^[1] k(*,y)|_{x}
        # Since the LaPlace is radial, the formula is the same as d2_eval
        return self.d2_eval(x, y)
    
    def d2t1_eval(self, t, s):
        # Evaluation of d^2/dt1^2 k(t,s), i.e. w.r.t the first variable
        if self.dim != 1:
            raise NotImplementedError("Time derivative only implemented for 1D kernels.")

        array_dists = distance_matrix(np.atleast_2d(t), np.atleast_2d(s))

        return self.rbf2(self.ep, array_dists)
    
    def d2t2_eval(self, t, s):
        # Evaluation of d^2/ds^2 k(t,s), i.e. w.r.t the second variable
        return self.d2t1_eval(t, s) # since symmetric in t and s

    def dt1_eval(self, t, s):
        # Evaluation of d/dt1 k(t,s), i.e. w.r.t the first variable
        if self.dim != 1:
            raise NotImplementedError("Time derivative only implemented for 1D kernels.")
        array_dists = distance_matrix(np.atleast_2d(t), np.atleast_2d(s))
        return np.sign( (t-s.T)) * self.rbf1(self.ep, array_dists)
    
    def dt2_eval(self, t, s):
        # Evaluation of d/ds k(t,s), i.e. w.r.t the second variable
        return -self.dt1_eval(t, s)
    
    def dddt1_eval(self, t, s):
        # Evaluation of d^3/dt^3 k(t,s) w.r.t the first variable
        if self.dim != 1:
            raise NotImplementedError("Time derivative only implemented for 1D kernels.")

        array_dists = distance_matrix(np.atleast_2d(t), np.atleast_2d(s))

        return self.rbf3(self.ep, array_dists) *np.sign( (t-s.T))
    
    def dddt2_eval(self, t, s):
        # Evaluation of d^3/ds^3 k(t,s) w.r.t the second variable
        return -self.dddt1_eval(t, s)

# Implementation of concrete RBFs
class Gaussian(RBF):
    # Gaussian kernel phi(r) = exp(-(eps*r)^2)

    def __init__(self, dim, ep=1):
        super().__init__()
        self.dim = dim
        self.ep = ep
        self.name = 'gauss'

        # Common factor
        # a = eps^2
        # phi(r) = exp(-a r^2)

        self.rbf = lambda ep, r: np.exp(-(ep * r)**2)

        self.rbf1 = lambda ep, r: (
            -2 * ep**2 * r * np.exp(-(ep * r)**2)
        )

        self.rbf1_divided_by_r = lambda ep, r: (
            -2 * ep**2 * np.exp(-(ep * r)**2)
        )

        self.rbf2 = lambda ep, r: (
            (4 * ep**4 * r**2 - 2 * ep**2)
            * np.exp(-(ep * r)**2)
        )

        self.rbf3 = lambda ep, r: (
            (-8 * ep**6 * r**3 + 12 * ep**4 * r)
            * np.exp(-(ep * r)**2)
        )

        # Double Laplacian Δ²φ
        self.double_laplacian = lambda ep, r: (
            np.exp(-(ep * r)**2)
            * (
                16 * ep**8 * r**4
                - 16 * ep**6 * (self.dim + 2) * r**2
                + 4 * ep**4 * (self.dim**2 + 2 * self.dim)
            )
        )


class IMQ(RBF):
    # Inverse multiquadric kernel
    # phi(r) = (1 + (eps*r)^2)^(-1/2)

    def __init__(self, dim, ep=1):
        super().__init__()
        self.dim = dim
        self.ep = ep
        self.name = 'imq'

        self.rbf = lambda ep, r: (
            (1 + (ep * r)**2)**(-1/2)
        )

        self.rbf1 = lambda ep, r: (
            -ep**2 * r * (1 + (ep * r)**2)**(-3/2)
        )

        self.rbf1_divided_by_r = lambda ep, r: (
            -ep**2 * (1 + (ep * r)**2)**(-3/2)
        )

        self.rbf2 = lambda ep, r: (
            ep**2
            * (2 * ep**2 * r**2 - 1)
            * (1 + (ep * r)**2)**(-5/2)
        )

        self.rbf3 = lambda ep, r: (
            3 * ep**4 * r
            * (3 - 2 * ep**2 * r**2)
            * (1 + (ep * r)**2)**(-7/2)
        )

        self.double_laplacian = lambda ep, r: (
            3
            * ep**4
            * (1 + (ep * r)**2)**(-9/2)
            * (
                self.dim * (self.dim + 2)
                + (2 * self.dim**2 - 6 * self.dim - 20)
                  * ep**2 * r**2
                + (self.dim - 3) * (self.dim - 5)
                  * (ep**2 * r**2)**2
            )
        )

class Matern_7_2(RBF):
    # Matérn-7/2 kernel: C^6 globally (6 continuous derivatives).
    # phi(r) = (1 + ε r + 2(ε r)²/5 + (ε r)³/15) * exp(-ε r)

    def __init__(self, dim, ep=1):
        super().__init__()
        self.dim = dim
        self.ep = ep
        self.name = 'matern_7_2'

        self.rbf = lambda ep, r: (
            (1 + ep*r + (2/5)*(ep*r)**2 + (1/15)*(ep*r)**3)
            * np.exp(-ep*r)
        )

        self.rbf1 = lambda ep, r: (
            -ep**2 * r * (3 + 3*ep*r + (ep*r)**2)
            / 15.0
            * np.exp(-ep*r)
        )

        # Finite at r=0; limit = -ε²/5
        self.rbf1_divided_by_r = lambda ep, r: (
            -ep**2 * (3 + 3*ep*r + (ep*r)**2)
            / 15.0
            * np.exp(-ep*r)
        )

        self.rbf2 = lambda ep, r: (
            ep**2 * ((ep*r)**3 - 3*ep*r - 3)
            / 15.0
            * np.exp(-ep*r)
        )

        self.rbf3 = lambda ep, r: (
            ep**4 * r * (-((ep*r)**2) + 3*ep*r + 3)
            / 15.0
            * np.exp(-ep*r)
        )

        self.double_laplacian = lambda ep, r: (
            ep**4
            * np.exp(-ep*r)
            / 15.0
            * (
                self.dim * (self.dim + 2)
                + self.dim * (self.dim + 2) * (ep*r)
                - 2 * (self.dim + 2) * (ep*r)**2
                + (ep*r)**3
            )
        )

class Wendland_C6(RBF):
    # Wendland phi_{3,3}: C^6 smoothness

    def __init__(self, dim, ep=1):
        super().__init__()
        self.dim = dim
        self.ep = ep
        self.name = 'wendland_c6'
        d = dim

        self.rbf = lambda ep, r: (
            np.maximum(1 - ep*r, 0)**8
            * (32*(ep*r)**3 + 25*(ep*r)**2 + 8*(ep*r) + 1))

        self.rbf1 = lambda ep, r: ep * (
            -22.0 * (ep*r) * (16*(ep*r)**2 + 7*(ep*r) + 1)
            * np.maximum(1 - ep*r, 0)**7)

        # phi'(r)/r = eps^2 psi'(s)/s   (finite at r=0, limit -22 * eps^2)
        self.rbf1_divided_by_r = lambda ep, r: ep**2 * (
            -22.0 * (16*(ep*r)**2 + 7*(ep*r) + 1) * np.maximum(1 - ep*r, 0)**7)

        self.rbf2 = lambda ep, r: ep**2 * (
            22.0 * (160*(ep*r)**3 + 15*(ep*r)**2 - 6*(ep*r) - 1)
            * np.maximum(1 - ep*r, 0)**6)

        # psi'''(s) = -1584 s (1-s)^5 (20 s^2 - 5 s - 1)
        self.rbf3 = lambda ep, r: ep**3 * (
            -1584.0 * (ep*r) * (20*(ep*r)**2 - 5*(ep*r) - 1)
            * np.maximum(1 - ep*r, 0)**5)

        self.double_laplacian = lambda ep, r: ep**4 * (
            (528*d**2 + 1056*d)
            + (ep*r)**2 * (-11088*d**2 - 66528*d - 88704)
            + (ep*r)**3 * (36960*d**2 + 295680*d + 554400)
            + (ep*r)**4 * (-55440*d**2 - 554400*d - 1330560)
            + (ep*r)**5 * (44352*d**2 + 532224*d + 1552320)
            + (ep*r)**6 * (-18480*d**2 - 258720*d - 887040)
            + (ep*r)**7 * (3168*d**2 + 50688*d + 199584)
        ) * (ep*r < 1.0)
