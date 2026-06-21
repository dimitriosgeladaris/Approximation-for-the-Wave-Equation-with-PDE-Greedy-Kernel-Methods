import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye as sp_eye

class LinearWave1D:
    """
    Semi-discrete 1D wave equation on (0,L) with homogeneous Dirichlet BC:
        u(0,t) = u(L,t) = 0

    State: y = [q; p]
        q ∈ R^N  : displacement at interior nodes
        p = M q̇ : momentum

    PDE:
        u_tt = c^2 u_xx

    Spatial discretization (FD / P1 FEM equivalent):
        M = I        (lumped mass)
        K = (c^2 / h^2) * tridiag(-1, 2, -1)

    Hamiltonian:
        H(q,p) = 1/2 pᵀ M⁻¹ p + 1/2 qᵀ K q

    ODE:
        q̇ = M⁻¹ p
        ṗ = -K q
    """

    def __init__(self, L=1.0, c=1.0, N=200, datastring="wholeSpace",
                 mass_lumping=True, sparse=True, q_scale=0.5, p_scale=0.5):

        self.L = float(L)
        self.c = float(c)
        self.N = int(N)
        self.datastring = datastring
        self.mass_lumping = bool(mass_lumping)
        self.sparse = bool(sparse)
        self.q_scale = float(q_scale)
        self.p_scale = float(p_scale)

        # grid spacing (N interior nodes)
        self.h = self.L / (self.N + 1)

        # ---- stiffness matrix K (Dirichlet BCs) ----
        main = np.full(self.N, 2.0)
        off  = np.full(self.N - 1, -1.0)

        K = diags([off, main, off], offsets=[-1, 0, 1], format='csr')
        K = (self.c ** 2 / self.h ** 2) * K
        self.K = K if self.sparse else K.toarray()

        # ---- mass matrix M and inverse (lumped) ----
        self.M = sp_eye(self.N, format='csr')
        self.Minv = sp_eye(self.N, format='csr')

        # ---- canonical symplectic matrix J ----
        n = self.N
        ty = np.float32
        J1 = np.concatenate([np.zeros((n, n), dtype=ty), -np.eye(n, dtype=ty)], axis=0)
        J2 = np.concatenate([np.eye(n, dtype=ty),  np.zeros((n, n), dtype=ty)], axis=0)
        self.J = np.concatenate([J1, J2], axis=1)

    # ------------------------------------------------------------------
    # initial conditions
    # ------------------------------------------------------------------
    # def getInitialStates(self, nGrid, endT, dt, dT=0.1):
    #     """
    #     Generate sine-wave initial states:
    #         q0(x) = sin(b π x / L)
    #         p0(x) = sin(e π x / L)

    #     Returns array of shape (2N, nSamples)
    #     """

    #     nSamples = int(nGrid)
    #     rng = np.random.default_rng(42)

    #     Bq = min(2, self.N)
    #     Bp = min(2, self.N)

    #     total_pairs = Bq * Bp
    #     if nSamples > total_pairs:
    #         raise ValueError(
    #             f"Requested {nSamples} samples but only {total_pairs} unique mode pairs available."
    #         )

    #     all_pairs = np.array(
    #         [(b, e) for b in range(1, Bq + 1) for e in range(1, Bp + 1)],
    #         dtype=int
    #     )
    #     rng.shuffle(all_pairs)
    #     pairs = all_pairs[:nSamples]

    #     Y = np.zeros((2 * self.N, nSamples))
    #     x = np.arange(1, self.N + 1) * self.h

    #     for s, (b, e) in enumerate(pairs):
    #         q0 = np.sin(b * np.pi * x / self.L)
    #         p0 = np.sin(e * np.pi * x / self.L)
    #         Y[:, s] = np.concatenate([q0, p0])

    #     return Y
    
    def getInitialState(self, endT, dt, dT=0.1):
        """
        Generate a single compactly supported initial displacement
        with zero initial velocity.

        q0(x) = exp(-1 / (1 - ((x - x0)/(width/2))^2))   for |x - x0| < width/2
            = 0                                       otherwise

        p0(x) = 0

        Returns array of shape (2N, 1)
        """

        # only ONE initial condition
        nSamples = 1

        Y = np.zeros((2 * self.N, nSamples))

        # interior grid points
        x = np.arange(1, self.N + 1) * self.h

        # compact support parameters
        x0 = 0.5 * self.L          # center of the bump
        width = 2               # support width

        # initial displacement
        q0 = np.zeros_like(x)
        mask = np.abs(x - x0) < (width / 2)

        q0[mask] = np.exp(1.0) * np.exp(
            -1.0 / (1.0 - ((x[mask] - x0) / (width / 2)) ** 2)
        )

        # initial momentum (zero velocity)
        p0 = np.zeros_like(x)

        Y[:, 0] = np.concatenate([q0, p0])
        
        return Y


    # ------------------------------------------------------------------
    # Hamiltonian dynamics
    # ------------------------------------------------------------------
    def ode(self, y):
        q = y[:self.N]
        p = y[self.N:]
        qdot = p 
        pdot = -self.K.dot(q)
        return np.concatenate([qdot, pdot])

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------
    def energy(self, q, p):
        kin = 0.5 * (p @ self.Minv @ p)
        pot = 0.5 * ((self.K.dot(q) if self.sparse else self.K @ q) @ q)
        return float(kin + pot)

    def plotEnergy(self, solution, tSpan):
        N = self.N
        E = np.zeros_like(tSpan)
        for j in range(len(tSpan)):
            q = solution[:N, j]
            p = solution[N:, j]
            E[j] = self.energy(q, p)

        plt.figure(figsize=(8, 4))
        plt.plot(tSpan, E)
        plt.xlabel("Time")
        plt.ylabel("H(q,p)")
        plt.title("Total Energy")
        plt.tight_layout()
        plt.show()

    def plotErrors(self, solution, tSpan, labels, linestyles, title, xPos=None):
        plt.figure(figsize=(10, 12))
        for curve, style, label in zip(solution, linestyles, labels):
            plt.semilogy(tSpan, curve, linestyle=style, linewidth=5, label=label)

        ymin, ymax = plt.ylim()
        if xPos is not None:
            plt.plot([xPos, xPos], [ymin, ymax], linestyle='dashed', color='black')

        plt.legend(loc='lower right', fontsize=20)
        plt.xlabel('Time t', fontsize=20)
        plt.ylabel('Error', fontsize=20)
        plt.title(title, fontsize=24)
        plt.show()
