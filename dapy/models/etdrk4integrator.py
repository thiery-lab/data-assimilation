"""Exponential time-differencing fourth-order Runge-Kutta integrators.

References:

  1. Kassam, Aly-Khan and Trefethen, Lloyd N.
     Fourth-order time-stepping for stiff PDEs.
     SIAM Journal on Scientific Computing 26.4 (2005): 1214-1233.
  2. Cox, Steven M. and Matthews, Paul C.
     Exponential time differencing for stiff systems.
     Journal of Computational Physics 176.2 (2002): 430-455.
"""

import numpy as np
import pyfftw.interfaces.numpy_fft as fft


class FourierETDRK4Integrator(object):
    """ETD RK4 integrator for 1D semi-linear PDE models on periodic domains."""

    def __init__(self, linear_operator, nonlinear_operator, n_grid, grid_size,
                 dt, n_roots=16):
        self.dt = dt
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.n_grid = n_grid
        self.grid_size = grid_size
        self.dt = dt
        self.n_roots = n_roots
        # Calculate spatial frequencies for spectral derivatives
        self.freqs = 2 * np.pi * np.arange(n_grid // 2 + 1) / grid_size
        self.freqs_sq = self.freqs**2
        # Set Nyquist frequency to zero for odd-derivatives when dim_z odd
        # See http://math.mit.edu/~stevenj/fft-deriv.pdf
        if n_grid % 2 == 0:
            self.freqs[-1] = 0.
        # Diagonal linear operator and corresponding matrix exponentials
        lin_op = linear_operator(self.freqs, self.freqs_sq)
        self.exp_lin_full = np.exp(dt * lin_op)
        self.exp_lin_half = np.exp(0.5 * dt * lin_op)
        # Approximate contour integrals in complex plane to calculate
        # ETDRK4 update coefficients using quadrature
        roots_of_unity = np.exp(
            1j * np.pi * (np.arange(n_roots) + 0.5) / n_roots)
        lr = dt * lin_op[:, None] + roots_of_unity[None, :]
        lr_squ = lr**2
        lr_cub = lr**3
        exp_lr = np.exp(lr)
        self.coeff_f0 = dt * (((np.exp(lr / 2.) - 1) / lr).mean(1)).real
        self.coeff_f1 = dt * (
            ((-4 - lr + exp_lr * (4 - 3 * lr + lr_squ)) / lr_cub).mean(1)).real
        self.coeff_f2 = dt * (
            ((2 + lr + exp_lr * (lr - 2)) / lr_cub).mean(1)).real
        self.coeff_f3 = dt * (
            ((-4 - 3 * lr - lr_squ + exp_lr * (4 - lr)) / lr_cub).mean(1)).real

    def step_fft(self, v):
        n_v = self.nonlinear_operator(v, self.freqs, self.freqs_sq)
        a = self.exp_lin_half * v + self.coeff_f0 * n_v
        n_a = self.nonlinear_operator(a, self.freqs, self.freqs_sq)
        b = self.exp_lin_half * v + self.coeff_f0 * n_a
        n_b = self.nonlinear_operator(b, self.freqs, self.freqs_sq)
        c = self.exp_lin_half * a + self.coeff_f0 * (2 * n_b - n_v)
        n_c = self.nonlinear_operator(c, self.freqs, self.freqs_sq)
        return (self.exp_lin_full * v + self.coeff_f1 * n_v +
                self.coeff_f2 * (n_a + n_b) * 2 + self.coeff_f3 * n_c)

    def forward_integrate(self, z, start_time_index, n_step=1):
        v = np.fft.rfft(z)
        for s in range(n_step):
            # Step forward in Fourier space
            v = self.step_fft(v)
        return np.fft.irfft(v)
