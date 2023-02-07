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


class FourierETDRK4Integrator(object):
    """ETD RK4 integrator for 1D semi-linear PDE models on periodic domains."""

    def __init__(self, linear_operator, nonlinear_operator, num_mesh_point, domain_size,
                 time_step, num_roots_of_unity=16):
        self.time_step = time_step
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.num_mesh_point = num_mesh_point
        self.domain_size = domain_size
        self.num_roots_of_unity = num_roots_of_unity
        # Calculate spatial frequencies for spectral derivatives
        self.freqs = 2 * np.pi * np.arange(num_mesh_point // 2 + 1) / domain_size
        self.freqs_sq = self.freqs**2
        # Set Nyquist frequency to zero for odd-derivatives when dim_z odd
        # See http://math.mit.edu/~stevenj/fft-deriv.pdf
        if num_mesh_point % 2 == 0:
            self.freqs[-1] = 0.
        # Diagonal linear operator and corresponding matrix exponentials
        lin_op = linear_operator(self.freqs, self.freqs_sq)
        self.exp_lin_full = np.exp(time_step * lin_op)
        self.exp_lin_half = np.exp(0.5 * time_step * lin_op)
        # Approximate contour integrals in complex plane to calculate
        # ETDRK4 update coefficients using quadrature
        roots_of_unity = np.exp(
            1j * np.pi * (np.arange(num_roots_of_unity) + 0.5) / num_roots_of_unity)
        lr = time_step * lin_op[:, None] + roots_of_unity[None, :]
        lr_squ = lr**2
        lr_cub = lr**3
        exp_lr = np.exp(lr)
        self.coeff_f0 = time_step * (((np.exp(lr / 2.) - 1) / lr).mean(1)).real
        self.coeff_f1 = time_step * (
            ((-4 - lr + exp_lr * (4 - 3 * lr + lr_squ)) / lr_cub).mean(1)).real
        self.coeff_f2 = time_step * (
            ((2 + lr + exp_lr * (lr - 2)) / lr_cub).mean(1)).real
        self.coeff_f3 = time_step * (
            ((-4 - 3 * lr - lr_squ + exp_lr * (4 - lr)) / lr_cub).mean(1)).real

    def step(self, states):
        states_0 = states
        n_0 = self.nonlinear_operator(states_0, self.freqs, self.freqs_sq)
        states_1 = self.exp_lin_half * states_0 + self.coeff_f0 * n_0
        n_1 = self.nonlinear_operator(states_1, self.freqs, self.freqs_sq)
        states_2 = self.exp_lin_half * states_0 + self.coeff_f0 * n_1
        n_2 = self.nonlinear_operator(states_2, self.freqs, self.freqs_sq)
        states_3 = self.exp_lin_half * states_1 + self.coeff_f0 * (2 * n_2 - n_0)
        n_3 = self.nonlinear_operator(states_3, self.freqs, self.freqs_sq)
        return (self.exp_lin_full * states_0 + self.coeff_f1 * n_0 +
                self.coeff_f2 * (n_1 + n_2) * 2 + self.coeff_f3 * n_3)

    def forward_integrate(self, states, start_time_index, num_step=1):
        for s in range(num_step):
            states = self.step(states)
        return states
