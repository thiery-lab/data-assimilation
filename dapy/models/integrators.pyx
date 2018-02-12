cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport isnan, isinf, fabs

class ConvergenceError(Exception):
    """Raised when implicit integrator step fails to converge."""

cdef class ImplicitMidpointIntegrator:

    def __init__(self, int dim_z, double dt,  double tol, int max_iters,
                 int n_threads=4):
        self.dim_z = dim_z
        self.dt = dt
        self.tol = tol
        self.max_iters = max_iters
        self.n_threads = n_threads
        self.intervals = np.empty((n_threads,), dtype='int32')
        self.z_temp = np.empty((n_threads, dim_z), dtype='double')
        self.z_half = np.empty((n_threads, dim_z), dtype='double')
        self.dz_dt = np.empty((n_threads, dim_z), dtype='double')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void update_dz_dt(self, double[:] z, double t, double[:] dz_dt) nogil:
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef bint implicit_midpoint_step(
            self, double[:] z, double time, double[:] dz_dt,
            double[:] z_half, double[:] z_next) nogil:
        cdef double max_abs_diff, abs_diff, prev_val
        cdef int i, j
        self.update_dz_dt(z, time + self.dt / 2., dz_dt)
        for j in range(self.dim_z):
            z_next[j] = z[j] + self.dt * dz_dt[j]
        i = 0
        max_abs_diff = self.tol + 1.
        while max_abs_diff > self.tol and i < self.max_iters:
            max_abs_diff = 0.
            for j in range(self.dim_z):
                z_half[j] = (z_next[j] + z[j]) / 2.
            self.update_dz_dt(z_half, time + self.dt / 2., dz_dt)
            for j in range(self.dim_z):
                prev_val = z_next[j]
                z_next[j] = z[j] + self.dt * dz_dt[j]
                abs_diff = fabs(z_next[j] - prev_val)
                if isnan(abs_diff) or isinf(abs_diff):
                    return 1
                if abs_diff > max_abs_diff:
                    max_abs_diff = abs_diff
            i += 1
        if i == self.max_iters:
            return 1
        else:
            return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void partition_particles(self, int n_particles):
        """Partition particle index range in to equal sized intervals.
        Used to allocate particle updates to different parallel threads.
        Args:
            n_particles (int): Total number of particles.
            n_threads (int): Number of parallel threads being used.
        """
        cdef int t
        for t in range(self.n_threads):
            self.intervals[t] = <int>(
                t * <float>(n_particles) / self.n_threads)
        self.intervals[self.n_threads] = n_particles

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def forward_integrate(
            self, double[:, :] z_particles, int start_time_index,
            int n_steps=1):
        """Integrate a set of state particles forward in time.

        Args:
            z_particles (array): Array of current state particle values of
                shape `(n_particles, dim_z)`.
            start_time_index (int): Integer indicating current time index
                associated with the `z_curr` states (i.e. number of previous
                `forward_integrate` calls) to allow for calculate of time for
                non-homogeneous systems.
            n_step (int): Number of integrator time steps to perform.

        Returns:
            Array of forward propagated state particle values of shape
            `(n_particles, dim_z)`.
        """
        cdef int n_particles = z_particles.shape[0]
        self.partition_particles(n_particles)
        cdef double[:, :] z_particles_next = np.empty(
            (n_particles, self.dim_z), dtype='double')
        cdef int t, p, s,
        cdef bint error
        cdef double time = start_time_index * n_steps * self.dt
        for t in prange(self.n_threads, nogil=True, schedule='static',
                        chunksize=1, num_threads=self.n_threads):
            for p in range(self.intervals[t], self.intervals[t+1]):
                for s in range(n_steps):
                    if s == 0:
                        self.z_temp[t, :] = z_particles[p]
                    else:
                        self.z_temp[t, :] = z_particles_next[p]
                    error = self.implicit_midpoint_step(
                        self.z_temp[t], time, self.dz_dt[t], self.z_half[t],
                        z_particles_next[p])
                    if error == 1:
                        with gil:
                            raise ConvergenceError(
                                'Convergence error in implicit midpoint step.')
                    time = time + self.dt
        return z_particles_next
