"""Implicit mid-point integrator for ordinary differential equations."""

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport isnan, isinf, fabs

class ConvergenceError(Exception):
    """Raised when implicit integrator step fails to converge."""

cdef class ImplicitMidpointIntegrator:

    def __init__(self, int dim_state, double time_step, double fixed_point_tol, 
                 int max_fixed_point_iter, int num_thread=4):
        self.dim_state = dim_state
        self.time_step = time_step
        self.fixed_point_tol = fixed_point_tol
        self.max_fixed_point_iter = max_fixed_point_iter
        self.num_thread = num_thread
        self.intervals = np.empty((num_thread,), dtype='int32')
        self.x_temp = np.empty((num_thread, dim_state), dtype='double')
        self.x_half = np.empty((num_thread, dim_state), dtype='double')
        self.dx_dt = np.empty((num_thread, dim_state), dtype='double')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void update_dx_dt(self, double[:] x, double t, double[:] dx_dt) nogil:
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef bint implicit_midpoint_step(
            self, double[:] x, double time, double[:] dx_dt,
            double[:] x_half, double[:] x_next) nogil:
        cdef double max_abs_diff, abs_diff, prev_val
        cdef int i, j
        self.update_dx_dt(x, time + self.time_step / 2., dx_dt)
        for j in range(self.dim_state):
            x_next[j] = x[j] + self.time_step * dx_dt[j]
        i = 0
        max_abs_diff = self.fixed_point_tol + 1.
        while max_abs_diff > self.fixed_point_tol and i < self.max_fixed_point_iter:
            max_abs_diff = 0.
            for j in range(self.dim_state):
                x_half[j] = (x_next[j] + x[j]) / 2.
            self.update_dx_dt(x_half, time + self.time_step / 2., dx_dt)
            for j in range(self.dim_state):
                prev_val = x_next[j]
                x_next[j] = x[j] + self.time_step * dx_dt[j]
                abs_diff = fabs(x_next[j] - prev_val)
                if isnan(abs_diff) or isinf(abs_diff):
                    return 1
                if abs_diff > max_abs_diff:
                    max_abs_diff = abs_diff
            i += 1
        if i == self.max_fixed_point_iter:
            return 1
        else:
            return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void partition_states(self, int num_state):
        """State particle index range in to equal sized intervals.
        Used to allocate state updates to different parallel threads.
        Args:
            num_stat: Total number of states.
        """
        cdef int t
        for t in range(self.num_thread):
            self.intervals[t] = <int>(
                t * <float>(num_state) / self.num_thread)
        self.intervals[self.num_thread] = num_state

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def forward_integrate(
            self, double[:, :] states, int start_time_index, int num_step=1):
        """Integrate a set of state particles forward in time.

        Args:
            states: Array of current states of shape `(num_state, dim_state)`.
            start_time_index: Integer indicating current time index.
            num_step: Number of integrator time steps to perform.

        Returns:
            Array of forward integrated states values of shape `(num_state, dim_state)`.
        """
        cdef int num_state = states.shape[0]
        self.partition_states(num_state)
        cdef double[:, :] next_states = np.empty(
            (num_state, self.dim_state), dtype='double')
        cdef int t, p, s,
        cdef bint error
        cdef double time = start_time_index * num_step * self.time_step
        for t in prange(self.num_thread, nogil=True, schedule='static',
                        chunksize=1, num_threads=self.num_thread):
            for p in range(self.intervals[t], self.intervals[t+1]):
                for s in range(num_step):
                    if s == 0:
                        self.x_temp[t, :] = states[p]
                    else:
                        self.x_temp[t, :] = next_states[p]
                    error = self.implicit_midpoint_step(
                        self.x_temp[t], time, self.dx_dt[t], self.x_half[t],
                        next_states[p])
                    if error == 1:
                        with gil:
                            raise ConvergenceError(
                                'Convergence error in implicit midpoint step.')
                    time = time + self.time_step
        return next_states
