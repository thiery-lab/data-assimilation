cimport dapy.integrators.implicit_midpoint as impl
import dapy.integrators.implicit_midpoint as impl


cdef class Lorenz1996Integrator(impl.ImplicitMidpointIntegrator):

    cdef double delta, force

    def __init__(self, dim_state=40, double delta=1./3., double force=8.,
                 double time_step=0.005, double fixed_point_tol=1e-8, 
                 int max_fixed_point_iter=100, int num_thread=4):
        self.delta = delta
        self.force = force
        super(Lorenz1996Integrator, self).__init__(
            dim_state, time_step, fixed_point_tol, max_fixed_point_iter, num_thread)

    cdef int circ_shift(self, int i, int s) nogil:
        cdef int j = (i + s) % self.dim_state
        if j >= 0:
            return j
        else:
            return j + self.dim_state

    cdef void update_dx_dt(self, double[:] x, double t, double[:] dx_dt) nogil:
        cdef int i
        for i in range(self.dim_state):
            dx_dt[i] = -x[self.circ_shift(i, -1)] * (
                x[self.circ_shift(i, 1)] - x[self.circ_shift(i, -2)]) / (
                    3 * self.delta) - x[i] + self.force
