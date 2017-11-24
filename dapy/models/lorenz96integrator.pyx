cimport dapy.models.integrators as integrators
import dapy.models.integrators as integrators


cdef class Lorenz96Integrator(integrators.ImplicitMidpointIntegrator):

    cdef double delta, force

    def __init__(self, dim_z=40, double delta=1./3., double force=8.,
                 double dt=0.005, double tol=1e-8,  int max_iters=100,
                 int n_threads=4):
        self.delta = delta
        self.force = force
        super(Lorenz96Integrator, self).__init__(
            dim_z, dt, tol, max_iters, n_threads)

    cdef int circ_shift(self, int i, int s) nogil:
        cdef int j = (i + s) % self.dim_z
        if j >= 0:
            return j
        else:
            return j + self.dim_z

    cdef void update_dz_dt(self, double[:] z, double t, double[:] dz_dt) nogil:
        cdef int i
        for i in range(self.dim_z):
            dz_dt[i] = -z[self.circ_shift(i, -1)] * (
                z[self.circ_shift(i, 1)] - z[self.circ_shift(i, -2)]) / (
                    3 * self.delta) - z[i] + self.force
