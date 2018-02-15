cimport dapy.integrators.implicitmidpoint as impl
import dapy.integrators.implicitmidpoint as impl


cdef class Lorenz63Integrator(impl.ImplicitMidpointIntegrator):

    cdef double sigma, rho, beta

    def __init__(self, double sigma=10., double rho=28., double beta=8./3.,
                 double dt=0.01, double tol=1e-8, int max_iters=100,
                 int n_threads=4):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super(Lorenz63Integrator, self).__init__(
            3, dt, tol, max_iters, n_threads)

    cdef void update_dz_dt(self, double[:] z, double t, double[:] dz_dt) nogil:
        dz_dt[0] = self.sigma * (z[1] - z[0])
        dz_dt[1] = z[0] * (self.rho - z[2]) - z[1]
        dz_dt[2] = z[0] * z[1] - self.beta * z[2]
