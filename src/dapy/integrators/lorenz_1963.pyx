from . cimport implicit_midpoint as impl
from . import implicit_midpoint as impl


cdef class Lorenz1963Integrator(impl.ImplicitMidpointIntegrator):

    cdef double sigma, rho, beta

    def __init__(self, double sigma=10., double rho=28., double beta=8./3.,
                 double time_step=0.01, double fixed_point_tol=1e-8,
                 int max_fixed_point_iter=100, int num_thread=4):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super(Lorenz1963Integrator, self).__init__(
            3, time_step, fixed_point_tol, max_fixed_point_iter, num_thread)

    cdef void update_dx_dt(self, double[:] x, double t, double[:] dx_dt) nogil:
        dx_dt[0] = self.sigma * (x[1] - x[0])
        dx_dt[1] = x[0] * (self.rho - x[2]) - x[1]
        dx_dt[2] = x[0] * x[1] - self.beta * x[2]
