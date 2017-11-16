cdef class ImplicitMidpointIntegrator:
    
    cdef double dt, tol
    cdef int dim_z, n_steps_per_update, max_iters, n_threads
    cdef int[:] intervals
    cdef double[:, :] z_temp
    cdef double[:, :] z_half
    cdef double[:, :] dz_dt
    
    cdef void update_dz_dt(self, double[:] z, double t, double[:] dz_dt) nogil

    cdef bint implicit_midpoint_step(
            self, double[:] z, double time, double[:] dz_dt, 
            double[:] z_half, double[:] z_next) nogil
            
    cdef void partition_particles(self, int n_particles)
