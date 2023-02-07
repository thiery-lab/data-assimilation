cdef class ImplicitMidpointIntegrator:
    
    cdef double time_step, fixed_point_tol
    cdef int dim_state, max_fixed_point_iter, num_thread
    cdef int[:] intervals
    cdef double[:, :] x_temp
    cdef double[:, :] x_half
    cdef double[:, :] dx_dt
    
    cdef void update_dx_dt(self, double[:] x, double t, double[:] dx_dt) nogil

    cdef bint implicit_midpoint_step(
            self, double[:] x, double time, double[:] dx_dt, 
            double[:] x_half, double[:] x_next) nogil
            
    cdef void partition_states(self, int num_state)
