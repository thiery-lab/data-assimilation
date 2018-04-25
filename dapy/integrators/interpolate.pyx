import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport floor
cimport cython

@cython.cdivision(True)
cdef int python_mod(int n, int m) nogil:
    return ((n % m) + m) % m

def batch_bilinear_interpolate(
        double[:, :, :] fields, double[:, :, :, :] interp_points,
        int n_thread=1):
    """Use bilinear interpolation to map 2D fields to a new set of points.

    Periodic boundary conditions are assumed. In the comments below the
    following layout is assumed for the grid cell surrounding each
    interpolation point

      (l_ix, t_ix) ---- (r_ix, t_ix)
            |                |
            |                |
      (l_ix, b_ix) ---- (r_ix, b_ix)

    Args:
        fields (array): Stack of two dimensional arrays defining values of a
            scalar field on a rectilinear grid. The array should be of shape
            `(n_field, grid_shape_0, grid_shape_1)` where `n_field` specifies
            the number of (independent) spatial fields and `grid_shape_0` and
            `grid_shape_1` specify the number of grid points along the two grid
            dimensions.
        interp_points (array): Stack of three dimensional arrays defining
            spatial points to resample field at in array index coordinates. The
            array should be of shape `(n_field, 2, grid_shape_0, grid_shape_1)`
            where `n_field`, `grid_shape_0` and `grid_shape_1` are as above and
            the size 2 dimension represents the two spatial coordinates.
    """
    cdef int p, i, j
    cdef int l_ix, r_ix, t_ix, b_ix
    cdef int n_field = fields.shape[0]
    cdef int dim_0 = fields.shape[1]
    cdef int dim_1 = fields.shape[2]
    cdef double h_wt, v_wt
    cdef np.ndarray[double, ndim=3, mode='c'] new_fields = np.empty(
        (n_field, dim_0, dim_1), dtype='double', order='C')
    cdef double[:, :, :] new_fields_mv = new_fields
    for p in prange(n_field, schedule='static', num_threads=n_thread,
                    nogil=True):
        for i in range(dim_0):
            for j in range(dim_1):
                # Calculate left edge index of interpolation point.
                l_ix = int(floor(interp_points[p, 0, i, j]))
                # Calculate horizontal weight coefficient for interpolation.
                h_wt = interp_points[p, 0, i, j] - l_ix
                # Wrap index to [0, grid_shape_0).
                l_ix = python_mod(l_ix, dim_0)
                # Calculate right edge index of interpolation point cell.
                r_ix = python_mod(l_ix + 1, dim_0)
                # Calculate top edge index of interpolation point.
                t_ix = int(floor(interp_points[p, 1, i, j]))
                # Calculate vertical weight coefficient for interpolation.
                v_wt = interp_points[p, 1, i, j] - t_ix
                # Wrap index to [0, grid_shape_1).
                t_ix = python_mod(t_ix, dim_1)
                # Calculate bottom edge index of interpolation point cell.
                b_ix = python_mod(t_ix + 1, dim_1)
                # Calculate new field value as weighted sum of field values
                # at grid points on corners of interpolation point grid cell.
                new_fields_mv[p, i, j] = (
                    (1 - h_wt) * (1 - v_wt) * fields[p, l_ix, t_ix] +
                    (1 - h_wt) * v_wt * fields[p, l_ix, b_ix] +
                    h_wt * (1 - v_wt) * fields[p, r_ix, t_ix] +
                    h_wt * v_wt * fields[p, r_ix, b_ix]
                )
    return new_fields
