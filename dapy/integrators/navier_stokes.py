"""Simple Navier-Stokes fluid simulation on two-dimensional grid.

Based on methods proposed in

> Stam, Jos. A simple fluid solver based on the FFT.
> Journal of graphics tools 6.2 (2001): 43-52.
>
> Stam, Jos. Stable fluids. Proceedings of the 26th annual conference on
> Computer graphics and interactive techniques. ACM Press/Addison-Wesley
> Publishing Co., 1999.
>
> Fedkiw, Ronald, Jos Stam, and Henrik Wann Jensen. Visual simulation of smoke.
> Proceedings of the 28th annual conference on Computer graphics and
> interactive techniques. ACM, 2001.
>
> Kim, ByungMoon, Yingjie Liu, Ignacio Llamas, and Jarek Rossignac.
> FlowFixer: using BFECC for fluid simulation.
> Proceedings of the First Eurographics conference on Natural Phenomena.
> Eurographics Association, 2005.
"""

import math
import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
import numba as nb


@nb.njit(nb.double[:, :, :](nb.double[:, :, :], nb.double[:, :, :, :]))
def batch_bilinear_interpolate(fields, interp_points):
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
    new_fields = np.empty(fields.shape)
    for p in nb.prange(fields.shape[0]):
        for i in range(fields.shape[1]):
            for j in range(fields.shape[2]):
                # Calculate left edge index of interpolation point.
                l_ix = int(math.floor(interp_points[p, 0, i, j]))
                # Calculate horizontal weight coefficient for interpolation.
                h_wt = interp_points[p, 0, i, j] - l_ix
                # Wrap index to [0, grid_shape_0).
                l_ix = l_ix % fields.shape[1]
                # Calculate right edge index of interpolation point cell.
                r_ix = (l_ix + 1) % fields.shape[1]
                # Calculate top edge index of interpolation point.
                t_ix = int(math.floor(interp_points[p, 1, i, j]))
                # Calculate vertical weight coefficient for interpolation.
                v_wt = interp_points[p, 1, i, j] - t_ix
                # Wrap index to [0, grid_shape_1).
                t_ix = t_ix % fields.shape[2]
                # Calculate bottom edge index of interpolation point cell.
                b_ix = (t_ix + 1) % fields.shape[2]
                # Calculate new field value as weighted sum of field values
                # at grid points on corners of interpolation point grid cell.
                new_fields[p, i, j] = (
                    (1 - h_wt) * (1 - v_wt) * fields[p, l_ix, t_ix] +
                    (1 - h_wt) * v_wt * fields[p, l_ix, b_ix] +
                    h_wt * (1 - v_wt) * fields[p, r_ix, t_ix] +
                    h_wt * v_wt * fields[p, r_ix, b_ix]
                )
    return new_fields


@nb.njit(nb.double[:, :, :](nb.double[:, :, :, :], nb.double, nb.double))
def batch_calc_curl(vector_fields, cell_size_0, cell_size_1):
    """Calculate centred FD approximations of curl of 2D vector fields.

    Args:
        vector_fields (array): Stack of three dimensional arrays defining
            values of two-component vector fields on a rectlinear grid. The
            array should be of shape `(n_field, 2, grid_shape_0, grid_shape_1)`
            where `n_field` specifies the number of (independent) fields,
            `grid_shape_0` and `grid_shape_1` specify the number of grid
            points along the two grid dimensions and the size 2 dimension
            represents the two components of the vector field.
        cell_size_0 (float): Spatial extent of each grid cell along zero-axis.
        cell_size_1 (float): Spatial extent of each grid cell along one-axis.

    Returns:
        Three dimensional array corresponding to a stack of scalar curl fields,
        one per provided vector field. The array is of shape
        `(n_field, grid_shape_0, grid_shape_1)` with the fields in the same
        order as in the provided `vector_fields`.
    """
    n_field, _, grid_size_0, grid_size_1 = vector_fields.shape
    curls = np.empty((n_field, grid_size_0, grid_size_1))
    for f in nb.prange(n_field):
        for i in range(grid_size_0):
            for j in range(grid_size_1):
                curls[f, i, j] = 0.5 * (
                    (vector_fields[f, 0, (i + 1) % grid_size_0, j] -
                     vector_fields[f, 0, (i - 1) % grid_size_0, j]) /
                    cell_size_0 +
                    (vector_fields[f, 1, i, (j + 1) % grid_size_1] -
                     vector_fields[f, 1, i, (j - 1) % grid_size_1]) /
                    cell_size_1
                )
    return curls


@nb.njit(nb.double[:, :, :, :](nb.double[:, :, :], nb.double, nb.double))
def batch_calc_grad(fields, cell_size_0, cell_size_1):
    """Calculate centred FD approximations of gradient of scalar fields.

    Args:
        fields (array): Stack of two dimensional arrays defining values of
            scalar fields on a rectlinear grid. The array should be of shape
            `(n_field, grid_shape_0, grid_shape_1)`where `n_field` specifies
            the number of (independent) fields and `grid_shape_0` and
            `grid_shape_1` specify the number of grid points along the two
            grid dimensions.
        cell_size_0 (float): Spatial extent of each grid cell along zero-axis.
        cell_size_1 (float): Spatial extent of each grid cell along one-axis.

    Returns:
        Four dimensional array corresponding to a stack of vector gradient
        fields, one per provided vector field. The array is of shape
        `(n_field, 2, grid_shape_0, grid_shape_1)` with the fields in the same
        order as in the provided `vector_fields`.
    """
    n_field, grid_size_0, grid_size_1 = fields.shape
    grads = np.empty((n_field, 2, grid_size_0, grid_size_1))
    for f in nb.prange(n_field):
        for i in range(grid_size_0):
            for j in range(grid_size_1):
                grads[f, 0, i, j] = 0.5 * (
                    fields[f, (i + 1) % grid_size_0, j] -
                    fields[f, (i - 1) % grid_size_0, j]) / cell_size_0
                grads[f, 1, i, j] = 0.5 * (
                    fields[f, i, (j + 1) % grid_size_1] -
                    fields[f, i, (j - 1) % grid_size_1]) / cell_size_1
    return grads


class FourierNavierStokes2dIntegrator:
    """Incompressible Navier-Stokes fluid simulation on 2D periodic grid.

    Simulates evolution of a fluid velocity field and density field of a
    carrier particle in a field on a 2-torus using a finite difference based
    implementation of the incompressible Navier-Stokes equations in
    two-dimensions. A semi-Lagrangian method is used for the advection updates
    and a FFT-based method used to implement the diffusion of the velocity and
    density fields and to project the velocity field to a divergence-free flow
    to respect the incompressibility condition [1,3].

    Optionally vorticity confinement [2] may be used to compensate for the
    non-physical numerical dissipation of small-scale rotational structure in
    the simulated fields by adding an external force which amplifies
    voriticity in the velocity field.

    As an alternative (or in addition) to vorticity confinement, a second-
    order accurate BFECC (back and forward error correction and compensation)
    method can be used to simulate the advection steps [4] which decreases
    numerical dissipation of voriticity.

    References:
      1. Stam, Jos. A simple fluid solver based on the FFT.
         Journal of graphics tools 6.2 (2001): 43-52.
      2. Fedkiw, Ronald, Jos Stam, and Henrik Wann Jensen. Visual simulation of
         smoke. Proceedings of the 28th annual conference on Computer graphics
         and interactive techniques. ACM, 2001
      3. Stam, Jos. Stable fluids. Proceedings of the 26th annual conference on
         Computer graphics and interactive techniques. ACM Press/Addison-Wesley
         Publishing Co., 1999.
      4. Kim, ByungMoon, Yingjie Liu, Ignacio Llamas, and Jarek Rossignac.
         FlowFixer: using BFECC for fluid simulation. Proceedings of the First
         Eurographics conference on Natural Phenomena. Eurographics
         Association, 2005.
    """

    def __init__(self, grid_shape, grid_size=(2., 2.), density_source=None,
                 dt=0.05, dens_diff_coeff=2e-4, visc_diff_coeff=1e-4,
                 vort_coeff=5., use_vort_conf=False, use_bfecc=True,
                 dens_min=1e-8):
        """
        Args:
            grid_shape (tuple): Grid dimensions as a 2-tuple.
            grid_size (tuple): Spatial extent of simulation grid as a 2-tuple.
            density_source (array or None): Array defining density source field
                used to increment density field on each integrator time step.
                Should be of shape `grid_shape`. If `None` no density source
                update is applied.
            dt (float): Integrator time-step.
            dens_diff_coeff (float): Density diffusion coefficient.
            visc_diff_coeff (float): Velocity viscous diffusion coefficient.
            vort_coeff (float): Vorticity coeffient for voriticity confinement.
            use_vort_conf (bool): Whether to apply vorticity confinement
                update on each time step to velocity field.
            use_bfecc (bool): Whether to use BFECC advection steps instead of
                first-order semi-Lagrangian method.
            dens_min (float): Lower bound for density field values.
        """
        self.grid_shape = grid_shape
        self.grid_size = grid_size
        self.n_grid = grid_shape[0] * grid_shape[1]
        self.dt = dt
        # Calculate spatial size of each cell in grid.
        self.cell_size = np.array([
            self.grid_size[0] / float(self.grid_shape[0]),
            self.grid_size[0] / float(self.grid_shape[1])
        ])
        self.dens_diff_coeff = dens_diff_coeff
        self.visc_diff_coeff = visc_diff_coeff
        self.vort_coeff = vort_coeff
        self.use_vort_conf = use_vort_conf
        self.use_bfecc = use_bfecc
        self.dens_min = dens_min
        self.density_source = density_source
        # Coordinate indices of grid cell corners.
        self.cell_indices = np.array(np.meshgrid(
                np.arange(self.grid_shape[0]),
                np.arange(self.grid_shape[1]), indexing='ij'))
        # Spatial angular frequency values for real-valued FFT grid layout.
        freq_grid_0 = np.fft.fftfreq(
                grid_shape[0], grid_size[0] / grid_shape[0]) * 2 * np.pi
        freq_grid_1 = np.fft.rfftfreq(
                grid_shape[1], grid_size[1] / grid_shape[1]) * 2 * np.pi
        # Squared wavenumbers on FFT grid.
        self.wavnum_sq_grid = freq_grid_0[:, None]**2 + freq_grid_1[None, :]**2
        # Kernels in frequency space to simulate diffusion terms.
        # Corresponds to solving diffusion equation in 2D exactly in time with
        # spectral method to approximate second-order spatial derivatives.
        self.visc_diff_kernel = np.exp(
            -visc_diff_coeff * dt * self.wavnum_sq_grid)
        self.dens_diff_kernel = np.exp(
            -dens_diff_coeff * dt * self.wavnum_sq_grid)
        # For first derivative expressions zero Nyquist frequency for even
        # number of grid points:
        # > Notes on FFT-based differentiation.
        # > Steven G. Johnson, MIT Applied Mathematics.
        # > http://math.mit.edu/~stevenj/fft-deriv.pdf
        self.grad_0_kernel = freq_grid_0 * 1j
        self.grad_1_kernel = freq_grid_1 * 1j
        if grid_shape[0] % 2 == 0:
            self.grad_0_kernel[grid_shape[0] // 2] = 0
        if grid_shape[1] % 2 == 0:
            self.grad_1_kernel[grid_shape[1] // 2] = 0
        # Clip zero wave number square values to small positive constant to
        # avoid divide by zero warnings.
        wavnum_sq_grid_clip = np.maximum(self.wavnum_sq_grid, 1e-8)
        # Coefficients of vector field frequency components to solve Poisson's
        # equation to project to divergence-free field.
        self.fft_proj_coeff_00 = freq_grid_1[None, :]**2 / wavnum_sq_grid_clip
        self.fft_proj_coeff_11 = freq_grid_0[:, None]**2 / wavnum_sq_grid_clip
        self.fft_proj_coeff_01 = (
            freq_grid_0[:, None] * freq_grid_1[None, :] / wavnum_sq_grid_clip)

    def project_and_diffuse_velocity(self, velocity):
        """Diffuse velocity and project so divergence free using FFT method."""
        # Calculate 2D real-valued FFT of velocity fields.
        velocity_fft = fft.rfft2(velocity)
        # Convolve with viscous diffusion kernel in frequency space.
        velocity_fft_new = self.visc_diff_kernel * velocity_fft
        # Solve Poisson equation to project to divergence free field.
        velocity_fft_new = np.stack([
            self.fft_proj_coeff_00 * velocity_fft_new[:, 0] -
            self.fft_proj_coeff_01 * velocity_fft_new[:, 1],
            self.fft_proj_coeff_11 * velocity_fft_new[:, 1] -
            self.fft_proj_coeff_01 * velocity_fft_new[:, 0]
        ], axis=1)
        # Set zero-frequency component to previous value.
        velocity_fft_new[:, :, 0, 0] = velocity_fft[:, :, 0, 0]
        # Perform inverse 2D real-valued FFT to map back to spatial fields.
        return fft.irfft2(velocity_fft_new, overwrite_input=True)

    def diffuse_density(self, density):
        """Isotropically diffuse density field using FFT method."""
        # Calculate 2D real-valued FFT of density fields.
        density_fft = fft.rfft2(density)
        # Perform convolution with diffusion kernel in frequency space.
        density_fft *= self.dens_diff_kernel
        # Perform inverse 2D real-valued FFT to map back to spatial fields.
        return fft.irfft2(density_fft, overwrite_input=True)

    def semi_lagrangian_advect(self, field, velocity):
        """Use semi-Lagrangian method to advect a given field a single step."""
        # Estimate grid coordinates of particles which end up at grid corners
        # points when following current velocity field.
        particle_centers = self.cell_indices[None] - (
            velocity * self.dt / self.cell_size[None, :, None, None])
        # Calculate advected field values by bilinearly interpolating field
        # values at back traced particle locations.
        return batch_bilinear_interpolate(field, particle_centers)

    def advect(self, field, velocity):
        """Advect a given field a single step."""
        if self.use_bfecc:
            # Initial forwards step
            field_1 = self.semi_lagrangian_advect(field, velocity)
            # Backwards step from field_1
            field_2 = self.semi_lagrangian_advect(field_1, -velocity)
            # Compute error corrected original field
            field_3 = (3 * field - field_2) / 2.
            # Final forwards step
            return self.semi_lagrangian_advect(field_3, velocity)
        else:
            return self.semi_lagrangian_advect(field, velocity)

    def vorticity_confinement_force(self, velocity):
        """Calculate vorticity confinement force for current velocity field."""
        # Estimate two-dimensional curl of velocity field (vorticity).
        curl = batch_calc_curl(velocity, self.cell_size[0], self.cell_size[1])
        # Estimate gradient of curl field.
        grad_curl = batch_calc_grad(curl, self.cell_size[0], self.cell_size[1])
        # Calculate magnitude of curl gradient with fudge factor to avoid
        # division by zero.
        grad_curl_mag = (grad_curl**2).sum(1)**0.5 + 1e-8
        return self.vort_coeff * curl[:, None] * np.stack([
            grad_curl[:, 0] * self.cell_size[0],
            grad_curl[:, 1] * self.cell_size[1]
        ], axis=1) / grad_curl_mag[:, None]

    def update_velocity(self, velocity):
        """Perform single time step update of velocity field."""
        # If using vorticity confinement calculate force and update velocities.
        if self.use_vort_conf:
            velocity = velocity + self.dt * (
                self.vorticity_confinement_force(velocity))
        # Self advect both velocity field components using current velocities.
        velocity = np.stack([
            self.advect(velocity[:, 0], velocity),
            self.advect(velocity[:, 1], velocity)
        ], axis=1)
        return self.project_and_diffuse_velocity(velocity)

    def update_density(self, density, velocity):
        """Perform single time step update of density field."""
        # If density source term present use to increment field.
        if self.density_source is not None:
            density = density + self.dt * self.density_source
        # Advect density field by current velocity field.
        density = self.advect(density, velocity)
        # Diffuse density field.
        density = self.diffuse_density(density)
        # Diffusion and advection steps can produce non-positive density
        # fields therefore clip.
        np.clip(density, self.dens_min, None, density)
        return density

    def forward_integrate(self, z_curr, z_next, start_time_index, n_step=1):
        """Forward integrate system state one or more time steps.

        State vectors are arranged such that first `2 * n_grid` elements
        corrrespond to the velocity field, where
            n_grid = grid_shape[0] * grid_shape[1]
        is the number of elements in the spatial grid and the last `n_grid`
        elements correspond to the *logarithm* of the density field.

        Args:
            z_curr (array): Two dimensional array of current system states of
                shape `(n_state, dim_z)` where `n_state` is the number of
                independent system states being simulated and `dim_z` is the
                state dimension.
            z_next (array): Empty two dimensional array to write forward
                integrated states two. Should be of same shape as `z_curr`.
            start_time_index (int): Integer indicating current time index
                associated with the `z_curr` states (i.e. number of previous
                `forward_integrate` calls) to allow for calculate of time for
                non-homogeneous systems.
            n_step (int): Number of integrator time steps to perform.
        """
        n_batch = z_curr.shape[0]
        # Extract velocity and density fields from state vectors.
        velocity = z_curr[:, :2 * self.n_grid].reshape(
            (n_batch, 2,) + self.grid_shape)
        density = z_curr[:, 2 * self.n_grid:].reshape(
            (n_batch,) + self.grid_shape)
        for s in range(n_step):
            # Update velocity fields by a single time step.
            velocity = self.update_velocity(velocity)
            # Update density fields by a single time step.
            density = self.update_density(density, velocity)
        # Return concatenated velocity and density fields as new state vectors.
        return np.concatenate([
                velocity.reshape((n_batch, -1)),
                density.reshape((n_batch, -1))], axis=-1)
