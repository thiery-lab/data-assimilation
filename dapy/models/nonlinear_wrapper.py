"""Model wrapping linear-Gaussian model with bijective non-linearity."""

import numpy as np
import scipy.linalg as la
from dapy.utils.doc import inherit_docstrings
from dapy.models.base import AbstractModel


class NonLinearWrapperModel(AbstractModel):

    def __init__(self, base_model, bijection):
        self.base_model = base_model
        self.bijection = bijection

    @property
    def dim_z(self):
        return self.base_model.dim_z

    @property
    def dim_x(self):
        return self.base_model.dim_x

    @property
    def rng(self):
        return self.base_model.rng

    def init_state_sampler(self, n=None):
        return self.bijection.forward(self.base_model.init_state_sampler(n))

    def next_state_sampler(self, z, t):
        return self.bijection.forward(
            self.base_model.next_state_sampler(self.bijection.backward(z), t))

    def observation_func(self, z, t):
        return self.base_model.observation_func(self.bijection.backward(z), t)

    def observation_sampler(self, z, t):
        return self.base_model.observation_sampler(
            self.bijection.backward(z), t)

    def log_prob_dens_obs_gvn_state(self, x, z, t):
        return self.base_model.log_prob_dens_obs_gvn_state(
            x, self.bijection.backward(z), t)

    def log_prob_dens_init_state(self, z):
        return (
            self.base_model.log_prob_dens_init_state(
                self.bijection.backward(z)) -
            self.bijection.log_bwd_jacobian_det(z))

    def log_prob_dens_state_transition(self, z_n, z_c, t):
        return (
            self.base_model.log_prob_dens_state_transition(
                self.bijection.backward(z_n),
                self.bijection.backward(z_c), t) -
            self.bijection.log_bwd_jacobian_det(z_n))
