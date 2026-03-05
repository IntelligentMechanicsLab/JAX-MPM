"""
JAX-MPM: A Learning-Augmented Differentiable Meshfree Framework
================================================================

A general-purpose differentiable MPM solver built on JAX, enabling GPU-accelerated
forward and inverse modeling of large deformations, frictional contact, and inelastic
materials for geomechanics and geophysical hazard applications.

Reference:
    Du, H., & He, Q. (2025). JAX-MPM: A Learning-Augmented Differentiable Meshfree
    Framework for GPU-Accelerated Lagrangian Simulation and Geophysical Inverse Modeling.
    arXiv preprint arXiv:2507.04192.
"""

from jaxmpm.config import MPMConfig
from jaxmpm.particles import initialize_block_particles
from jaxmpm.solver import build_solver
from jaxmpm.shape_functions import build_shape_fn, build_shape_fn_with_grad
from jaxmpm.transfer import build_p2g, build_g2p
from jaxmpm.transfer_tpic import build_g2p_tpic
from jaxmpm.transfer_flip import build_p2g_flip, build_grid_op_flip, build_g2p_flip
from jaxmpm.boundary import build_grid_op, build_grid_op_frictionless

__all__ = [
    "MPMConfig",
    "initialize_block_particles",
    "build_solver",
    # shape functions
    "build_shape_fn",
    "build_shape_fn_with_grad",
    # transfers
    "build_p2g",
    "build_g2p",
    "build_g2p_tpic",
    "build_p2g_flip",
    "build_grid_op_flip",
    "build_g2p_flip",
    # boundary / grid ops
    "build_grid_op",
    "build_grid_op_frictionless",
]
