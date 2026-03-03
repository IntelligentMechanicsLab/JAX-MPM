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

__all__ = [
    "MPMConfig",
    "initialize_block_particles",
    "build_solver",
]
