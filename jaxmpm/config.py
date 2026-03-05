"""
Configuration for JAX-MPM simulations.

Stores all physical and numerical parameters in a single dataclass so that
every function can be configured from one place.
"""

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp


@dataclass
class MPMConfig:
    """Central configuration for a 2-D weakly-compressible MPM simulation.

    Parameters
    ----------
    domain_x, domain_y : float
        Physical size of the computational domain (metres).
    n_grid_x : int
        Number of grid cells along the *x*-direction.  ``n_grid_y`` is derived
        automatically so that the grid spacing is uniform.
    dt : float
        Time-step size (seconds).
    total_time : float
        Total simulation time (seconds).
    rho0 : float
        Reference density of the material (kg/m³).
    c : float
        Artificial speed of sound used in the equation of state.
    gravity : float
        Gravitational acceleration (m/s²).
    particles_per_cell_axis : int
        Number of particles seeded per grid-cell edge during initialisation.
    save_every : int
        Interval (in time-steps) at which particle states are stored.
    block_size : int
        Number of time-steps per ``jax.lax.scan`` block (controls the
        rematerialisation / memory trade-off).
    """

    # --- domain geometry ---
    domain_x: float = 2.0
    domain_y: float = 0.4
    n_grid_x: int = 200

    # --- time integration ---
    dt: float = 3e-5
    total_time: float = 0.4

    # --- material ---
    rho0: float = 1000.0
    c: float = 35.0
    mu: float = 0.0        # dynamic viscosity  (0 = inviscid fluid)
    gravity: float = 9.8

    # --- particle seeding ---
    particles_per_cell_axis: int = 2

    # --- I/O ---
    save_every: int = 10

    # --- scan block size for memory management ---
    block_size: int = 500

    # ------------------------------------------------------------------
    # Derived quantities (set in __post_init__)
    # ------------------------------------------------------------------
    dh: float = field(init=False)
    inv_dh: float = field(init=False)
    n_grid_y: int = field(init=False)
    n_steps: int = field(init=False)
    n_blocks: int = field(init=False)
    remainder: int = field(init=False)
    p_vol0: float = field(init=False)
    p_mass: float = field(init=False)
    modulus: float = field(init=False)

    def __post_init__(self):
        self.dh = self.domain_x / self.n_grid_x
        self.inv_dh = 1.0 / self.dh
        self.n_grid_y = round(self.domain_y / self.dh)
        self.n_steps = round(self.total_time / self.dt)
        self.n_blocks = self.n_steps // self.block_size
        self.remainder = self.n_steps % self.block_size
        self.p_vol0 = (self.dh * 0.5) ** 2
        self.p_mass = self.p_vol0 * self.rho0
        self.modulus = self.c ** 2 * self.rho0
