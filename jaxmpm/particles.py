"""
Particle initialisation utilities.
"""

import jax.numpy as jnp

from jaxmpm.config import MPMConfig


def initialize_block_particles(
    cfg: MPMConfig,
    block_x: float,
    block_y: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
):
    """Create a uniform block of particles inside the domain.

    Parameters
    ----------
    cfg : MPMConfig
        Simulation configuration.
    block_x, block_y : float
        Width and height of the particle block (metres).
    origin_x, origin_y : float
        Bottom-left corner of the block.

    Returns
    -------
    x : jnp.ndarray, shape ``(n_particles, 2)``
        Initial particle positions.
    n_particles : int
        Total number of particles created.
    """
    ppc = cfg.particles_per_cell_axis
    w_count = round(cfg.n_grid_x / (cfg.domain_x / block_x) * ppc)
    d_count = round(cfg.n_grid_y / (cfg.domain_y / block_y) * ppc)

    dx = block_x / w_count
    dy = block_y / d_count

    ii, kk = jnp.meshgrid(jnp.arange(w_count), jnp.arange(d_count), indexing="ij")
    x = jnp.stack(
        [origin_x + (ii + 0.5) * dx, origin_y + (kk + 0.5) * dy], axis=-1
    ).reshape(-1, 2)

    return x, x.shape[0]
