"""
High-level time-stepping solver.

``build_solver`` assembles all MPM kernels (shape functions, P2G, grid
operations, G2P) into a single differentiable ``simulate`` function that can
be passed directly to ``jax.grad``.
"""

import jax
import jax.numpy as jnp
from jax import jit

from jaxmpm.shape_functions import build_shape_fn
from jaxmpm.transfer import build_p2g, build_g2p
from jaxmpm.boundary import build_grid_op


def expand_bands_to_grid(band_values, n_grid_x, n_grid_y, num_bands=5):
    """Map a 1-D array of *num_bands* friction values to a 2-D grid field.

    The domain is divided into ``num_bands`` equal-width vertical strips.
    """
    band_width = (n_grid_x + 1) // num_bands
    friction_map = jnp.zeros((n_grid_x + 1, n_grid_y + 1))
    for i in range(num_bands):
        x_start = i * band_width
        x_end = (i + 1) * band_width if i < num_bands - 1 else n_grid_x + 1
        friction_map = friction_map.at[x_start:x_end, :].set(band_values[i])
    return friction_map


def build_solver(cfg, n_particles):
    """Assemble the full differentiable MPM time-stepper.

    Parameters
    ----------
    cfg : MPMConfig
    n_particles : int

    Returns
    -------
    simulate : callable
        ``(p_rho, v, x, C, pressure, friction_bands) -> (p_rho, v, x, C, pressure, x_history)``

        *friction_bands* is a 1-D array of shape ``(num_bands,)``; the function
        is fully differentiable w.r.t. this argument.
    """
    compute_weights = build_shape_fn(cfg)
    p2g = build_p2g(cfg)
    g2p = build_g2p(cfg)
    grid_op = build_grid_op(cfg)

    n_steps = cfg.n_steps
    block_size = cfg.block_size
    n_blocks = cfg.n_blocks
    remainder = cfg.remainder
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    save_every = cfg.save_every

    # ------------------------------------------------------------------
    # Single sub-step (inside jax.lax.scan)
    # Mirrors the original validated script exactly:
    # - step index passed so x_history can be written at the right slot
    # - x_history lives in carry (same gradient path as original)
    # ------------------------------------------------------------------
    @jit
    @jax.remat
    def substep(carry, xs):
        current, band_raw = xs
        band_pos = jax.nn.softplus(band_raw)
        band_full = jnp.concatenate([band_pos, jnp.array([0.0])])
        friction_field = expand_bands_to_grid(band_full, n_grid_x, n_grid_y)

        p_rho, v, x, C, pressure, x_history = carry

        base, fx, w = compute_weights(x)
        p_rho, pressure, grid_v, grid_m = p2g(p_rho, v, x, C, pressure, base, fx, w)
        grid_v_out = grid_op(grid_v, grid_m, friction_field)
        v, x, C = g2p(grid_v_out, v, x, base, fx, w)

        x_history = x_history.at[current].set(x)
        return (p_rho, v, x, C, pressure, x_history), None

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------
    @jit
    def simulate(p_rho, v, x, C, pressure, friction_bands_raw):
        """Run the complete forward simulation.

        Parameters
        ----------
        friction_bands_raw : jnp.ndarray, shape ``(num_bands - 1,)``
            *Raw* (pre-softplus) friction values for each vertical strip.
            The last band is fixed to 0.

        Returns
        -------
        Tuple ``(p_rho, v, x, C, pressure, x_history)``
        where ``x_history`` is sub-sampled every ``save_every`` steps.
        x_history[0] is the initial position.
        """
        x_history = jnp.zeros((n_steps + 1, n_particles, 2))
        x_history = x_history.at[0].set(x)
        carry = (p_rho, v, x, C, pressure, x_history)

        for blk in range(n_blocks):
            indices = jnp.arange(blk * block_size + 1, (blk + 1) * block_size + 1)
            bands_tile = jnp.tile(friction_bands_raw[None, :], (block_size, 1))
            carry, _ = jax.remat(
                lambda c: jax.lax.scan(substep, c, xs=(indices, bands_tile))
            )(carry)

        if remainder > 0:
            indices = jnp.arange(n_blocks * block_size + 1,
                                 n_blocks * block_size + remainder + 1)
            bands_tile = jnp.tile(friction_bands_raw[None, :], (remainder, 1))
            carry, _ = jax.remat(
                lambda c: jax.lax.scan(substep, c, xs=(indices, bands_tile))
            )(carry)

        p_rho, v, x, C, pressure, x_history = carry
        return p_rho, v, x, C, pressure, x_history[::save_every]

    return simulate
