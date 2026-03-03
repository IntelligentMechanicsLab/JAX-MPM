"""
Quadratic B-spline shape functions for MPM with boundary-aware corrections.

The standard 3-node quadratic B-spline support is shifted near domain edges so
that the stencil never reaches outside the grid.
"""

import jax
import jax.numpy as jnp
from jax import jit


def build_shape_fn(cfg):
    """Return a JIT-compiled ``compute_weights(x)`` closure.

    Parameters
    ----------
    cfg : MPMConfig
        Simulation configuration.

    Returns
    -------
    compute_weights : callable
        ``(x) -> (base, fx, w)`` where *base* is the integer base-node index,
        *fx* the fractional position inside the cell, and *w* the shape-function
        values with shape ``(3, n_particles, 2)``.
    """
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    inv_dh = cfg.inv_dh

    @jit
    def _classify_base(base):
        """Classify each particle stencil type near boundaries."""
        bx = jnp.where(base[:, 0] == -1, 1,
              jnp.where(base[:, 0] == n_grid_x - 1, 2, 0))
        by = jnp.where(base[:, 1] == -1, 1,
              jnp.where(base[:, 1] == n_grid_y - 1, 2, 0))
        return jnp.stack([bx, by], axis=-1)

    @jit
    def compute_weights(x):
        n_p = x.shape[0]

        base = jnp.floor(x * inv_dh - 0.5).astype(jnp.int32)
        fx = x * inv_dh - base

        # Standard quadratic B-spline
        w0 = jnp.array([0.5 * (1.5 - fx) ** 2,
                         0.75 - (fx - 1) ** 2,
                         0.5 * (fx - 0.5) ** 2])

        # Modified weights for left boundary (base == -1)
        w1 = jnp.array([jnp.zeros_like(fx),
                         1.0 - (fx - 1),
                         fx - 1])

        # Modified weights for right boundary (base == n_grid - 1)
        w2 = jnp.array([1.0 - fx, fx, jnp.zeros_like(fx)])

        dtype = _classify_base(base)

        # Select the correct weight set per particle per axis
        all_wx = jnp.stack((w0[:, :, 0], w1[:, :, 0], w2[:, :, 0]), axis=-1)
        all_wy = jnp.stack((w0[:, :, 1], w1[:, :, 1], w2[:, :, 1]), axis=-1)

        sel_wx = all_wx[:, jnp.arange(n_p), dtype[:, 0]]
        sel_wy = all_wy[:, jnp.arange(n_p), dtype[:, 1]]

        w = jnp.stack((sel_wx, sel_wy), axis=-1)  # (3, n_p, 2)
        return base, fx, w

    return compute_weights
