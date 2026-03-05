"""
Quadratic B-spline shape functions for MPM with boundary stencil correction.

For particles whose 3-node stencil would extend outside the domain (i.e.
``base == -1`` near the left/bottom walls, ``base == n_grid-1`` near the
right/top walls), the weights are adjusted so that the out-of-bounds node
always receives **zero weight**.  This ensures:

* Partition of unity is maintained within the valid grid nodes.
* No momentum or mass is scattered to (or gathered from) out-of-bounds nodes.
* The G2P ``Dp`` matrix only accumulates contributions from valid nodes,
  keeping ``B @ inv(Dp)`` well-conditioned in the backward pass.

Near the left/bottom boundary (``base == -1``):
    w = [0,  1-(fx-1),  fx-1]   (reduced linear-like, node-0 weight = 0)

Near the right/top boundary (``base == n_grid - 1``):
    w = [1-fx,  fx,  0]         (node-2 weight = 0)

Everywhere else: standard quadratic B-spline.
"""

import jax
import jax.numpy as jnp
from jax import jit


def build_shape_fn(cfg):
    """Return a JIT-compiled ``compute_weights(x)`` closure.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    compute_weights : callable
        ``(x) -> (base, fx, w)``

        * ``base`` – integer base-node index, shape ``(n_p, 2)``
        * ``fx``   – fractional position inside the cell, shape ``(n_p, 2)``
        * ``w``    – shape-function values, shape ``(3, n_p, 2)``
    """
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    inv_dh   = cfg.inv_dh

    @jit
    def _boundary_type(base):
        """Return per-particle stencil type: 0=interior, 1=left/bottom, 2=right/top."""
        bx = jnp.where(base[:, 0] == -1, 1,
              jnp.where(base[:, 0] == n_grid_x - 1, 2, 0))
        by = jnp.where(base[:, 1] == -1, 1,
              jnp.where(base[:, 1] == n_grid_y - 1, 2, 0))
        return jnp.stack([bx, by], axis=-1)  # (n_p, 2)

    @jit
    def compute_weights(x):
        n_p  = x.shape[0]
        base = jnp.floor(x * inv_dh - 0.5).astype(jnp.int32)
        fx   = x * inv_dh - base

        # Standard quadratic B-spline
        w0 = jnp.array([0.5 * (1.5 - fx) ** 2,
                         0.75 - (fx - 1.0) ** 2,
                         0.5 * (fx - 0.5) ** 2])       # (3, n_p, 2)

        # Near left / bottom wall (base == -1): zero weight on node 0
        w1 = jnp.array([jnp.zeros_like(fx),
                         1.0 - (fx - 1.0),
                         fx - 1.0])

        # Near right / top wall (base == n_grid - 1): zero weight on node 2
        w2 = jnp.array([1.0 - fx, fx, jnp.zeros_like(fx)])

        btype = _boundary_type(base)  # (n_p, 2)

        all_wx = jnp.stack((w0[:, :, 0], w1[:, :, 0], w2[:, :, 0]), axis=-1)
        all_wy = jnp.stack((w0[:, :, 1], w1[:, :, 1], w2[:, :, 1]), axis=-1)

        sel_wx = all_wx[:, jnp.arange(n_p), btype[:, 0]]
        sel_wy = all_wy[:, jnp.arange(n_p), btype[:, 1]]

        w = jnp.stack((sel_wx, sel_wy), axis=-1)  # (3, n_p, 2)
        return base, fx, w

    return compute_weights
