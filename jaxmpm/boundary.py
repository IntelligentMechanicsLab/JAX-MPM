"""
Boundary condition helpers.

The ``build_grid_op`` factory returns a JIT-compiled function that

1. converts grid momentum to velocity,
2. applies gravity,
3. enforces wall boundary conditions with Coulomb-type frictional contact
   on the bottom wall.

Friction can be either a scalar or a 2-D spatial map defined on the grid.
"""

import jax
import jax.numpy as jnp
from jax import jit


def build_grid_op(cfg):
    """Return a JIT-compiled ``grid_op(grid_v, grid_m, friction_field)`` closure.

    Parameters
    ----------
    cfg : MPMConfig
        Simulation configuration.
    """
    dt = cfg.dt
    dh = cfg.dh
    gravity = cfg.gravity
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    domain_x = cfg.domain_x
    domain_y = cfg.domain_y

    # Pre-compute wall masks (constant across time-steps)
    ones = jnp.ones((n_grid_x + 1, n_grid_y + 1), dtype=bool)

    mask_left = (jnp.arange(n_grid_x + 1) * dh <= 0.0)[:, None] * ones
    mask_right = (jnp.arange(n_grid_x + 1) * dh >= domain_x)[:, None] * ones
    mask_bottom = (jnp.arange(n_grid_y + 1) * dh <= 0.0)[None, :] * ones
    mask_top = (jnp.arange(n_grid_y + 1) * dh >= domain_y)[None, :] * ones

    @jit
    def grid_op(grid_v, grid_m, friction_field):
        """Apply momentum → velocity conversion, gravity, and BCs.

        Parameters
        ----------
        grid_v : jnp.ndarray  (n_grid_x+1, n_grid_y+1, 2)
            Grid momentum.
        grid_m : jnp.ndarray  (n_grid_x+1, n_grid_y+1)
            Grid mass.
        friction_field : jnp.ndarray  (n_grid_x+1, n_grid_y+1)
            Spatially varying Coulomb friction coefficient on the bottom wall.

        Returns
        -------
        grid_v_out : jnp.ndarray  (n_grid_x+1, n_grid_y+1, 2)
        """
        inv_m = 1.0 / (grid_m + 1e-12)
        v_out = jnp.where(
            grid_m[:, :, None] > 0,
            inv_m[:, :, None] * grid_v,
            jnp.zeros_like(grid_v),
        )
        # gravity
        v_out = v_out.at[:, :, 1].add(-dt * gravity)

        # --- Wall BCs ---
        vx, vy = v_out[..., 0], v_out[..., 1]

        # No-slip side walls & top wall
        vx = jnp.where(mask_left | mask_right, 0.0, vx)
        vy = jnp.where(mask_top, 0.0, vy)

        # Bottom wall – Coulomb friction
        # Outward normal: n = (0, -1), i.e. pointing *into* the domain
        nI_x, nI_y = 0.0, -1.0
        D = vx * nI_x + vy * nI_y          # normal component (positive = approaching)
        C_tan = vx * nI_y - vy * nI_x      # tangential component
        abs_C = jnp.abs(C_tan)

        mu_eff = jnp.minimum(friction_field, abs_C / (D + 1e-12))

        vx = jnp.where(
            mask_bottom & (D > 0),
            vx - D * (nI_x + mu_eff / (abs_C + 1e-12) * nI_y * C_tan),
            vx,
        )
        vy = jnp.where(
            mask_bottom & (D > 0),
            vy - D * (nI_y - mu_eff / (abs_C + 1e-12) * nI_x * C_tan),
            vy,
        )

        return jnp.stack((vx, vy), axis=-1)

    return grid_op
