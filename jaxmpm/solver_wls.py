"""
WLS-corrected time-stepping solver.

This module mirrors ``solver.py`` exactly, but swaps the shape-function and
transfer kernels for their WLS-corrected counterparts:

    shape_functions_wls  →  pure quadratic B-spline + WLS moment solve
    transfer_wls         →  explicit force with corrected ∇Φ (P2G)
                            corrected Φ weights in velocity gather (G2P)

The block-scan / rematerialisation structure and the friction parametrisation
(softplus-reparameterised piecewise-constant bands) are unchanged.
"""

import jax
import jax.numpy as jnp
from jax import jit

from jaxmpm.shape_functions_wls import build_shape_fn_wls
from jaxmpm.transfer_wls import build_p2g_wls, build_g2p_wls
from jaxmpm.boundary import build_grid_op


def expand_bands_to_grid(band_values, n_grid_x, n_grid_y, num_bands=5):
    """Map a 1-D friction array to a 2-D grid field (identical to solver.py)."""
    band_width  = (n_grid_x + 1) // num_bands
    friction_map = jnp.zeros((n_grid_x + 1, n_grid_y + 1))
    for i in range(num_bands):
        x_start = i * band_width
        x_end   = (i + 1) * band_width if i < num_bands - 1 else n_grid_x + 1
        friction_map = friction_map.at[x_start:x_end, :].set(band_values[i])
    return friction_map


def build_solver_wls(cfg, n_particles):
    """Assemble the WLS-corrected differentiable MPM time-stepper.

    Parameters
    ----------
    cfg : MPMConfig
    n_particles : int

    Returns
    -------
    simulate_wls : callable
        ``simulate_wls(p_rho, v, x, C, pressure, friction_bands_raw)
          -> (p_rho, v, x, C, pressure, x_history)``

        Fully differentiable w.r.t. ``friction_bands_raw``.
        ``x_history`` is sub-sampled every ``cfg.save_every`` steps.
    """
    compute_weights, compute_wls = build_shape_fn_wls(cfg)
    p2g_wls  = build_p2g_wls(cfg)
    g2p_wls  = build_g2p_wls(cfg)
    grid_op  = build_grid_op(cfg)

    n_steps    = cfg.n_steps
    block_size = cfg.block_size
    n_blocks   = cfg.n_blocks
    remainder  = cfg.remainder
    n_grid_x   = cfg.n_grid_x
    n_grid_y   = cfg.n_grid_y

    # ------------------------------------------------------------------
    # Single sub-step  (inside jax.lax.scan)
    # The carry includes x_history so the write path is identical to the
    # validated original script and the gradient path is preserved.
    # ------------------------------------------------------------------
    @jit
    @jax.remat
    def substep(carry, xs):
        current, band_raw = xs

        # Reparameterise: raw → positive friction values
        band_pos  = jax.nn.softplus(band_raw)
        band_full = jnp.concatenate([band_pos, jnp.array([0.0])])
        friction_field = expand_bands_to_grid(band_full, n_grid_x, n_grid_y)

        p_rho, v, x, C, pressure, x_history = carry

        # ── WLS shape functions ─────────────────────────────────────────
        base, fx, w, dw = compute_weights(x)
        nb_mask, c0, cx, cy = compute_wls(base, x, w)

        # ── P2G (WLS weight-corrected) ───────────────────────────────
        p_rho, pressure, grid_v, grid_m = p2g_wls(
            p_rho, v, x, C, pressure,
            base, fx, w, dw,
            nb_mask, c0, cx, cy,
        )

        # ── Grid operations (gravity + Coulomb friction, unchanged) ─────
        grid_v_out = grid_op(grid_v, grid_m, friction_field)

        # ── G2P (WLS-corrected) ─────────────────────────────────────────
        v, x, C = g2p_wls(grid_v_out, v, x, base, fx, w, nb_mask, c0, cx, cy)

        # Write particle positions to history (carry-based, same as original)
        x_history = x_history.at[current].set(x)
        return (p_rho, v, x, C, pressure, x_history), None

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------
    @jit
    def simulate_wls(p_rho, v, x, C, pressure, friction_bands_raw):
        """Run the complete forward WLS-MPM simulation.

        Parameters
        ----------
        friction_bands_raw : jnp.ndarray, shape ``(num_bands - 1,)``
            Raw (pre-softplus) friction values.  The last band is fixed to 0.

        Returns
        -------
        Tuple ``(p_rho, v, x, C, pressure, x_history)``
        ``x_history`` is sub-sampled every ``cfg.save_every`` steps.
        ``x_history[0]`` is the initial configuration.
        """
        x_history = jnp.zeros((n_steps + 1, n_particles, 2))
        x_history = x_history.at[0].set(x)
        carry     = (p_rho, v, x, C, pressure, x_history)

        # Block-wise scan with rematerialisation
        for blk in range(n_blocks):
            indices    = jnp.arange(blk * block_size + 1,
                                    (blk + 1) * block_size + 1)
            bands_tile = jnp.tile(friction_bands_raw[None, :], (block_size, 1))
            carry, _   = jax.remat(
                lambda c: jax.lax.scan(substep, c, xs=(indices, bands_tile))
            )(carry)

        if remainder > 0:
            indices    = jnp.arange(n_blocks * block_size + 1,
                                    n_blocks * block_size + remainder + 1)
            bands_tile = jnp.tile(friction_bands_raw[None, :], (remainder, 1))
            carry, _   = jax.remat(
                lambda c: jax.lax.scan(substep, c, xs=(indices, bands_tile))
            )(carry)

        p_rho, v, x, C, pressure, x_history = carry

        # Sub-sample history to every save_every steps
        return p_rho, v, x, C, pressure, x_history[::cfg.save_every]

    return simulate_wls
