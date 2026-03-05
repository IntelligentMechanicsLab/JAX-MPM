"""
FLIP (Fluid-Implicit-Particle) transfer operations.

FLIP differs from APIC in three ways:

1. **P2G** — pure momentum scatter (no affine term); internal forces are
   accumulated in a separate force grid ``grid_f`` via the weak form
   ``f_i = -V · σ · ∇N_i``.

2. **Grid op** — mass-normalised velocity *and* acceleration grids are both
   returned so that the G2P step can perform the FLIP velocity blend.

3. **G2P** — velocity is updated via the FLIP blend
   ``v_new = v_old + Σ a_i · w_i · dt``  (accumulate acceleration increment),
   while position is advected with the PIC velocity for stability.
   The velocity-gradient tensor ``Grad_v`` is computed via shape-fn gradients.

The state matrix for FLIP is called ``Grad_v`` (shape ``(n_p, 2, 2)``) and
plays the same role as ``C`` in APIC — it stores the particle velocity
gradient used in the next P2G step to compute volumetric strain.
"""

import jax.numpy as jnp
from jax import jit


def build_p2g_flip(cfg):
    """Return a JIT-compiled FLIP particle-to-grid transfer function.

    P2G scatters pure momentum (no affine) to ``grid_v`` and accumulates
    internal forces in ``grid_f`` via the divergence theorem.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    p2g_flip : callable
        ``(p_rho, v, x, Grad_v, pressure, base, fx, w, dw)``
        ``-> (p_rho_next, pressure, grid_v, grid_m, grid_f)``
    """
    dt       = cfg.dt
    dh       = cfg.dh
    inv_dh   = cfg.inv_dh
    p_mass   = cfg.p_mass
    rho0     = cfg.rho0
    modulus  = cfg.modulus
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    dim      = 2

    @jit
    def p2g_flip(p_rho, v, x, Grad_v, pressure, base, fx, w, dw):
        """Scatter particle data to the grid (FLIP formulation).

        Returns
        -------
        p_rho_next : updated particle density
        pressure   : updated pressure tensor
        grid_v     : grid momentum  (mass × velocity, no affine)
        grid_m     : grid mass
        grid_f     : grid internal force  (from stress divergence)
        """
        grid_v = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))
        grid_m = jnp.zeros((n_grid_x + 1, n_grid_y + 1))
        grid_f = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))

        # Density / volume update (same equation of state as APIC)
        vol_strain = jnp.trace(Grad_v, axis1=1, axis2=2) * dt
        p_vol_prev = p_mass / p_rho
        dJ          = 1.0 + vol_strain
        p_vol_next  = p_vol_prev * dJ
        p_rho_next  = p_rho / dJ

        dp       = p_rho_next * modulus / rho0 * vol_strain
        pressure = pressure - dp[:, None, None] * jnp.eye(2)
        stress   = -pressure   # mu = 0 (inviscid fluid)

        for i in range(3):
            for j in range(3):
                offset  = jnp.array([i, j])
                weight  = w[i, :, 0] * w[j, :, 1]
                dw_x    = dw[i, :, 0] * w[j, :, 1]
                dw_y    = w[i, :, 0] * dw[j, :, 1]
                dweight = jnp.stack([dw_x, dw_y], axis=1)  # (n_p, 2)

                idx = base + offset
                # Pure momentum scatter (no affine term)
                grid_v = grid_v.at[idx[:, 0], idx[:, 1], :].add(
                    weight[:, None] * p_mass * v
                )
                grid_m = grid_m.at[idx[:, 0], idx[:, 1]].add(weight * p_mass)
                # Internal force via divergence theorem:  f = -V · σ · ∇N
                internal_force = -p_vol_next[:, None] * jnp.einsum(
                    "ijk,ik->ij", stress, dweight
                )
                grid_f = grid_f.at[idx[:, 0], idx[:, 1], :].add(internal_force)

        return p_rho_next, pressure, grid_v, grid_m, grid_f

    return p2g_flip


def build_grid_op_flip(cfg):
    """Return a JIT-compiled FLIP grid update.

    Unlike the standard ``build_grid_op``, this returns **both** the
    updated velocity and the acceleration grids, which are needed by
    the FLIP G2P step.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    grid_op_flip : callable
        ``(grid_v, grid_m, grid_f) -> (grid_v_out, grid_a_out)``
    """
    dt       = cfg.dt
    dh       = cfg.dh
    gravity  = cfg.gravity
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    domain_x = cfg.domain_x

    ones        = jnp.ones((n_grid_x + 1, n_grid_y + 1), dtype=bool)
    mask_left   = (jnp.arange(n_grid_x + 1) * dh <= 0.0)[:, None] * ones
    mask_right  = (jnp.arange(n_grid_x + 1) * dh >= domain_x)[:, None] * ones
    mask_bottom = (jnp.arange(n_grid_y + 1) * dh <= 0.0)[None, :] * ones

    @jit
    def _apply_bc(v_field):
        vx = jnp.where(mask_left | mask_right, 0.0, v_field[..., 0])
        vy = jnp.where(mask_bottom, 0.0, v_field[..., 1])
        return jnp.stack((vx, vy), axis=-1)

    @jit
    def grid_op_flip(grid_v, grid_m, grid_f):
        """Apply mass normalisation, gravity, force grid, and BCs.

        Parameters
        ----------
        grid_v : jnp.ndarray  (n_grid_x+1, n_grid_y+1, 2)  grid momentum
        grid_m : jnp.ndarray  (n_grid_x+1, n_grid_y+1)      grid mass
        grid_f : jnp.ndarray  (n_grid_x+1, n_grid_y+1, 2)  internal force

        Returns
        -------
        grid_v_out : updated grid velocity  (n_grid_x+1, n_grid_y+1, 2)
        grid_a_out : grid acceleration      (n_grid_x+1, n_grid_y+1, 2)
                     (needed by FLIP G2P for the velocity increment)
        """
        inv_m = 1.0 / (grid_m + 1e-12)
        # Velocity from momentum
        v_out = jnp.where(
            grid_m[:, :, None] > 0,
            inv_m[:, :, None] * grid_v,
            jnp.zeros_like(grid_v),
        )
        # Acceleration from internal force
        a_out = jnp.where(
            grid_m[:, :, None] > 0,
            inv_m[:, :, None] * grid_f,
            jnp.zeros_like(grid_f),
        )
        # Apply gravity to acceleration
        a_out = a_out.at[:, :, 1].add(-gravity)
        # Integrate velocity
        v_out = v_out + a_out * dt

        return _apply_bc(v_out), _apply_bc(a_out)

    return grid_op_flip


def build_g2p_flip(cfg):
    """Return a JIT-compiled FLIP grid-to-particle transfer function.

    Velocity is updated via the FLIP blend
    ``v_new = v_old + Σ_i  w_i · a_i · dt``
    and position is advected with the PIC velocity (sum of ``w_i · v_i``)
    for numerical stability.

    The velocity-gradient tensor ``Grad_v`` is computed from
    ``Grad_v = Σ_i  v_i ⊗ ∇N_i``.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    g2p_flip : callable
        ``(grid_v_out, grid_a_out, v, x, base, fx, w, dw)``
        ``-> (v_new, x_new, Grad_v_new)``
    """
    dt  = cfg.dt
    dim = 2

    @jit
    def g2p_flip(grid_v_out, grid_a_out, v, x, base, fx, w, dw):
        """Gather grid data to particles (FLIP formulation).

        Returns
        -------
        v_new     : FLIP-blended particle velocity
        x_new     : updated position  (advected with PIC velocity)
        Grad_v_new: updated velocity-gradient tensor  (for next P2G)
        """
        n_p       = x.shape[0]
        new_vpic  = jnp.zeros((n_p, dim))
        new_vflip = v    # FLIP: start from old velocity, add Δv = Σ a_i·w_i·dt
        Grad_v    = jnp.zeros((n_p, dim, dim))

        for i in range(3):
            for j in range(3):
                g_v    = grid_v_out[base[:, 0] + i, base[:, 1] + j]
                g_a    = grid_a_out[base[:, 0] + i, base[:, 1] + j]
                weight = w[i, :, 0] * w[j, :, 1]
                dw_x   = dw[i, :, 0] * w[j, :, 1]
                dw_y   = w[i, :, 0] * dw[j, :, 1]
                dweight = jnp.stack([dw_x, dw_y], axis=1)  # (n_p, 2)

                new_vflip += weight[:, None] * g_a * dt   # FLIP increment
                new_vpic  += weight[:, None] * g_v        # PIC (for position)
                Grad_v    += jnp.einsum("ij,ik->ijk", g_v, dweight)

        # FLIP velocity for particles; stable PIC velocity for advection
        return new_vflip, x + dt * new_vpic, Grad_v

    return g2p_flip
