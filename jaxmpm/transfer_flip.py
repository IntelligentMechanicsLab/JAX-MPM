"""
FLIP (Fluid-Implicit-Particle) transfer operations.

FLIP differs from APIC in three ways:

1. **P2G** — the deformation gradient ``F`` is tracked explicitly (initialised
   to ``I``); particle volume and density are recomputed each step from
   ``J = det(F)``; pressure is computed fresh via the absolute EOS
   ``p = c^2 * (rho0/J - rho0)``; internal forces accumulate in a separate
   force grid ``grid_f`` via the weak form ``f = -V * sigma * grad_N``.

2. **Grid op** — mass-normalised velocity *and* acceleration grids are both
   returned so that the G2P step can perform the FLIP velocity blend.

3. **G2P** — velocity is updated via the FLIP blend
   ``v_new = v_old + sum_i a_i * w_i * dt``  (accumulate acceleration increment),
   while position is advected with the PIC velocity for stability.
   The velocity-gradient tensor ``Grad_v`` is computed via shape-fn gradients.

The persistent particle state for FLIP is ``(F, v, x, Grad_v)`` where ``F``
is the per-particle deformation gradient (``n_p x 2 x 2``, initialised to
``I``).  Unlike APIC, pressure is **not** carried as state -- it is
recomputed fresh every step from the current ``J``.
"""

import jax.numpy as jnp
from jax import jit


def build_p2g_flip(cfg):
    """Return a JIT-compiled FLIP particle-to-grid transfer function.

    Tracks the deformation gradient ``F`` and uses the absolute
    weakly-compressible EOS (same as the original FLIP reference code)::

        F_new = (I + dt * Grad_v) @ F
        J     = det(F_new)
        p     = c^2 * (rho0/J - rho0)

    P2G scatters pure momentum (no affine) to ``grid_v`` and accumulates
    internal forces in ``grid_f`` via the divergence theorem.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    p2g_flip : callable
        ``(F, v, x, Grad_v, base, fx, w, dw)``
        ``-> (F_new, grid_v, grid_m, grid_f)``
    """
    dt       = cfg.dt
    p_mass   = cfg.p_mass
    p_vol0   = cfg.p_vol0       # initial particle volume = (dh/2)^2
    rho0     = cfg.rho0
    c_sq     = cfg.c ** 2       # speed of sound squared
    mu       = cfg.mu           # dynamic viscosity (0 = inviscid)
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    dim      = 2

    @jit
    def p2g_flip(F, v, x, Grad_v, base, fx, w, dw):
        """Scatter particle data to the grid (FLIP formulation).

        Parameters
        ----------
        F      : deformation gradient  (n_p, 2, 2), initialised to identity
        Grad_v : velocity gradient gathered in the previous G2P step (n_p, 2, 2)

        Returns
        -------
        F_new  : updated deformation gradient  (n_p, 2, 2)
        grid_v : grid momentum  (mass x velocity, no affine term)
        grid_m : grid mass
        grid_f : grid internal force  (from stress divergence)
        """
        grid_v = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))
        grid_m = jnp.zeros((n_grid_x + 1, n_grid_y + 1))
        grid_f = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))

        # Update deformation gradient: F_new = (I + dt*grad_v) @ F
        new_F      = (jnp.eye(2) + dt * Grad_v) @ F
        J          = jnp.linalg.det(new_F)
        p_vol_next = p_vol0 * J

        # Absolute weakly-compressible EOS: p = c^2*(rho - rho0),  rho = rho0/J
        pressure_each = c_sq * (rho0 / J - rho0)           # (n_p,)  scalar

        # Full Newtonian Cauchy stress:
        #   σ = -p·I  - (2/3)·μ·tr(d)·I  + 2·μ·d
        # For FLIP, d = (Grad_v + Grad_v^T)/2  (velocity gradient from previous G2P)
        d = 0.5 * (Grad_v + Grad_v.transpose(0, 2, 1))
        stress = (
            -pressure_each[:, None, None] * jnp.eye(2)
            - (2.0 / 3.0 * mu) * jnp.trace(Grad_v, axis1=1, axis2=2)[:, None, None] * jnp.eye(2)
            + 2.0 * mu * d
        )

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
                # Internal force via divergence theorem:  f = -V * sigma * grad_N
                internal_force = -p_vol_next[:, None] * jnp.einsum(
                    "ijk,ik->ij", stress, dweight
                )
                grid_f = grid_f.at[idx[:, 0], idx[:, 1], :].add(internal_force)

        return new_F, grid_v, grid_m, grid_f

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
    ``v_new = v_old + sum_i  w_i * a_i * dt``
    and position is advected with the PIC velocity (sum of ``w_i * v_i``)
    for numerical stability.

    The velocity-gradient tensor ``Grad_v`` is computed from
    ``Grad_v = sum_i  v_i outer grad_N_i``.

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
        new_vflip = v    # FLIP: start from old velocity, add dv = sum a_i*w_i*dt
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
