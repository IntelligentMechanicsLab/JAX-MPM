"""
APIC Particle ↔ Grid transfer with WLS boundary correction.

Key differences from ``transfer.py``:

* **P2G**: The internal stress force is computed with WLS-corrected shape-
  function *gradients* (Soga 2016, Eq. 62) instead of the MLS-MPM affine
  approximation (which is only valid for a uniform interior stencil).
  Near-boundary particles use:
      ∇Φ_ij = (C₂ + C₃ · r_ij) · φ_ij
  Interior particles use the standard analytic gradient:
      ∇φ_ij = (dN_i/dx · N_j,  N_i · dN_j/dy)

* **G2P**: Mass-weighted velocity gather uses WLS-corrected weights Φ_ij
  near walls, standard weights φ_ij elsewhere.

The constitutive model (incremental weakly-compressible pressure) is
identical to the original ``transfer.py``.
"""

import jax
import jax.numpy as jnp
from jax import jit


def build_p2g_wls(cfg):
    """Return a JIT-compiled WLS-corrected particle-to-grid transfer.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    p2g_wls : callable
        ``p2g_wls(p_rho, v, x, C, pressure, base, fx, w, dw,
                  nb_mask, c0, cx, cy, C2, C3)
         -> (p_rho_next, pressure, grid_v, grid_m)``
    """
    dt       = cfg.dt
    dh       = cfg.dh
    p_mass   = cfg.p_mass
    rho0     = cfg.rho0
    modulus  = cfg.modulus
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    dim      = 2

    @jit
    def p2g_wls(p_rho, v, x, C, pressure,
                base, fx, w, dw,
                nb_mask, c0, cx, cy):
        """WLS-corrected scatter from particles to grid.

        WLS correction is applied to the *weights* (ensuring partition of unity
        near walls).  The *force* term always uses the standard analytic B-spline
        gradient: the corrected gradient (from inv(A)) is numerically unstable
        at domain corners where only 2×2 stencil nodes are active.

        Parameters
        ----------
        p_rho     : (n_p,)      particle densities
        v         : (n_p, 2)    particle velocities
        x         : (n_p, 2)    particle positions
        C         : (n_p,2,2)   APIC affine velocity matrix
        pressure  : (n_p,2,2)   Cauchy pressure tensor (diagonal)
        base      : (n_p, 2)    base grid-node indices
        fx        : (n_p, 2)    fractional cell positions
        w         : (3, n_p, 2) standard B-spline weights
        dw        : (3, n_p, 2) standard B-spline gradients [m⁻¹]
        nb_mask   : (n_p,)      True = particle is near a wall node
        c0,cx,cy  : (n_p,)      WLS weight-correction scalars

        Returns
        -------
        p_rho_next, pressure, grid_v, grid_m
        """
        grid_v = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))
        grid_m = jnp.zeros((n_grid_x + 1, n_grid_y + 1))

        # ── Density / pressure update (identical to transfer.py) ──────────
        vol_strain = jnp.trace(C, axis1=1, axis2=2) * dt
        p_vol_prev = p_mass / p_rho
        dJ         = 1.0 + vol_strain
        p_vol_next = p_vol_prev * dJ
        p_rho_next = p_rho / dJ

        dp       = p_rho_next * modulus / rho0 * vol_strain
        pressure = pressure - dp[:, None, None] * jnp.eye(2)
        stress   = -pressure  # (n_p, 2, 2)

        # ── Loop over 3×3 stencil ─────────────────────────────────────────
        for i in range(3):
            for j in range(3):
                # -- Node indices with OOB guard --
                ii  = base[:, 0] + i
                jj  = base[:, 1] + j
                act = (ii >= 0) & (ii <= n_grid_x) & (jj >= 0) & (jj <= n_grid_y)
                am  = act.astype(jnp.float64)   # (n_p,) 0 for inactive nodes
                iic = jnp.clip(ii, 0, n_grid_x)
                jjc = jnp.clip(jj, 0, n_grid_y)

                # r = x_node - x_p  (physical units)
                r  = (base + jnp.array([i, j])) * dh - x   # (n_p, 2)

                # -- WLS-corrected weight --
                phi_ij  = w[i, :, 0] * w[j, :, 1]          # (n_p,)  standard
                phi_cor = (c0 + cx * r[:, 0] + cy * r[:, 1]) * phi_ij  # corrected
                # clamp to non-negative: WLS can produce tiny negatives at corners
                phi_cor = jnp.maximum(0.0, phi_cor)
                weight  = jnp.where(nb_mask, phi_cor, phi_ij) * am

                # Standard analytic B-spline gradient (always used for force).
                # The WLS-corrected gradient via inv(A) is ill-conditioned at
                # domain corners and is omitted for forward stability.
                grad_phi = jnp.stack([
                    dw[i, :, 0] * w[j, :, 1],   # ∂N/∂x
                    w[i, :, 0] * dw[j, :, 1],   # ∂N/∂y
                ], axis=-1) * am[:, None]        # (n_p, 2)

                dpos = (jnp.array([i, j], dtype=float) - fx) * dh   # (n_p, 2)

                # -- APIC momentum scatter (no stress in affine) --
                add_mom = p_mass * v + jnp.einsum('nij,nj->ni', p_mass * C, dpos)
                grid_v  = grid_v.at[iic, jjc, :].add(weight[:, None] * add_mom)
                grid_m  = grid_m.at[iic, jjc].add(weight * p_mass)

                # -- Explicit internal force via corrected gradient --
                # f_ip = -V_p · σ_p : ∇Φ_ip   (scattered to node i)
                force  = -p_vol_next[:, None] * jnp.einsum('nab,nb->na', stress, grad_phi)
                grid_v = grid_v.at[iic, jjc, :].add(dt * force)

        return new_F, grid_v, grid_m

    return p2g_wls


def build_g2p_wls(cfg):
    """Return a JIT-compiled WLS-corrected grid-to-particle transfer (APIC).

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    g2p_wls : callable
        ``g2p_wls(grid_v, v, x, base, fx, w, nb_mask, c0, cx, cy)
         -> (v_new, x_new, C_new)``
    """
    dt       = cfg.dt
    dh       = cfg.dh
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    dim      = 2

    @jit
    def g2p_wls(grid_v, v, x, base, fx, w, nb_mask, c0, cx, cy):
        """WLS-corrected gather from grid to particles (APIC).

        Parameters
        ----------
        grid_v  : (nx+1, ny+1, 2)  updated grid velocity
        v       : (n_p, 2)         particle velocities (unused, overwritten)
        x       : (n_p, 2)         particle positions
        base    : (n_p, 2)         base grid-node indices
        fx      : (n_p, 2)         fractional cell positions
        w       : (3, n_p, 2)      standard B-spline weights
        nb_mask : (n_p,)           True = near boundary
        c0,cx,cy: (n_p,)           WLS weight-correction scalars

        Returns
        -------
        v_new, x_new, C_new
        """
        n_p   = x.shape[0]
        new_v = jnp.zeros((n_p, dim))
        B     = jnp.zeros((n_p, dim, dim))
        Dp    = jnp.zeros((n_p, dim, dim))

        for i in range(3):
            for j in range(3):
                # OOB guard
                ii  = base[:, 0] + i
                jj  = base[:, 1] + j
                act = (ii >= 0) & (ii <= n_grid_x) & (jj >= 0) & (jj <= n_grid_y)
                am  = act.astype(jnp.float64)
                iic = jnp.clip(ii, 0, n_grid_x)
                jjc = jnp.clip(jj, 0, n_grid_y)
                g_v = grid_v[iic, jjc]  # (n_p, 2)

                r       = (base + jnp.array([i, j])) * dh - x   # x_node - x_p
                phi_ij  = w[i, :, 0] * w[j, :, 1]
                phi_cor = (c0 + cx * r[:, 0] + cy * r[:, 1]) * phi_ij
                phi_cor = jnp.maximum(0.0, phi_cor)  # clamp negatives at corners
                weight  = jnp.where(nb_mask, phi_cor, phi_ij) * am  # (n_p,)

                # dpos in physical units for B / Dp matrices
                dpos = (jnp.array([i, j], dtype=float) - fx) * dh   # (n_p, 2)

                new_v = new_v + weight[:, None] * g_v
                B     = B  + weight[:, None, None] * jnp.einsum('ni,nj->nij', g_v, dpos)
                Dp    = Dp + weight[:, None, None] * jnp.einsum('ni,nj->nij', dpos, dpos)

        # APIC affine:  C = B · D_p⁻¹
        Dp    = Dp + 1e-12 * jnp.eye(2)[None]
        C_new = B @ jnp.linalg.inv(Dp)
        x_new = x + dt * new_v
        return new_v, x_new, C_new

    return g2p_wls
