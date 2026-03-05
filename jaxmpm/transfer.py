"""
Particle ↔ grid transfer operations (APIC formulation).

* **P2G** – scatter particle momentum and mass onto the background grid.
* **G2P** – gather updated grid velocities back to particles and update the
  APIC affine velocity field *C*.
"""

import jax
import jax.numpy as jnp
from jax import jit


def build_p2g(cfg):
    """Return a JIT-compiled particle-to-grid transfer function.

    Uses F-tracking for density (same approach as FLIP):

        F_new = (I + dt·C) @ F      (C ≈ ∇v in APIC)
        J     = det(F_new)
        p     = c² · (ρ₀/J − ρ₀)   (absolute weakly-compressible EOS)
    """
    dt       = cfg.dt
    dh       = cfg.dh
    inv_dh   = cfg.inv_dh
    p_mass   = cfg.p_mass
    p_vol0   = cfg.p_vol0
    rho0     = cfg.rho0
    c_sq     = cfg.c ** 2
    mu       = cfg.mu
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    dim      = 2

    @jit
    def p2g(F, v, x, C, base, fx, w):
        """Scatter particle data to the Eulerian grid.

        Parameters
        ----------
        F : (n_p, 2, 2)  deformation gradient, initialised to identity
        C : (n_p, 2, 2)  APIC affine velocity matrix (≈ ∇v)

        Returns
        -------
        F_new  : updated deformation gradient
        grid_v : grid momentum
        grid_m : grid mass
        """
        grid_v = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))
        grid_m = jnp.zeros((n_grid_x + 1, n_grid_y + 1))

        # F-tracking density / volume update
        new_F      = (jnp.eye(2) + dt * C) @ F
        J          = jnp.linalg.det(new_F)
        p_vol_next = p_vol0 * J

        # Absolute weakly-compressible EOS: p = c²·(ρ − ρ₀),  ρ = ρ₀/J
        pressure_scalar = c_sq * (rho0 / J - rho0)   # (n_p,)

        # Full Newtonian Cauchy stress:
        #   σ = -p·I  - (2/3)·μ·tr(d)·I  + 2·μ·d
        # where d = (C + Cᵀ)/2  (APIC: C ≈ ∇v)
        d = 0.5 * (C + C.transpose(0, 2, 1))
        cauchy_stress = (
            -pressure_scalar[:, None, None] * jnp.eye(2)
            - (2.0 / 3.0 * mu) * jnp.trace(C, axis1=1, axis2=2)[:, None, None] * jnp.eye(2)
            + 2.0 * mu * d
        )
        stress = (-dt * 4.0 * p_vol_next[:, None, None] * inv_dh * inv_dh) * cauchy_stress
        affine = stress + p_mass * C

        for i in range(3):
            for j in range(3):
                offset = jnp.array([i, j])
                dpos = (jnp.array([i, j], dtype=float) - fx) * dh
                weight = w[i, :, 0] * w[j, :, 1]
                contrib = weight[:, None] * (
                    p_mass * v + jnp.einsum("ijk,ik->ij", affine, dpos)
                )
                idx = base + offset
                grid_v = grid_v.at[idx[:, 0], idx[:, 1], :].add(contrib)
                grid_m = grid_m.at[idx[:, 0], idx[:, 1]].add(weight * p_mass)

        return new_F, grid_v, grid_m

    return p2g


def build_g2p(cfg):
    """Return a JIT-compiled grid-to-particle transfer function (APIC)."""
    dt       = cfg.dt
    dh       = cfg.dh
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    dim      = 2

    @jit
    def g2p(grid_v, v, x, base, fx, w):
        """Gather grid velocities to particles and advect.

        Returns
        -------
        v_new : updated particle velocities
        x_new : updated particle positions
        C_new : updated APIC affine matrix
        """
        n_p = x.shape[0]
        new_v = jnp.zeros((n_p, dim))
        B = jnp.zeros((n_p, dim, dim))
        Dp = jnp.zeros((n_p, dim, dim))

        for i in range(3):
            for j in range(3):
                dpos = jnp.array([i, j], dtype=float) - fx
                g_v = grid_v[base[:, 0] + i, base[:, 1] + j]
                weight = w[i, :, 0] * w[j, :, 1]

                new_v += weight[:, None] * g_v
                B += weight[:, None, None] * jnp.einsum("ij,ik->ijk", g_v, dpos) * dh
                Dp += weight[:, None, None] * jnp.einsum("ij,ik->ijk", dpos, dpos) * dh * dh

        C_new = B @ jnp.linalg.inv(Dp)
        x_new = x + dt * new_v
        return new_v, x_new, C_new

    return g2p
