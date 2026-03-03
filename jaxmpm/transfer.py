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

    The weakly-compressible equation of state is embedded in this transfer:
    pressure is updated incrementally from the volumetric strain rate.
    """
    dt = cfg.dt
    dh = cfg.dh
    inv_dh = cfg.inv_dh
    p_mass = cfg.p_mass
    rho0 = cfg.rho0
    modulus = cfg.modulus
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    dim = 2

    @jit
    def p2g(p_rho, v, x, C, pressure, base, fx, w):
        """Scatter particle data to the Eulerian grid.

        Returns
        -------
        p_rho_next : updated particle density
        pressure   : updated pressure tensor
        grid_v     : grid momentum
        grid_m     : grid mass
        """
        grid_v = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))
        grid_m = jnp.zeros((n_grid_x + 1, n_grid_y + 1))

        vol_strain_inc = jnp.trace(C, axis1=1, axis2=2) * dt
        p_vol_prev = p_mass / p_rho
        dJ = 1.0 + vol_strain_inc
        p_vol_next = p_vol_prev * dJ
        p_rho_next = p_rho / dJ

        dp = p_rho_next * modulus / rho0 * vol_strain_inc
        pressure = pressure - dp[:, None, None] * jnp.eye(2)
        stress = (-dt * 4.0 * p_vol_next[:, None, None] * inv_dh * inv_dh) * (-pressure)
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

        return p_rho_next, pressure, grid_v, grid_m

    return p2g


def build_g2p(cfg):
    """Return a JIT-compiled grid-to-particle transfer function (APIC)."""
    dt = cfg.dt
    dh = cfg.dh
    dim = 2

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
