"""
TPIC (Taylor Particle-In-Cell) grid-to-particle transfer.

TPIC differs from APIC only in how the affine velocity matrix **C** is
updated during the G2P step:

* **APIC** — ``C = B @ inv(Dp)``  (B and Dp accumulated per particle)
* **TPIC** — ``C = Σ_i  v_i ⊗ ∇N_i``  (direct velocity-gradient via shape-fn
  derivatives)

P2G and the grid update are identical to APIC; only
:func:`build_g2p_tpic` differs.
"""

import jax.numpy as jnp
from jax import jit


def build_g2p_tpic(cfg):
    """Return a JIT-compiled TPIC grid-to-particle transfer function.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    g2p_tpic : callable
        ``(grid_v_out, v, x, base, fx, w, dw) -> (v_new, x_new, C_new)``

        * ``dw`` – shape-function gradients ``(3, n_p, 2)`` from
          :func:`~jaxmpm.shape_functions.build_shape_fn_with_grad`.
    """
    dt  = cfg.dt
    dim = 2

    @jit
    def g2p_tpic(grid_v_out, v, x, base, fx, w, dw):
        """Gather grid velocities to particles (TPIC formulation).

        Returns
        -------
        v_new : updated particle velocities  (PIC)
        x_new : updated particle positions   (advected with v_new)
        C_new : updated affine matrix via  C = Σ v_i ⊗ ∇N_i
        """
        n_p   = x.shape[0]
        new_v = jnp.zeros((n_p, dim))
        new_C = jnp.zeros((n_p, dim, dim))

        for i in range(3):
            for j in range(3):
                g_v    = grid_v_out[base[:, 0] + i, base[:, 1] + j]
                weight = w[i, :, 0] * w[j, :, 1]
                dw_x   = dw[i, :, 0] * w[j, :, 1]
                dw_y   = w[i, :, 0] * dw[j, :, 1]
                dweight = jnp.stack([dw_x, dw_y], axis=1)  # (n_p, 2)

                new_v += weight[:, None] * g_v
                new_C += jnp.einsum("ij,ik->ijk", g_v, dweight)

        return new_v, x + dt * new_v, new_C

    return g2p_tpic
