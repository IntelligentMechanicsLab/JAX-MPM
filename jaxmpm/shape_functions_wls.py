"""
Quadratic B-spline shape functions with WLS (Weighted Least Squares) boundary
correction.

Unlike ``shape_functions.py``, which switches to linear weights near grid
boundaries, this module uses the **pure quadratic B-spline everywhere** and
applies a multiplicative WLS correction to enforce the partition-of-unity and
linear-completeness conditions near the domain walls.

The correction follows Soga et al. (2016), Equations 57–62:

    Φ_ij(x_p)  = [c₀ + c₁(x_i−x_p) + c₂(y_i−y_p)] · φ_ij(x_p)     (weight)
    ∇Φ_ij(x_p) = [C₂ + C₃ · r_ij] · φ_ij(x_p)                       (gradient)

where (c₀, c₁, c₂) and (C₂, C₃) are obtained by inverting the 3×3 WLS moment
matrix assembled from the standard shape function values.

References
----------
Soga, K., Alonso, E., Yerro, A., Kumar, K., Bandara, S. (2016).
"Trends in large-deformation analysis of landslide mass movements with
particular emphasis on the material point method."
Géotechnique, 66(3), 248–273.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import jit


def build_shape_fn_wls(cfg):
    """Return two JIT-compiled callables for WLS-corrected shape functions.

    Parameters
    ----------
    cfg : MPMConfig

    Returns
    -------
    compute_weights : callable
        ``compute_weights(x) -> (base, fx, w, dw)``

        *Pure* quadratic B-spline values and gradients.  No boundary switching.

        * ``base`` – integer base-node index, shape ``(n_p, 2)``
        * ``fx``   – fractional cell position, shape ``(n_p, 2)``
        * ``w``    – weights, shape ``(3, n_p, 2)``
        * ``dw``   – weight gradients [m⁻¹], shape ``(3, n_p, 2)``

    compute_wls : callable
        ``compute_wls(base, x, w) -> (nb_mask, c0, cx, cy)``

        WLS weight-correction data.

        * ``nb_mask`` – ``(n_p,)`` bool: True if particle touches a wall node
        * ``c0, cx, cy`` – ``(n_p,)`` WLS weight-correction scalars

        Gradient-correction blocks (C2, C3) are intentionally omitted: the
        inverted WLS moment matrix is ill-conditioned at domain corners (where
        only 2×2 stencil nodes are active), causing the corrected gradients to
        be numerically enormous and the explicit P2G force to blow up.
        Standard B-spline gradients are used for the force term instead.
    """
    n_grid_x = cfg.n_grid_x
    n_grid_y = cfg.n_grid_y
    inv_dh   = cfg.inv_dh
    dh       = cfg.dh
    domain_x = cfg.domain_x
    domain_y = cfg.domain_y

    # --- Pre-compute the fixed wall-node mask (shape: n_grid_x+1, n_grid_y+1) ---
    ones        = jnp.ones((n_grid_x + 1, n_grid_y + 1), dtype=bool)
    mask_left   = (jnp.arange(n_grid_x + 1) * dh <= 0.0)[:, None] * ones
    mask_right  = (jnp.arange(n_grid_x + 1) * dh >= domain_x)[:, None] * ones
    mask_bottom = (jnp.arange(n_grid_y + 1) * dh <= 0.0)[None, :] * ones
    mask_top    = (jnp.arange(n_grid_y + 1) * dh >= domain_y)[None, :] * ones
    is_boundary = mask_left | mask_right | mask_bottom | mask_top  # (nx+1, ny+1)

    # ------------------------------------------------------------------
    @jit
    def compute_weights(x):
        """Pure quadratic B-spline, no boundary type-switching."""
        base = jnp.floor(x * inv_dh - 0.5).astype(jnp.int32)  # (n_p, 2)
        fx   = x * inv_dh - base                               # (n_p, 2)
        t    = fx

        w = jnp.stack([
            0.5 * (1.5 - t) ** 2,          # node offset 0
            0.75 - (t - 1.0) ** 2,         # node offset 1
            0.5 * (t - 0.5) ** 2,          # node offset 2
        ], axis=0)  # (3, n_p, 2)

        # d(w)/d(x_p) = d(w)/d(t) * d(t)/d(x_p) = d(w)/d(t) * inv_dh
        # Note: t = x*inv_dh - base, so dt/dx_p = inv_dh
        dw = jnp.stack([
            -(1.5 - t) * inv_dh,           # d(w0)/dx_p
            -2.0 * (t - 1.0) * inv_dh,     # d(w1)/dx_p
             (t - 0.5) * inv_dh,           # d(w2)/dx_p
        ], axis=0)  # (3, n_p, 2)

        return base, fx, w, dw

    # ------------------------------------------------------------------
    @jit
    def compute_wls(base, x, w):
        """Build the WLS moment matrix and solve for correction coefficients.

        Particles far from any wall get identity corrections (nb_mask=False),
        but the coefficients are still computed for all particles to keep the
        computation XLA-static (no data-dependent branching).
        """
        n_p = x.shape[0]

        # ---- 1. Near-boundary mask ----
        nb = jnp.zeros(n_p, dtype=bool)
        for i in range(3):
            for j in range(3):
                ii  = base[:, 0] + i
                jj  = base[:, 1] + j
                act = (ii >= 0) & (ii <= n_grid_x) & (jj >= 0) & (jj <= n_grid_y)
                iic = jnp.clip(ii, 0, n_grid_x)
                jjc = jnp.clip(jj, 0, n_grid_y)
                nb  = nb | (is_boundary[iic, jjc] & act)

        # ---- 2. WLS moment matrix A (3×3 per particle) ----
        # A = [[S0,  Sx,  Sy ],
        #      [Sx,  Sxx, Sxy],
        #      [Sy,  Sxy, Syy]]
        # where S0 = Σ φ_ij, Sx = Σ φ_ij r_x, …  and r = x_node - x_p
        S0  = jnp.zeros(n_p)
        Sx  = jnp.zeros(n_p)
        Sy  = jnp.zeros(n_p)
        Sxx = jnp.zeros(n_p)
        Sxy = jnp.zeros(n_p)
        Syy = jnp.zeros(n_p)

        for i in range(3):
            for j in range(3):
                ii  = base[:, 0] + i
                jj  = base[:, 1] + j
                act = (ii >= 0) & (ii <= n_grid_x) & (jj >= 0) & (jj <= n_grid_y)
                am  = act.astype(jnp.float64)   # 0/1 active mask

                phi = w[i, :, 0] * w[j, :, 1]                   # (n_p,)
                r   = (base + jnp.array([i, j])) * dh - x       # (n_p,2) x_node - x_p
                rx, ry = r[:, 0], r[:, 1]

                S0  = S0  + phi * am
                Sx  = Sx  + phi * rx * am
                Sy  = Sy  + phi * ry * am
                Sxx = Sxx + phi * rx * rx * am
                Sxy = Sxy + phi * rx * ry * am
                Syy = Syy + phi * ry * ry * am

        A = jnp.stack([
            jnp.stack([S0,  Sx,  Sy ], axis=-1),
            jnp.stack([Sx,  Sxx, Sxy], axis=-1),
            jnp.stack([Sy,  Sxy, Syy], axis=-1),
        ], axis=-2)  # (n_p, 3, 3)
        A = A + 1e-8 * jnp.eye(3)[None]  # Tikhonov regularisation

        # ---- 3. Weight-correction scalars (c0, cx, cy) ----
        # Solve A @ [c0, cx, cy]^T = e_0 = [1, 0, 0]^T  for each particle.
        b     = jnp.zeros((n_p, 3)).at[:, 0].set(1.0)
        # b[..., None] avoids the "batched 1D solve" deprecation warning
        coeff = jsl.solve(A, b[..., None])[..., 0]   # (n_p, 3)
        c0, cx_c, cy_c = coeff[:, 0], coeff[:, 1], coeff[:, 2]

        # NOTE: gradient-correction blocks (C2, C3 from inv(A)) are intentionally
        # omitted.  Near domain corners only 2×2 stencil nodes are active; inv(A)
        # becomes numerically large, producing enormous corrected gradients that
        # destabilise the explicit force scatter in P2G.
        # The weight-correction scalars (c0, cx, cy) alone already enforce the
        # partition-of-unity condition near walls and are well-conditioned.
        return nb, c0, cx_c, cy_c

    return compute_weights, compute_wls
