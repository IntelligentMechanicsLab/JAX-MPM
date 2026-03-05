"""
Dam-break validation against the analytical Ritter solution.

Simulates a 2-D dam-break on a frictionless bed and compares the computed
wave front with the analytical solution

    x_front(y, t) = L0 + [2√(g·H0) − 3√(g·y)] · t   (Ritter 1892)

Three particle-to-grid / grid-to-particle transfer schemes are supported:

  APIC  — Affine Particle-In-Cell (affine velocity field via B @ inv(Dp))
  TPIC  — Taylor PIC (affine velocity field via shape-function gradients)
  FLIP  — Fluid-Implicit-Particle (FLIP/PIC blend, separate force grid)

Usage
-----
  python run_validation.py --transfer apic          # default
  python run_validation.py --transfer flip --gif
  python run_validation.py --transfer tpic --no-display --gif

Output
------
  dam_break_<scheme>.gif  — animation (if --gif is set)
  results/<scheme>_t-<T>.png  — PNG snapshots every 0.01 s (if --save-png)
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Dam-break validation against the Ritter analytical solution."
)
parser.add_argument(
    "--transfer",
    choices=["apic", "flip", "tpic"],
    default="apic",
    help="Particle ↔ grid transfer scheme (default: apic).",
)
parser.add_argument(
    "--gif",
    action="store_true",
    help="Save an animated GIF of the simulation.",
)
parser.add_argument(
    "--save-png",
    action="store_true",
    help="Save PNG snapshots every 0.01 s of simulation time.",
)
parser.add_argument(
    "--no-display",
    action="store_true",
    help="Run fully off-screen (headless); forces --gif if omitted.",
)
args = parser.parse_args()

# Use float64 for FLIP/TPIC (greater accuracy for gradient-based quantities),
# float32 suffices for APIC.
use_f64 = args.transfer in ("flip", "tpic")
jax.config.update("jax_enable_x64", use_f64)

# ---------------------------------------------------------------------------
# Domain & discretisation
# ---------------------------------------------------------------------------

cal_x, cal_y = 2.0, 0.12          # computational domain (m)
n_grid_x     = 500
dh           = cal_x / n_grid_x
n_grid_y     = round(cal_y / dh)
inv_dh       = 1.0 / dh
dim          = 2

# Initial water column
real_x, real_y = 1.0, 0.1
each_el        = 2                 # particles per grid cell per direction
w_count = round(n_grid_x / (cal_x / real_x) * each_el)
d_count = round(n_grid_y / (cal_y / real_y) * each_el)

real_dx = real_x / w_count
real_dy = real_y / d_count

ii_grid, kk_grid = np.meshgrid(
    np.arange(w_count), np.arange(d_count), indexing="ij"
)
x_np = np.stack(
    [
        0.0 + (ii_grid + 0.5) * real_dx,
        0.0 + (kk_grid + 0.5) * real_dy,
    ],
    axis=-1,
).reshape(-1, 2)

n_particles = x_np.shape[0]

# Water-column geometry (for analytical solution)
left, right  = x_np[:, 0].min(), x_np[:, 0].max()
bottom, top  = x_np[:, 1].min(), x_np[:, 1].max()
L0  = float(right - left)   # initial front position
H0  = float(top - bottom)   # initial water height

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

dt          = 1e-5
real_time   = 0.6                  # total simulation time (s)
p_vol_orig  = (dh * 0.5) ** 2     # reference particle volume
p_rho_orig  = 1000.0              # reference density (kg/m³)
p_mass      = p_vol_orig * p_rho_orig
c           = 35.0                 # speed of sound (m/s)
gravity     = 9.8                  # (m/s²)
modulus     = c ** 2 * p_rho_orig  # bulk modulus

# ---------------------------------------------------------------------------
# Boundary masks  (wall nodes → zero normal velocity)
# ---------------------------------------------------------------------------

true_matrix = np.ones((n_grid_x + 1, n_grid_y + 1), dtype=bool)
mask_left   = (np.arange(n_grid_x + 1) * dh <= 0.0)[:, None] * true_matrix
mask_right  = (np.arange(n_grid_x + 1) * dh >= cal_x)[:, None] * true_matrix
mask_bottom = (np.arange(n_grid_y + 1) * dh <= 0.0)[None, :] * true_matrix

mask_left   = jnp.array(mask_left)
mask_right  = jnp.array(mask_right)
mask_bottom = jnp.array(mask_bottom)

# ---------------------------------------------------------------------------
# Shape functions (quadratic B-spline with boundary correction)
# ---------------------------------------------------------------------------

@jit
def _define_dtype(base):
    """Return per-particle stencil type: 0=interior, 1=left/bot, 2=right/top."""
    bx = jnp.where(base[:, 0] == -1, 1,
         jnp.where(base[:, 0] == n_grid_x - 1, 2, 0))
    by = jnp.where(base[:, 1] == -1, 1,
         jnp.where(base[:, 1] == n_grid_y - 1, 2, 0))
    return jnp.stack([bx, by], axis=0).T   # (n_p, 2)


def _build_pre_compute(need_dw: bool):
    """Build a JIT-compiled ``pre_compute(x)`` closure.

    Parameters
    ----------
    need_dw : bool
        If ``True``, also return shape-function gradients ``dw``.

    Returns
    -------
    pre_compute : callable
        ``(x) -> (base, fx, w)``  or  ``(x) -> (base, fx, w, dw)``
    """

    @jit
    def pre_compute(x):
        base = jnp.floor(x * inv_dh - 0.5).astype(jnp.int32)
        fx   = x * inv_dh - base

        # Standard quadratic B-spline
        w0 = jnp.array([
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ])   # (3, n_p, 2)

        # Near left/bottom (base == -1): node-0 weight → 0
        w1 = jnp.array([jnp.zeros_like(fx), 1.0 - (fx - 1.0), fx - 1.0])

        # Near right/top (base == n_grid - 1): node-2 weight → 0
        w2 = jnp.array([1.0 - fx, fx, jnp.zeros_like(fx)])

        dtype   = _define_dtype(base)   # (n_p, 2)
        n_p     = x.shape[0]

        all_wx  = jnp.stack((w0[:, :, 0], w1[:, :, 0], w2[:, :, 0]), axis=-1)
        all_wy  = jnp.stack((w0[:, :, 1], w1[:, :, 1], w2[:, :, 1]), axis=-1)
        sel_wx  = all_wx[:, jnp.arange(n_p), dtype[:, 0]]
        sel_wy  = all_wy[:, jnp.arange(n_p), dtype[:, 1]]
        w_all   = jnp.stack((sel_wx, sel_wy), axis=-1)   # (3, n_p, 2)

        if not need_dw:
            return base, fx, w_all

        # Shape-function derivatives (needed for FLIP / TPIC)
        dw_0 = -(1.5 - fx) * inv_dh
        dw_1 = -2.0 * (fx - 1.0) * inv_dh
        dw_2 = (fx - 0.5) * inv_dh

        dwx = jnp.array([dw_0[:, 0], dw_1[:, 0], dw_2[:, 0]])
        dwy = jnp.array([dw_0[:, 1], dw_1[:, 1], dw_2[:, 1]])
        dw0 = jnp.stack((dwx, dwy), axis=-1)   # (3, n_p, 2) — interior

        # Boundary-corrected dw
        dw1 = jnp.array([
            jnp.zeros_like(fx),
            -jnp.ones_like(fx) * inv_dh,
             jnp.ones_like(fx) * inv_dh,
        ])
        dw2 = jnp.array([
            -jnp.ones_like(fx) * inv_dh,
             jnp.ones_like(fx) * inv_dh,
             jnp.zeros_like(fx),
        ])

        all_dwx = jnp.stack((dw0[:, :, 0], dw1[:, :, 0], dw2[:, :, 0]), axis=-1)
        all_dwy = jnp.stack((dw0[:, :, 1], dw1[:, :, 1], dw2[:, :, 1]), axis=-1)
        sel_dwx = all_dwx[:, jnp.arange(n_p), dtype[:, 0]]
        sel_dwy = all_dwy[:, jnp.arange(n_p), dtype[:, 1]]
        dw_all  = jnp.stack((sel_dwx, sel_dwy), axis=-1)   # (3, n_p, 2)

        return base, fx, w_all, dw_all

    return pre_compute


# ---------------------------------------------------------------------------
# Boundary enforcement
# ---------------------------------------------------------------------------

@jit
def _apply_boundary(v_field):
    """Zero normal velocity on all four walls (no-slip left/right, slip bottom)."""
    vx = v_field[..., 0]
    vy = v_field[..., 1]
    vx = jnp.where(mask_left | mask_right, 0.0, vx)
    vy = jnp.where(mask_bottom, 0.0, vy)
    return jnp.stack((vx, vy), axis=-1)


# ===========================================================================
# APIC transfer
# ===========================================================================

@jit
def _p2g_apic(p_rho, v, x, C, pressure, base, fx, w):
    """APIC particle-to-grid with affine stress-momentum."""
    grid_v = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))
    grid_m = jnp.zeros((n_grid_x + 1, n_grid_y + 1))

    vol_strain = jnp.trace(C, axis1=1, axis2=2) * dt
    p_vol_prev = p_mass / p_rho
    dJ          = 1.0 + vol_strain
    p_vol_next  = p_vol_prev * dJ
    p_rho_next  = p_rho / dJ

    dp       = p_rho_next * modulus / p_rho_orig * vol_strain
    pressure = pressure - dp[:, None, None] * jnp.eye(2)
    stress   = -pressure   # mu = 0 → no deviatoric term

    stress_scaled = (-dt * 4.0 * p_vol_next[:, None, None] * inv_dh * inv_dh) * stress
    affine        = stress_scaled + p_mass * C

    for i in range(3):
        for j in range(3):
            offset = jnp.array([i, j])
            dpos   = (jnp.array([i, j], dtype=float) - fx) * dh
            weight = w[i, :, 0] * w[j, :, 1]
            contrib = weight[:, None] * (
                p_mass * v + jnp.einsum("ijk,ik->ij", affine, dpos)
            )
            idx    = base + offset
            grid_v = grid_v.at[idx[:, 0], idx[:, 1], :].add(contrib)
            grid_m = grid_m.at[idx[:, 0], idx[:, 1]].add(weight * p_mass)

    return p_rho_next, pressure, grid_v, grid_m


@jit
def _grid_op_apic(grid_v, grid_m):
    """Grid update for APIC: divide by mass, add gravity, enforce BCs."""
    inv_m = 1.0 / grid_m
    v_out = jnp.where(
        grid_m[:, :, None] > 0, inv_m[:, :, None] * grid_v, jnp.zeros_like(grid_v)
    )
    v_out = v_out.at[:, :, 1].add(-dt * gravity)
    return _apply_boundary(v_out)


@jit
def _g2p_apic(grid_v_out, v, x, base, fx, w):
    """APIC grid-to-particle: PIC velocity, C from B @ inv(Dp)."""
    n_p   = x.shape[0]
    new_v = jnp.zeros((n_p, dim))
    B     = jnp.zeros((n_p, dim, dim))
    Dp    = jnp.zeros((n_p, dim, dim))

    for i in range(3):
        for j in range(3):
            dpos   = jnp.array([i, j], dtype=float) - fx
            g_v    = grid_v_out[base[:, 0] + i, base[:, 1] + j]
            weight = w[i, :, 0] * w[j, :, 1]
            new_v += weight[:, None] * g_v
            B     += weight[:, None, None] * jnp.einsum("ij,ik->ijk", g_v, dpos) * dh
            Dp    += weight[:, None, None] * jnp.einsum("ij,ik->ijk", dpos, dpos) * dh * dh

    C_new = B @ jnp.linalg.inv(Dp)
    return new_v, x + dt * new_v, C_new


def _substep_apic(s, p_rho, v, x, C, pressure):
    base, fx, w = pre_compute(x)
    p_rho, pressure, grid_v, grid_m = _p2g_apic(p_rho, v, x, C, pressure, base, fx, w)
    grid_v_out = _grid_op_apic(grid_v, grid_m)
    v, x, C    = _g2p_apic(grid_v_out, v, x, base, fx, w)
    return p_rho, v, x, C, pressure


# ===========================================================================
# TPIC transfer
# ===========================================================================

@jit
def _p2g_tpic(p_rho, v, x, C, pressure, base, fx, w):
    """TPIC particle-to-grid — identical to APIC p2g (affine term with C)."""
    # Re-use APIC p2g; the only TPIC difference is in g2p (how C is computed).
    return _p2g_apic(p_rho, v, x, C, pressure, base, fx, w)


@jit
def _g2p_tpic(grid_v_out, v, x, base, fx, w, dw):
    """TPIC grid-to-particle: C updated via shape-function gradients (Taylor)."""
    n_p   = x.shape[0]
    new_v = jnp.zeros((n_p, dim))
    new_C = jnp.zeros((n_p, dim, dim))

    for i in range(3):
        for j in range(3):
            g_v    = grid_v_out[base[:, 0] + i, base[:, 1] + j]
            weight = w[i, :, 0] * w[j, :, 1]
            dw_x   = dw[i, :, 0] * w[j, :, 1]
            dw_y   = w[i, :, 0] * dw[j, :, 1]
            dweight = jnp.stack([dw_x, dw_y], axis=1)   # (n_p, 2)

            new_v += weight[:, None] * g_v
            new_C += jnp.einsum("ij,ik->ijk", g_v, dweight)

    return new_v, x + dt * new_v, new_C


def _substep_tpic(s, p_rho, v, x, C, pressure):
    base, fx, w, dw = pre_compute(x)
    p_rho, pressure, grid_v, grid_m = _p2g_tpic(p_rho, v, x, C, pressure, base, fx, w)
    grid_v_out = _grid_op_apic(grid_v, grid_m)   # same grid op as APIC
    v, x, C    = _g2p_tpic(grid_v_out, v, x, base, fx, w, dw)
    return p_rho, v, x, C, pressure


# ===========================================================================
# FLIP transfer
# ===========================================================================

@jit
def _p2g_flip(p_rho, v, x, Grad_v, pressure, base, fx, w, dw):
    """FLIP p2g: scatter pure momentum; internal force accumulated in grid_f."""
    grid_v  = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))
    grid_m  = jnp.zeros((n_grid_x + 1, n_grid_y + 1))
    grid_f  = jnp.zeros((n_grid_x + 1, n_grid_y + 1, dim))

    vol_strain = jnp.trace(Grad_v, axis1=1, axis2=2) * dt
    p_vol_prev = p_mass / p_rho
    dJ          = 1.0 + vol_strain
    p_vol_next  = p_vol_prev * dJ
    p_rho_next  = p_rho / dJ

    dp       = p_rho_next * modulus / p_rho_orig * vol_strain
    pressure = pressure - dp[:, None, None] * jnp.eye(2)
    stress   = -pressure   # mu = 0

    for i in range(3):
        for j in range(3):
            offset  = jnp.array([i, j])
            weight  = w[i, :, 0] * w[j, :, 1]
            dw_x    = dw[i, :, 0] * w[j, :, 1]
            dw_y    = w[i, :, 0] * dw[j, :, 1]
            dweight = jnp.stack([dw_x, dw_y], axis=1)   # (n_p, 2)

            idx = base + offset
            # Pure momentum scatter (no affine term)
            grid_v = grid_v.at[idx[:, 0], idx[:, 1], :].add(
                weight[:, None] * p_mass * v
            )
            grid_m = grid_m.at[idx[:, 0], idx[:, 1]].add(weight * p_mass)
            # Internal force via divergence theorem (stress · ∇N)
            internal_force = -p_vol_next[:, None] * jnp.einsum(
                "ijk,ik->ij", stress, dweight
            )
            grid_f = grid_f.at[idx[:, 0], idx[:, 1], :].add(internal_force)

    return p_rho_next, pressure, grid_v, grid_m, grid_f


@jit
def _grid_op_flip(grid_v, grid_m, grid_f):
    """FLIP grid op: returns both updated velocity and acceleration grids."""
    inv_m = 1.0 / grid_m
    v_out = jnp.where(
        grid_m[:, :, None] > 0,
        inv_m[:, :, None] * grid_v,
        jnp.zeros_like(grid_v),
    )
    a_out = jnp.where(
        grid_m[:, :, None] > 0,
        inv_m[:, :, None] * grid_f,
        jnp.zeros_like(grid_f),
    )
    # Apply gravity to acceleration, then update velocity
    a_out = a_out.at[:, :, 1].add(-gravity)
    v_out = v_out + a_out * dt

    grid_v_out = _apply_boundary(v_out)
    grid_a_out = _apply_boundary(a_out)
    return grid_v_out, grid_a_out


@jit
def _g2p_flip(grid_v_out, grid_a_out, v, x, base, fx, w, dw):
    """FLIP g2p: velocity via FLIP blend, position via PIC velocity."""
    n_p      = x.shape[0]
    new_vpic  = jnp.zeros((n_p, dim))
    new_vflip = v   # carry old velocity (FLIP: add acceleration increment)
    Grad_v    = jnp.zeros((n_p, dim, dim))

    for i in range(3):
        for j in range(3):
            g_v    = grid_v_out[base[:, 0] + i, base[:, 1] + j]
            g_a    = grid_a_out[base[:, 0] + i, base[:, 1] + j]
            weight = w[i, :, 0] * w[j, :, 1]
            dw_x   = dw[i, :, 0] * w[j, :, 1]
            dw_y   = w[i, :, 0] * dw[j, :, 1]
            dweight = jnp.stack([dw_x, dw_y], axis=1)

            new_vflip += weight[:, None] * g_a * dt   # FLIP: old_v + Δa·dt
            new_vpic  += weight[:, None] * g_v        # PIC velocity
            Grad_v    += jnp.einsum("ij,ik->ijk", g_v, dweight)

    # FLIP velocity for particles, but PIC velocity for stable advection
    return new_vflip, x + dt * new_vpic, Grad_v


def _substep_flip(s, p_rho, v, x, Grad_v, pressure):
    base, fx, w, dw = pre_compute(x)
    p_rho, pressure, grid_v, grid_m, grid_f = _p2g_flip(
        p_rho, v, x, Grad_v, pressure, base, fx, w, dw
    )
    grid_v_out, grid_a_out = _grid_op_flip(grid_v, grid_m, grid_f)
    v, x, Grad_v = _g2p_flip(grid_v_out, grid_a_out, v, x, base, fx, w, dw)
    return p_rho, v, x, Grad_v, pressure


# ===========================================================================
# Analytical Ritter solution  (wave-front polyline for plotting)
# ===========================================================================

def ritter_front(t, n_pts=100):
    """Return (x_front, y_front) polyline for the Ritter dam-break front.

    Parameters
    ----------
    t : float
        Simulation time (s).
    n_pts : int
        Number of points along the water depth.
    """
    y = np.linspace(bottom, top, n_pts)
    x = (2.0 * np.sqrt(gravity * H0) - 3.0 * np.sqrt(gravity * y)) * t + L0
    return x, y


# ===========================================================================
# PyVista visualisation helpers
# ===========================================================================

def _build_plotter(offscreen: bool):
    import pyvista as pv
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.set_background("white")
    plotter.set_scale(xscale=1, yscale=5, zscale=1)
    plotter.view_xy()
    plotter.add_axes(interactive=True)
    plotter.image_transparent_background = True

    # Domain outline
    points = [
        [0,     0,     0],
        [cal_x, 0,     0],
        [cal_x, cal_y, 0],
        [0,     cal_y, 0],
    ]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        import pyvista as pv_inner
        plotter.add_mesh(pv_inner.Line(points[a], points[b]), color="black", line_width=1)

    return plotter


def _make_particle_mesh(x_2d):
    import pyvista as pv
    z = np.zeros((x_2d.shape[0], 1))
    pts = np.hstack((np.array(x_2d), z))
    return pv.PolyData(pts)


def _make_front_polyline(t):
    import pyvista as pv
    xf, yf = ritter_front(t)
    # Add top horizontal segment
    x_top = np.linspace(left, xf.min(), 100)
    y_top = np.ones(100) * top
    pts   = np.vstack([
        np.column_stack([xf,    yf]),
        np.column_stack([x_top, y_top]),
    ])
    z     = np.zeros((pts.shape[0], 1))
    pts3d = np.hstack((pts, z))
    poly  = pv.PolyData(pts3d)
    N     = pts3d.shape[0]
    poly.lines = np.hstack([[N], np.arange(N)])
    return poly


# ===========================================================================
# Main simulation loop
# ===========================================================================

def run():
    scheme = args.transfer
    print(f"[dam_break_validation] transfer scheme: {scheme.upper()}")
    print(f"  n_particles = {n_particles},  dt = {dt},  real_time = {real_time} s")
    print(f"  float64 = {use_f64}")

    # ------------------------------------------------------------------
    # Select transfer functions
    # ------------------------------------------------------------------
    if scheme == "apic":
        pre_compute_fn = _build_pre_compute(need_dw=False)
        substep_fn = _substep_apic
    elif scheme == "tpic":
        pre_compute_fn = _build_pre_compute(need_dw=True)
        substep_fn = _substep_tpic
    else:  # flip
        pre_compute_fn = _build_pre_compute(need_dw=True)
        substep_fn = _substep_flip

    # Make pre_compute accessible to nested functions via module-level name
    global pre_compute
    pre_compute = pre_compute_fn

    # ------------------------------------------------------------------
    # Initialise state
    # ------------------------------------------------------------------
    x        = jnp.array(x_np)
    v        = jnp.zeros((n_particles, dim))
    p_rho    = jnp.ones(n_particles) * p_rho_orig
    pressure = jnp.zeros((n_particles, dim, dim))
    # C or Grad_v matrix (APIC/TPIC use C, FLIP uses Grad_v — same shape)
    state_mat = jnp.zeros((n_particles, dim, dim))

    # ------------------------------------------------------------------
    # Visualisation setup
    # ------------------------------------------------------------------
    offscreen = args.no_display
    import pyvista as pv

    plotter = _build_plotter(offscreen=offscreen)

    p_actor   = plotter.add_mesh(_make_particle_mesh(x), show_edges=False)
    f_actor   = plotter.add_mesh(_make_front_polyline(0.0), color="blue", line_width=0.4)
    txt_actor = plotter.add_text("Time: 0.00 s", position="upper_left", font_size=12, color="black")

    if not offscreen:
        plotter.show(interactive_update=True)

    gif_name = f"dam_break_{scheme}.gif"
    if args.gif or args.no_display:
        plotter.open_gif(gif_name)
        print(f"  Writing GIF → {gif_name}")

    if args.save_png:
        os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    step       = 0          # output frame counter (one frame per 0.01 s)
    step_count = 0          # total substep counter
    t          = 0.0
    start_wall = time.time()

    while t < real_time - 1e-12:
        t_target = 0.01 * (step + 1)

        # Advance to next frame
        while t < t_target - 1e-10:
            step_count += 1
            if scheme == "apic":
                p_rho, v, x, state_mat, pressure = substep_fn(
                    step_count, p_rho, v, x, state_mat, pressure
                )
            elif scheme == "tpic":
                p_rho, v, x, state_mat, pressure = substep_fn(
                    step_count, p_rho, v, x, state_mat, pressure
                )
            else:  # flip
                p_rho, v, x, state_mat, pressure = substep_fn(
                    step_count, p_rho, v, x, state_mat, pressure
                )
            t += dt

        step += 1

        # Update visualisation
        plotter.remove_actor(p_actor)
        plotter.remove_actor(f_actor)
        plotter.remove_actor(txt_actor)

        p_actor   = plotter.add_mesh(_make_particle_mesh(x), show_edges=False)
        f_actor   = plotter.add_mesh(_make_front_polyline(t), color="blue", line_width=0.4)
        txt_actor = plotter.add_text(
            f"Time: {t:.2f} s  [{scheme.upper()}]",
            position="upper_left", font_size=12, color="black",
        )

        if not offscreen:
            plotter.update()

        if args.gif or args.no_display:
            plotter.write_frame()

        if args.save_png:
            filename = f"results/{scheme}_t-{t:.3f}.png"
            plotter.screenshot(filename)

        elapsed = time.time() - start_wall
        print(f"  t = {t:.2f} s  ({step_count} steps, wall {elapsed:.1f} s)")

    plotter.close()
    print(f"\nDone. Total steps: {step_count}, wall time: {time.time()-start_wall:.1f} s")


if __name__ == "__main__":
    run()
