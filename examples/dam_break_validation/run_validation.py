"""
Dam-break validation against the analytical Ritter solution.

Simulates a 2-D dam-break on a frictionless bed and compares the computed
wave front with the analytical solution

    x_front(y, t) = L0 + [2√(g·H0) − 3√(g·y)] · t   (Ritter 1892)

Three transfer schemes are supported (all from ``jaxmpm``):

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
  dam_break_<scheme>.gif        — animation (if --gif is set)
  results/<scheme>_t-<T>.png   — PNG snapshots every 0.01 s (if --save-png)
"""

import argparse
import os
import sys
import time

# Allow ``import jaxmpm`` when running from the example directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp
import numpy as np

# All physics builders live in the jaxmpm package
from jaxmpm import MPMConfig
from jaxmpm.shape_functions import build_shape_fn, build_shape_fn_with_grad
from jaxmpm.transfer import build_p2g, build_g2p
from jaxmpm.transfer_tpic import build_g2p_tpic
from jaxmpm.transfer_flip import build_p2g_flip, build_grid_op_flip, build_g2p_flip
from jaxmpm.boundary import build_grid_op_frictionless

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
    help="Run fully off-screen (headless). Also enables --gif by default.",
)
args = parser.parse_args()

# float64 for all schemes
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Simulation configuration  (via MPMConfig)
# ---------------------------------------------------------------------------

cfg = MPMConfig(
    domain_x   = 2.0,
    domain_y   = 0.12,
    n_grid_x   = 500,
    dt         = 1e-5,
    total_time = 0.6,
    rho0       = 1000.0,
    c          = 35.0,
    gravity    = 9.8,
    particles_per_cell_axis = 2,
)

# ---------------------------------------------------------------------------
# Particle initialisation (initial water column: 1.0 m × 0.1 m)
# ---------------------------------------------------------------------------

real_x, real_y = 1.0, 0.1
w_count = round(cfg.n_grid_x / (cfg.domain_x / real_x) * cfg.particles_per_cell_axis)
d_count = round(cfg.n_grid_y / (cfg.domain_y / real_y) * cfg.particles_per_cell_axis)

ii_grid, kk_grid = np.meshgrid(
    np.arange(w_count), np.arange(d_count), indexing="ij"
)
x_np = np.stack(
    [0.0 + (ii_grid + 0.5) * (real_x / w_count),
     0.0 + (kk_grid + 0.5) * (real_y / d_count)],
    axis=-1,
).reshape(-1, 2)

n_particles = x_np.shape[0]

# Water-column geometry for Ritter solution
left,   right  = float(x_np[:, 0].min()), float(x_np[:, 0].max())
bottom, top    = float(x_np[:, 1].min()), float(x_np[:, 1].max())
L0 = right - left   # initial dam-front x-position
H0 = top - bottom   # initial water height

# ---------------------------------------------------------------------------
# Analytical Ritter (1892) wave-front
# ---------------------------------------------------------------------------

def ritter_front(t, n_pts=100):
    """Return (x_front, y_front) polyline for the Ritter dam-break solution."""
    y = np.linspace(bottom, top, n_pts)
    x = (2.0 * np.sqrt(cfg.gravity * H0) - 3.0 * np.sqrt(cfg.gravity * y)) * t + L0
    return x, y


# ---------------------------------------------------------------------------
# Build transfer functions from jaxmpm
# ---------------------------------------------------------------------------

scheme = args.transfer

if scheme == "apic":
    compute_weights = build_shape_fn(cfg)
    p2g             = build_p2g(cfg)
    g2p             = build_g2p(cfg)
    grid_op         = build_grid_op_frictionless(cfg)

    def substep(s, F, v, x, C):
        base, fx, w = compute_weights(x)
        F, grid_v, grid_m = p2g(F, v, x, C, base, fx, w)
        grid_v_out = grid_op(grid_v, grid_m)
        v, x, C   = g2p(grid_v_out, v, x, base, fx, w)
        return F, v, x, C

elif scheme == "tpic":
    compute_weights = build_shape_fn_with_grad(cfg)
    p2g             = build_p2g(cfg)       # P2G identical to APIC
    g2p_tpic        = build_g2p_tpic(cfg)
    grid_op         = build_grid_op_frictionless(cfg)

    def substep(s, F, v, x, C):
        base, fx, w, dw = compute_weights(x)
        F, grid_v, grid_m = p2g(F, v, x, C, base, fx, w)
        grid_v_out = grid_op(grid_v, grid_m)
        v, x, C   = g2p_tpic(grid_v_out, v, x, base, fx, w, dw)
        return F, v, x, C

else:  # flip
    compute_weights = build_shape_fn_with_grad(cfg)
    p2g_flip        = build_p2g_flip(cfg)
    grid_op_flip    = build_grid_op_flip(cfg)
    g2p_flip        = build_g2p_flip(cfg)

    def substep(s, F, v, x, Grad_v):
        base, fx, w, dw = compute_weights(x)
        F, grid_v, grid_m, grid_f = p2g_flip(F, v, x, Grad_v, base, fx, w, dw)
        grid_v_out, grid_a_out = grid_op_flip(grid_v, grid_m, grid_f)
        v, x, Grad_v = g2p_flip(grid_v_out, grid_a_out, v, x, base, fx, w, dw)
        return F, v, x, Grad_v


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

x         = jnp.array(x_np)
v         = jnp.zeros((n_particles, 2))
F         = jnp.tile(jnp.eye(2), (n_particles, 1, 1))  # deformation gradient
state_mat = jnp.zeros((n_particles, 2, 2))   # C (APIC/TPIC) or Grad_v (FLIP)

# ---------------------------------------------------------------------------
# Matplotlib frame renderer  (reliable headless GIF via imageio)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")          # non-interactive, no display required
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

# Visual aspect: domain 2.0 × 0.12, but water column is only 0.1 m tall;
# stretch y-axis ×5 so the thin layer is readable.
_Y_SCALE   = 5.0
_FIG_W     = 12.0                          # inches
_FIG_H     = _FIG_W * (cfg.domain_y * _Y_SCALE) / cfg.domain_x


def _render_frame(x_2d, t):
    """Render one GIF frame with matplotlib and return a NumPy RGBA array."""
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), dpi=100)

    # Particles
    xp = np.array(x_2d)
    ax.scatter(xp[:, 0], xp[:, 1] * _Y_SCALE,
               s=0.4, c="steelblue", alpha=0.8, linewidths=0)

    # Ritter analytical front (red)
    xf, yf = ritter_front(t)
    ax.plot(xf, yf * _Y_SCALE, color="red", linewidth=1.5,
            label="Ritter (1892)")
    # Top horizontal segment of the front
    x_top = np.linspace(left, float(xf.min()), 80)
    ax.plot(x_top, np.ones(80) * top * _Y_SCALE,
            color="red", linewidth=1.5)

    # Domain box
    for (x0, y0), (x1, y1) in [
        ((0, 0), (cfg.domain_x, 0)),
        ((cfg.domain_x, 0), (cfg.domain_x, cfg.domain_y * _Y_SCALE)),
        ((cfg.domain_x, cfg.domain_y * _Y_SCALE), (0, cfg.domain_y * _Y_SCALE)),
        ((0, cfg.domain_y * _Y_SCALE), (0, 0)),
    ]:
        ax.plot([x0, x1], [y0, y1], "k-", linewidth=0.8)

    ax.set_xlim(-0.02 * cfg.domain_x, 1.02 * cfg.domain_x)
    ax.set_ylim(-0.05 * cfg.domain_y * _Y_SCALE,
                 1.10 * cfg.domain_y * _Y_SCALE)
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel(f"y × {_Y_SCALE:.0f}  (m)", fontsize=10)
    ax.set_title(f"Dam-break [{scheme.upper()}]   t = {t:.2f} s", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_aspect("equal")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


# ---------------------------------------------------------------------------
# Optional PyVista interactive display  (only when --no-display is NOT set)
# ---------------------------------------------------------------------------

def _build_pv_plotter():
    import pyvista as pv
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.set_scale(xscale=1, yscale=_Y_SCALE, zscale=1)
    plotter.view_xy()
    plotter.add_axes(interactive=True)
    pts = [[0, 0, 0], [cfg.domain_x, 0, 0],
           [cfg.domain_x, cfg.domain_y, 0], [0, cfg.domain_y, 0]]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        plotter.add_mesh(pv.Line(pts[a], pts[b]), color="black", line_width=1)
    return plotter


def _pv_particle_mesh(x_2d):
    import pyvista as pv
    z   = np.zeros((x_2d.shape[0], 1))
    return pv.PolyData(np.hstack((np.array(x_2d), z)))


def _pv_front_polyline(t):
    import pyvista as pv
    xf, yf = ritter_front(t)
    x_top  = np.linspace(left, float(xf.min()), 80)
    pts    = np.vstack([np.column_stack([xf, yf]),
                        np.column_stack([x_top, np.ones(80) * top])])
    pts3d  = np.hstack((pts, np.zeros((pts.shape[0], 1))))
    poly   = pv.PolyData(pts3d)
    N      = pts3d.shape[0]
    poly.lines = np.hstack([[N], np.arange(N)])
    return poly


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print(f"[dam_break_validation] transfer = {scheme.upper()},  "
      f"n_particles = {n_particles},  dt = {cfg.dt},  "
      f"real_time = {cfg.total_time} s")

gif_name   = f"dam_break_{scheme}.gif"
write_gif  = args.gif or args.no_display
gif_writer = None

if write_gif:
    gif_writer = imageio.get_writer(gif_name, mode="I", fps=10,
                                    loop=0)          # loop=0 → infinite loop
    print(f"  Writing GIF → {gif_name}")

if args.save_png:
    os.makedirs("results", exist_ok=True)

# PyVista interactive window (only when NOT headless)
pv_plotter = p_actor = f_actor = txt_actor = None
if not args.no_display:
    pv_plotter = _build_pv_plotter()
    p_actor    = pv_plotter.add_mesh(_pv_particle_mesh(x), show_edges=False)
    f_actor    = pv_plotter.add_mesh(_pv_front_polyline(0.0), color="red",
                                      line_width=1.5)
    txt_actor  = pv_plotter.add_text("Time: 0.00 s", position="upper_left",
                                      font_size=12, color="black")
    pv_plotter.show(interactive_update=True)

step       = 0
step_count = 0
t          = 0.0
start_wall = time.time()

while t < cfg.total_time - 1e-12:
    t_target = 0.01 * (step + 1)

    while t < t_target - 1e-10:
        step_count += 1
        F, v, x, state_mat = substep(step_count, F, v, x, state_mat)
        t += cfg.dt

    step += 1

    # --- matplotlib GIF frame ---
    if write_gif:
        frame = _render_frame(x, t)
        gif_writer.append_data(frame)

    # --- PNG snapshot ---
    if args.save_png:
        frame = _render_frame(x, t)
        imageio.imwrite(f"results/{scheme}_t-{t:.3f}.png", frame)

    # --- PyVista interactive update ---
    if pv_plotter is not None:
        pv_plotter.remove_actor(p_actor)
        pv_plotter.remove_actor(f_actor)
        pv_plotter.remove_actor(txt_actor)
        p_actor   = pv_plotter.add_mesh(_pv_particle_mesh(x), show_edges=False)
        f_actor   = pv_plotter.add_mesh(_pv_front_polyline(t), color="red",
                                         line_width=1.5)
        txt_actor = pv_plotter.add_text(
            f"Time: {t:.2f} s  [{scheme.upper()}]",
            position="upper_left", font_size=12, color="black",
        )
        pv_plotter.update()

    elapsed = time.time() - start_wall
    print(f"  t = {t:.2f} s  ({step_count} steps,  wall {elapsed:.1f} s)")

if gif_writer is not None:
    gif_writer.close()
if pv_plotter is not None:
    pv_plotter.close()
print(f"\nDone.  Total steps: {step_count},  wall time: {time.time()-start_wall:.1f} s")
