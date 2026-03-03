"""
Generate synthetic ground-truth data for the dam-break friction inversion example.

This script runs a **forward** MPM simulation of a 2-D dam break with spatially
varying (piecewise-constant) bottom friction.  The resulting particle trajectory
is saved to disk and later used as the observation target for the inverse solve.

Ground-truth friction layout (5 equal-width bands along *x*):

    ┌───────┬───────┬───────┬───────┬───────┐
    │ μ = 0 │ μ=0.5 │ μ=0.1 │ μ=0.2 │ μ = 0 │
    └───────┴───────┴───────┴───────┴───────┘
     0      0.4     0.8     1.2     1.6     2.0  (m)

Usage
-----
    python generate_data.py

Output
------
    data/ground_truth_trajectory.npy   – shape (n_saved_frames, n_particles, 2)
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Allow importing from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jaxmpm import MPMConfig, initialize_block_particles, build_solver

# ──────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────
cfg = MPMConfig(
    domain_x=2.0,
    domain_y=0.4,
    n_grid_x=200,
    dt=3e-5,
    total_time=0.4,
    rho0=1000.0,
    c=35.0,
    gravity=9.8,
    particles_per_cell_axis=2,
    save_every=10,
    block_size=500,
)

# ──────────────────────────────────────────────────────────────────────
# 2. Particle initialisation  (water column: 0.5 m × 0.3 m at origin)
# ──────────────────────────────────────────────────────────────────────
x0, n_particles = initialize_block_particles(cfg, block_x=0.5, block_y=0.3)
print(f"Number of particles: {n_particles}")
print(f"Number of time-steps: {cfg.n_steps}")

v0 = jnp.zeros((n_particles, 2))
C0 = jnp.zeros((n_particles, 2, 2))
rho0 = jnp.ones(n_particles) * cfg.rho0
pressure0 = jnp.zeros((n_particles, 2, 2))

# ──────────────────────────────────────────────────────────────────────
# 3. Ground-truth friction bands (raw → softplus → physical)
#    We store raw values whose softplus matches the target:
#        softplus(raw) ≈ target
#    For the two zero-friction bands we simply append 0.0 inside the solver.
#    The 4 trainable bands have targets [0.0, 0.5, 0.1, 0.2].
#    We compute the corresponding raw values via inverse-softplus.
# ──────────────────────────────────────────────────────────────────────
target_friction = jnp.array([0.0, 0.5, 0.1, 0.2])  # first 4 bands
# inverse softplus: raw = log(exp(target) - 1), clamp for target ≈ 0
raw_friction = jnp.log(jnp.exp(target_friction + 1e-8) - 1.0 + 1e-8)

# ──────────────────────────────────────────────────────────────────────
# 4. Run forward simulation
# ──────────────────────────────────────────────────────────────────────
simulate = build_solver(cfg, n_particles)

print("Compiling & running forward simulation …")
t0 = time.time()
_, _, _, _, _, x_history = simulate(rho0, v0, x0, C0, pressure0, raw_friction)
x_history.block_until_ready()
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f} s  |  Saved frames: {x_history.shape[0]}")

# ──────────────────────────────────────────────────────────────────────
# 5. Save
# ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "ground_truth_trajectory.npy")
np.save(out_path, np.array(x_history))
print(f"Saved ground-truth trajectory → {out_path}")
