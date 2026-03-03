"""
Generate synthetic ground-truth data using the WLS-corrected MPM solver.

This script runs a forward simulation identical to ``generate_data.py`` except
that WLS boundary-corrected shape functions are used throughout.  The resulting
trajectory is saved separately so it can be used to train / compare with the
standard-kernel baseline.

Ground-truth friction layout (5 equal-width bands along *x*):

    ┌───────┬───────┬───────┬───────┬───────┐
    │ μ = 0 │ μ=0.5 │ μ=0.1 │ μ=0.2 │ μ = 0 │
    └───────┴───────┴───────┴───────┴───────┘
     0      0.4     0.8     1.2     1.6     2.0  (m)

Usage
-----
    python generate_data_wls.py

Output
------
    data/ground_truth_trajectory_wls.npy  – shape (n_frames, n_particles, 2)
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jaxmpm import MPMConfig, initialize_block_particles
from jaxmpm.solver_wls import build_solver_wls

# ──────────────────────────────────────────────────────────────────────
# 1. Configuration  (must match train_wls.py)
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
# 2. Particle initialisation
# ──────────────────────────────────────────────────────────────────────
x0, n_particles = initialize_block_particles(cfg, block_x=0.5, block_y=0.3)
print(f"Particles : {n_particles}")
print(f"Time-steps: {cfg.n_steps}")

v0        = jnp.zeros((n_particles, 2))
C0        = jnp.zeros((n_particles, 2, 2))
rho0_arr  = jnp.ones(n_particles) * cfg.rho0
pressure0 = jnp.zeros((n_particles, 2, 2))

# ──────────────────────────────────────────────────────────────────────
# 3. Ground-truth friction bands (same as generate_data.py)
# ──────────────────────────────────────────────────────────────────────
target_friction = jnp.array([0.0, 0.5, 0.1, 0.2])   # first 4 bands; 5th = 0
raw_friction    = jnp.log(jnp.exp(target_friction + 1e-8) - 1.0 + 1e-8)

# ──────────────────────────────────────────────────────────────────────
# 4. Run forward simulation with WLS solver
# ──────────────────────────────────────────────────────────────────────
simulate_wls = build_solver_wls(cfg, n_particles)

print("Compiling & running WLS forward simulation …")
t0 = time.time()
_, _, _, _, _, x_history = simulate_wls(
    rho0_arr, v0, x0, C0, pressure0, raw_friction
)
x_history.block_until_ready()
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f} s  |  Saved frames: {x_history.shape[0]}")

# ──────────────────────────────────────────────────────────────────────
# 5. Save
# ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "ground_truth_trajectory_wls.npy")
np.save(out_path, np.array(x_history))
print(f"Saved WLS ground-truth trajectory → {out_path}")
