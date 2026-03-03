# *JAX-MPM* — Differentiable Material Point Method in JAX

[![arXiv](https://img.shields.io/badge/arXiv-2507.04192-b31b1b.svg)](https://arxiv.org/abs/2507.04192)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/JAX-%E2%89%A50.4.20-blue.svg)](https://github.com/google/jax)

A general-purpose **differentiable meshfree solver** built on
[JAX](https://github.com/google/jax) and the
[Material Point Method (MPM)](https://en.wikipedia.org/wiki/Material_point_method),
enabling GPU-accelerated **forward and inverse modelling** of large
deformations, frictional contact, and inelastic materials — with emphasis on
geomechanics and geophysical hazard applications.

---

## Design Philosophy 🧠

- [x] **Fully differentiable** — gradients through the entire time-stepping
      pipeline via `jax.grad`, enabling gradient-based inverse modelling
- [x] **GPU-accelerated** — leverages JAX's XLA backend for high-throughput
      forward simulation
- [x] **Memory-efficient** — `jax.remat` + block-scan strategy keeps GPU memory
      bounded regardless of the number of time-steps
- [x] **Modular** — solver components (shape functions, transfers, BCs) are
      separate, composable building blocks
- [x] **Research-ready** — seamlessly couples with neural networks and deep
      learning models via JAX's ecosystem

---

## Installation ⚙️

```bash
git clone https://github.com/IntelligentMechanicsLab/DiffProg-JAX-MPM.git
cd DiffProg-JAX-MPM
pip install -r requirements.txt
```

> **Note:** For GPU support, install the appropriate
> [JAX GPU build](https://github.com/google/jax#installation) first.

---

## Repository Structure 📁

```
DiffProg-JAX-MPM/
├── jaxmpm/                         # Core solver library
│   ├── __init__.py
│   ├── config.py                   #   Simulation configuration (dataclass)
│   ├── particles.py                #   Particle initialisation
│   ├── shape_functions.py          #   Quadratic B-spline weights (boundary-aware)
│   ├── transfer.py                 #   P2G and G2P transfers (APIC)
│   ├── boundary.py                 #   Boundary conditions (Coulomb friction)
│   └── solver.py                   #   Time-stepping & scan-based integration
│
├── examples/                       # Reproducing paper results
│   └── dam_break_friction_inversion/   §5.1.3 — spatially varying friction
│       ├── README.md
│       ├── generate_data.py        #   Forward sim → ground truth
│       ├── train.py                #   Inverse solve
│       └── plot_results.py         #   Visualisation
│
├── requirements.txt
├── LICENSE
└── README.md                       # ← you are here
```

---

## Quick Start 🚀

### Forward simulation

```python
from jaxmpm import MPMConfig, initialize_block_particles, build_solver

cfg = MPMConfig(domain_x=2.0, domain_y=0.4, n_grid_x=200,
                dt=3e-5, total_time=0.4)

x0, n_p = initialize_block_particles(cfg, block_x=0.5, block_y=0.3)
simulate  = build_solver(cfg, n_p)

# friction_bands: raw values for 4 trainable bands (5th fixed to 0)
import jax.numpy as jnp
friction_raw = jnp.array([0.0, 0.5, 0.1, 0.2])

rho, v, x, C, P, x_hist = simulate(
    jnp.ones(n_p) * cfg.rho0,       # density
    jnp.zeros((n_p, 2)),            # velocity
    x0,                              # position
    jnp.zeros((n_p, 2, 2)),         # APIC matrix
    jnp.zeros((n_p, 2, 2)),         # pressure tensor
    friction_raw,                    # friction bands
)
```

### Inverse problem (gradient-based)

```python
import jax

@jax.jit
def loss_fn(friction_raw):
    _, _, _, _, _, x_hist = simulate(rho0, v0, x0, C0, P0, friction_raw)
    return jnp.sum(jnp.linalg.norm(x_hist - x_observed, axis=-1))

grads = jax.grad(loss_fn)(friction_raw)  # ← automatic differentiation!
```

See [`examples/dam_break_friction_inversion/`](examples/dam_break_friction_inversion/)
for a complete, runnable inverse-problem workflow.

---

## Examples 📖

| Example | Section | Description |
|---------|---------|-------------|
| [Dam-break friction inversion](examples/dam_break_friction_inversion/) | §5.1.3 | Recover spatially varying bottom friction from sparse Lagrangian observations |

*More examples will be added progressively.*

---

## Citation 🔥

If you use JAX-MPM in your research, please cite:

```bibtex
@article{du2025jax,
  title   = {JAX-MPM: A Learning-Augmented Differentiable Meshfree Framework
             for GPU-Accelerated Lagrangian Simulation and Geophysical
             Inverse Modeling},
  author  = {Du, Honghui and He, QiZhi},
  journal = {arXiv preprint arXiv:2507.04192},
  year    = {2025}
}
```

---

## Acknowledgement 👍

This work is developed at the
[Intelligent Mechanics Lab](https://github.com/IntelligentMechanicsLab).

---

## License

[MIT](LICENSE) © 2025 Honghui Du, QiZhi He

