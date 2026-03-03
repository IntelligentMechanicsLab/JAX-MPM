# Dam-Break Friction Inversion

> **Inverse problem**: recover spatially varying bottom friction from sparse
> Lagrangian observations of a 2-D dam-break flow using differentiable MPM.

This example corresponds to **Section 5.1.3** of the paper:

> Du, H. & He, Q. (2025). *JAX-MPM: A Learning-Augmented Differentiable Meshfree
> Framework for GPU-Accelerated Lagrangian Simulation and Geophysical Inverse
> Modeling.* [arXiv:2507.04192](https://arxiv.org/abs/2507.04192)

---

## Problem Setup

A rectangular water column (0.5 m × 0.3 m) collapses under gravity inside a
2.0 m × 0.4 m domain.  The bottom wall has **piecewise-constant Coulomb friction**
divided into 5 equal-width vertical bands:

```
 ┌───────┬───────┬───────┬───────┬───────┐
 │ μ₀= 0 │ μ₁=0.5│ μ₂=0.1│ μ₃=0.2│ μ₄= 0 │   ← ground truth
 └───────┴───────┴───────┴───────┴───────┘
  0      0.4     0.8     1.2     1.6     2.0 m
```

Given only **sparse particle-position observations** (e.g. 100 or 500 out of
6 000 particles), the inverse solver recovers μ₀ – μ₃ by back-propagating
the position-matching loss through the entire MPM time-stepper via JAX's
automatic differentiation.

### Key features demonstrated

| Feature | Description |
|---------|-------------|
| **Differentiable MPM** | Gradients flow through P2G → grid-ops → G2P via `jax.grad` |
| **Memory-efficient** | `jax.remat` + block-scan keeps GPU memory constant |
| **Softplus reparameterisation** | Ensures friction ≥ 0 without box constraints |
| **Sparse observations** | Works with as few as 100 randomly sampled particles |

---

## Quick Start

### 1. Generate ground-truth data

```bash
cd examples/dam_break_friction_inversion
python generate_data.py
```

This runs the forward simulation with the known friction layout and saves the
trajectory to `data/ground_truth_trajectory.npy`.

### 2. Train (inverse solve)

```bash
python train.py --n_obs 500 --epochs 500 --lr 0.1
```

<details>
<summary>All CLI options</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--n_obs` | 500 | Number of observed particles |
| `--epochs` | 500 | Optimisation epochs |
| `--lr` | 0.1 | Initial Adam learning rate |
| `--decay_rate` | 0.9 | Exponential LR decay factor |
| `--decay_steps` | 20 | LR decay interval |
| `--seed` | 0 | RNG seed for observation sampling |
| `--data` | `data/ground_truth_trajectory.npy` | Path to ground truth |

</details>

Results (log + parameter checkpoints) are written to `results/obs500_lr0.1/`.

### 3. Plot

```bash
python plot_results.py --logs results/obs500_lr0.1/log.txt \
                       --labels '$N_s=500$'
```

Produces `loss_and_friction.png` showing the loss curve and parameter
convergence.

---

## File Structure

```
dam_break_friction_inversion/
├── generate_data.py      # Forward simulation → ground truth
├── train.py              # Gradient-based inverse solve
├── plot_results.py       # Loss & parameter visualisation
├── data/                 # Generated observation data
│   └── ground_truth_trajectory.npy
└── results/              # Training outputs
    └── obs500_lr0.1/
        ├── log.txt
        └── epoch_*_params.npy
```

---

## Simulation Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Domain size | $L_x \times L_y$ | 2.0 m × 0.4 m |
| Grid resolution | $n_x$ | 200 |
| Grid spacing | $\Delta h$ | 0.01 m |
| Time step | $\Delta t$ | 3 × 10⁻⁵ s |
| Simulation time | $T$ | 0.4 s |
| Density | $\rho_0$ | 1000 kg/m³ |
| Speed of sound | $c$ | 35 m/s |
| Particles per cell | — | 2 × 2 |
| Total particles | $N_p$ | ~6 000 |
