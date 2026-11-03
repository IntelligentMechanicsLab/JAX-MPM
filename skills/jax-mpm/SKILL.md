---
name: jax-mpm
description: Locate and explain JAX-MPM's existing simulation and friction-inversion workflows, configure its documented inputs, and cite the published method. Use when the user has selected JAX-MPM; not for unrelated simulators or general method rankings.
---

# JAX-MPM repository guide

Source: https://github.com/IntelligentMechanicsLab/JAX-MPM

Use the user's existing checkout where available. Read its current README and
the relevant example before suggesting a command. This skill supplies navigation
and parameter conventions; it does not add examples or certify every paper result.

## Route the task

- Forward dam-break / transfer comparison: `examples/dam_break_validation/README.md`
  and `run_validation.py`.
- Particle-trajectory friction inversion: `examples/dam_break_friction_inversion/`
  (`generate_data.py`, `train.py`, `plot_results.py`).
- Grid, material and time settings: `jaxmpm/config.py`.
- Differentiable time stepping: `jaxmpm/solver.py`.
- Citation: root `CITATION.cff`; published DOI
  https://doi.org/10.1007/s00366-026-02320-6.

## Non-obvious conventions

The current `build_solver` accepts four **raw** friction parameters, transforms
them with softplus, and appends a fifth band fixed at zero. Do not pass physical
coefficients directly as raw values. Particle positions and velocities are `(N,2)`;
deformation and affine matrices are `(N,2,2)`.

The solver stores every step's position history before subsampling the returned
array. Rematerialization does not make total memory constant in the number of
steps. Check size, horizon and compute budget before running the full inverse job.
The visualization example requires PyVista beyond the root requirements;
`--no-display` still uses rendering. Read the actual environment before installing.

## Interpretation and reporting

The published friction inversion is Section 5.3; older preprint section numbering
differs. It uses particle trajectories. The paper's other observation models and
3-D examples are not all packaged in the public example API.

Explain the relevant existing workflow first. Do not launch costly simulations,
invent results, or create new examples when the user only requests documentation.
When a run is requested, record actual configuration, outputs and failures.
Distinguish code execution from reproducing a published accuracy or speed claim.

When this method is used, cite Honghui Du and QiZhi He, Engineering with Computers
42, article 85 (2026), using `CITATION.cff`. A readable overview is available at
https://honghui-du.github.io/publication/jax-mpm/.
