# Dam-break Validation — Transfer Scheme Comparison

This example runs a **forward-only 2-D dam-break simulation** and compares the computed wave front with the classical **Ritter (1892) analytical solution**:

$$x_{\text{front}}(y,t) = L_0 + \left[2\sqrt{g H_0} - 3\sqrt{g\, y}\right]\, t$$

Three particle ↔ grid transfer schemes are supported and can be compared head-to-head:

| Flag | Scheme | Velocity state | Position update |
|------|--------|---------------|-----------------|
| `apic` | **APIC** — Affine PIC | `C` via $\mathbf{B}\,\mathbf{D}_p^{-1}$ | PIC ($x + \Delta t\,v_{\rm pic}$) |
| `tpic` | **TPIC** — Taylor PIC | `C` via $\sum \mathbf{g}_v \otimes \nabla N$ | PIC |
| `flip` | **FLIP** — Fluid-Implicit-Particle | `Grad_v` via $\sum \mathbf{g}_v \otimes \nabla N$ | PIC; $v$ = FLIP blend |

---

## Setup

The computational domain is **2.0 × 0.12 m** with 500 grid cells along *x*.  
The initial water column occupies **1.0 × 0.1 m** (bottom-left corner).

| Parameter | Value |
|-----------|-------|
| Domain | 2.0 m × 0.12 m |
| Grid cells | 500 × 30 |
| Cell size `dh` | 4 mm |
| Particles per cell | 2 × 2 |
| `n_particles` | ≈ 3 000 |
| `dt` | 1 × 10⁻⁵ s |
| Simulation time | 0.6 s |
| Speed of sound | 35 m/s |
| Bulk modulus | 1.225 MPa |
| Gravity | 9.8 m/s² |

Boundary conditions: **no-slip** walls (left/right), **free-slip** floor, free surface on top.

---

## Usage

```bash
# APIC (default)
python run_validation.py

# FLIP with GIF output
python run_validation.py --transfer flip --gif

# TPIC, headless (no display), save GIF + PNG snapshots
python run_validation.py --transfer tpic --no-display --gif --save-png
```

### Options

| Flag | Description |
|------|-------------|
| `--transfer {apic,flip,tpic}` | Transfer scheme (default: `apic`) |
| `--gif` | Save animated GIF |
| `--save-png` | Save PNG snapshot every 0.01 s to `results/` |
| `--no-display` | Off-screen (headless) rendering; implies GIF |

---

## Output

- **Live PyVista window** — particles (grey dots) + Ritter wave front (blue line)
- **`dam_break_<scheme>.gif`** — animation (if `--gif`)
- **`results/<scheme>_t-<T>.png`** — PNG frames (if `--save-png`)

---

## Notes on transfer schemes

### APIC
The affine velocity matrix **C** is reconstructed each step as
$C = B\,D_p^{-1}$ where $B = \sum w_{ip}\,\mathbf{v}_{i}\otimes(\mathbf{x}_i - \mathbf{x}_p)$ and $D_p = \sum w_{ip}\,(\mathbf{x}_i-\mathbf{x}_p)\otimes(\mathbf{x}_i-\mathbf{x}_p)$.
This is the standard quadratic APIC of Jiang et al. (2015).

### TPIC
Uses shape-function gradients instead of the $\mathbf{B}$-matrix:
$C \mathrel{+}= \sum_{ij} \mathbf{v}_{ij} \otimes \nabla N_{ij}$.
The p2g transfer is identical to APIC; only the C-update in g2p differs.

### FLIP
Internal forces are accumulated in a separate grid vector `grid_f` via the weak form $f_i = -\sum_p V_p\,\boldsymbol{\sigma}_p\,\nabla N_{ip}$.
In g2p the particle velocity is updated with the **FLIP blend**
$v_p^{n+1} = v_p^n + \Delta t\,\tilde{a}_p$ (old velocity + grid acceleration increment),
while the position advection uses the **PIC velocity** for stability.

---

## Reference

* Ritter, A. (1892). *Die Fortpflanzung der Wasserwellen*. ZVDI, 36(33):947–954.  
* Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). *The affine particle-in-cell method*. ACM TOG, 34(4).
