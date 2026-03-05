"""Thin wrapper: build a WLS-corrected MPM solver.

``build_solver_wls(cfg, n_particles)`` is equivalent to calling
``build_solver(cfg, n_particles, use_wls=True)`` and is kept as a
convenience alias so that existing scripts importing from
``jaxmpm.solver_wls`` continue to work without modification.
"""

from jaxmpm.solver import build_solver


def build_solver_wls(cfg, n_particles):
    """Build a differentiable MPM time-stepper using WLS boundary-corrected
    shape functions.

    Parameters
    ----------
    cfg : MPMConfig
    n_particles : int

    Returns
    -------
    simulate : callable
        ``(F, v, x, C, friction_bands) -> (F, v, x, C, x_history)``
        Fully differentiable w.r.t. ``friction_bands``.
    """
    return build_solver(cfg, n_particles, use_wls=True)
