"""Profile the paracrine induction rate k_p against synthetic trajectories.

At each fixed k_p, direct induction and total clearance are refitted and all
other parameters stay at their generating values. Admissible values have a
root mean square error within an assumed measurement precision. These ranges
describe practical identifiability under that criterion.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .model import IDX, Intervention, Params, simulate

# All profiles use the same logarithmic grid and include the no-feedback case k_p = 0.
KP_GRID_LO = 0.02        # multiples of nominal k_p
KP_GRID_HI = 128.0       # multiples of nominal k_p
KP_GRID_N = 64


def _default_grid(p: Params) -> np.ndarray:
    return np.concatenate([[0.0], p.k_p * np.logspace(
        np.log10(KP_GRID_LO), np.log10(KP_GRID_HI), KP_GRID_N - 1)])


def _summarise(kp_grid: np.ndarray, rmse: np.ndarray, noise_sd: float,
               p: Params) -> dict:
    """Summarise the admissible range and flag an upper endpoint at the grid limit."""
    admissible = kp_grid[rmse <= noise_sd]
    width = (float(admissible.max() - admissible.min()) / p.k_p
             if len(admissible) else 0.0)
    censored = bool(len(admissible) and (rmse[-1] <= noise_sd))
    # Fold range uses the positive admissible values.
    nz = admissible[admissible > 0.0]
    fold = float(nz.max() / nz.min()) if len(nz) else 0.0
    return {
        "admissible": admissible,
        "kp_ridge_width": width,
        "kp_ridge_fold": fold,
        # An admissible upper grid endpoint makes the range a lower bound.
        "kp_ridge_censored": censored,
        "kp_grid_top": float(kp_grid[-1]),
        "null_admissible": bool(len(admissible) and admissible.min() <= 1e-12),
    }


def _traj(p: Params, E: float, t: np.ndarray, t_off: float = np.inf) -> np.ndarray:
    y = simulate(p, Intervention(), E, t, t_off=t_off)
    return y[IDX["S1"]] + y[IDX["S2"]]


def profile_kp(p: Params, E: float = 1.0, t_end: float = 60.0, n_t: int = 61,
               kp_grid: np.ndarray | None = None,
               noise_sd: float = 0.02) -> dict:
    """Fit the continuous-exposure trajectory at each k_p and compare its RMSE with noise_sd."""
    t = np.linspace(0.0, t_end, n_t)
    truth = _traj(p, E, t)

    if kp_grid is None:
        kp_grid = _default_grid(p)

    rmse = np.zeros_like(kp_grid)
    fits = []
    for i, kp in enumerate(kp_grid):
        def resid(v):
            q = replace(p, k_p=float(kp), k_S=abs(v[0]),
                        gam_intrinsic=abs(v[1]), gam_immune=0.0)
            try:
                return _traj(q, E, t) - truth
            except Exception:
                return np.full_like(truth, 1e3)

        sol = least_squares(resid, [p.k_S, p.gamma],
                            bounds=([1e-5, 1e-5], [1.0, 1.0]),
                            xtol=1e-10, ftol=1e-10)
        rmse[i] = float(np.sqrt(np.mean(sol.fun ** 2)))
        fits.append((abs(sol.x[0]), abs(sol.x[1])))

    return {
        "kp_grid": kp_grid,
        "rmse": rmse,
        "fits": fits,
        "noise_sd": noise_sd,
        **_summarise(kp_grid, rmse, noise_sd, p),
    }


def profile_kp_withdrawal_only(p: Params, E: float = 1.0, t_off: float = 14.0,
                               t_end: float = 400.0, n_t: int = 81,
                               noise_sd: float = 0.02) -> dict:
    """Fit a withdrawal trajectory sampled on its own time grid."""
    t = np.linspace(0.0, t_end, n_t)
    truth = _traj(p, E, t, t_off=t_off)
    kp_grid = _default_grid(p)

    rmse = np.zeros_like(kp_grid)
    for i, kp in enumerate(kp_grid):
        def resid(v):
            q = replace(p, k_p=float(kp), k_S=abs(v[0]),
                        gam_intrinsic=abs(v[1]), gam_immune=0.0)
            try:
                return _traj(q, E, t, t_off=t_off) - truth
            except Exception:
                return np.full_like(truth, 1e3)

        sol = least_squares(resid, [p.k_S, p.gamma],
                            bounds=([1e-5, 1e-5], [1.0, 1.0]),
                            xtol=1e-10, ftol=1e-10)
        rmse[i] = float(np.sqrt(np.mean(sol.fun ** 2)))

    return {"kp_grid": kp_grid, "rmse": rmse, "noise_sd": noise_sd,
            **_summarise(kp_grid, rmse, noise_sd, p)}


def profile_kp_joint(p: Params, E: float = 1.0, t_off: float = 14.0,
                     t_cont_end: float = 60.0, n_cont: int = 61,
                     t_with_end: float = 400.0, n_with: int = 81,
                     noise_sd: float = 0.02) -> dict:
    """Fit the continuous and withdrawal trajectories with shared parameters.

    The pooled RMSE weights each observation equally, so a longer record
    contributes more to the criterion. In the per-arm summary, each arm must
    meet the assumed measurement precision separately.
    """
    t_c = np.linspace(0.0, t_cont_end, n_cont)
    t_w = np.linspace(0.0, t_with_end, n_with)
    truth_c = _traj(p, E, t_c)
    truth_w = _traj(p, E, t_w, t_off=t_off)
    kp_grid = _default_grid(p)

    rmse = np.zeros_like(kp_grid)
    rmse_c = np.zeros_like(kp_grid)
    rmse_w = np.zeros_like(kp_grid)
    for i, kp in enumerate(kp_grid):
        def resid(v):
            q = replace(p, k_p=float(kp), k_S=abs(v[0]),
                        gam_intrinsic=abs(v[1]), gam_immune=0.0)
            try:
                return np.concatenate([_traj(q, E, t_c) - truth_c,
                                       _traj(q, E, t_w, t_off=t_off) - truth_w])
            except Exception:
                return np.full(n_cont + n_with, 1e3)

        sol = least_squares(resid, [p.k_S, p.gamma],
                            bounds=([1e-5, 1e-5], [1.0, 1.0]),
                            xtol=1e-10, ftol=1e-10)
        rmse[i] = float(np.sqrt(np.mean(sol.fun ** 2)))
        rmse_c[i] = float(np.sqrt(np.mean(sol.fun[:n_cont] ** 2)))
        rmse_w[i] = float(np.sqrt(np.mean(sol.fun[n_cont:] ** 2)))

    out = {"kp_grid": kp_grid, "rmse": rmse, "rmse_continuous": rmse_c,
           "rmse_withdrawal": rmse_w, "noise_sd": noise_sd,
           **_summarise(kp_grid, rmse, noise_sd, p)}

    worst = np.maximum(rmse_c, rmse_w)
    per_arm = _summarise(kp_grid, worst, noise_sd, p)
    out.update({f"{k}_perarm": v for k, v in per_arm.items()})
    return out
