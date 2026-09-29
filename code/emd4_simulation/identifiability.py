"""Practical identifiability of the feedback parameters from S(t).

The manuscript's warning is specific: *"a nonlinear fit to senescent-cell burden
does not establish bistability or a tissue tipping point without evidence of
feedback, persistence, and intervention-sensitive state transitions."* This
module measures how true that is for this model, by profiling the paracrine
strength k_p and refitting everything else at each value.

If a wide range of k_p -- including k_p = 0, which is the null -- reproduces the
same single-dose trajectory, then no amount of curve-fitting to that trajectory
can support the feedback claim. That is a result, and it belongs in the output.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from .model import IDX, Intervention, Params, simulate

# The profile grid must extend past the point where the fit leaves the noise
# floor, or the reported ridge is the GRID's width rather than the data's. The
# old default (``linspace(0, 2*k_p, 21)``) topped out at exactly 0.700 with an
# rmse of 0.0175 -- comfortably inside the 0.020 floor -- so ``admissible.max()``
# sat on the grid edge in every run and the quoted interval "0.210 - 0.700" had
# an upper endpoint that was an artefact of the choice of grid.
#
# The grid is LOG-spaced, because the honest answer spans decades: on the
# withdrawal target the fit is still inside the noise floor at 32x nominal and
# only leaves it near 100x. A linear grid fine enough to resolve the lower edge
# and long enough to reach the upper one is not affordable in the default run.
# Zero is prepended explicitly, because k_p = 0 is the null hypothesis and a log
# grid cannot contain it.
#
# ALL profiles use the same grid, so the comparison between them is not
# confounded by the grid. Note what each comparison is: continuous-only versus
# withdrawal-only is two different EXPERIMENTS, and only profile_kp_joint
# answers "does adding the withdrawal arm narrow the ridge?".
#
# A caution that applies to every profile here: admissibility is an AVERAGE
# RMSE threshold, so the admissible total error grows with the number of
# observations. Comparing designs with different record lengths on this
# criterion compares the averaging as much as the information. profile_kp_joint
# returns a per-arm criterion that is invariant to that.
KP_GRID_LO = 0.02        # multiples of nominal k_p
KP_GRID_HI = 128.0       # multiples of nominal k_p
KP_GRID_N = 64


def _default_grid(p: Params) -> np.ndarray:
    return np.concatenate([[0.0], p.k_p * np.logspace(
        np.log10(KP_GRID_LO), np.log10(KP_GRID_HI), KP_GRID_N - 1)])


def _summarise(kp_grid: np.ndarray, rmse: np.ndarray, noise_sd: float,
               p: Params) -> dict:
    """Admissible set, ridge width, and whether the grid truncated either end."""
    admissible = kp_grid[rmse <= noise_sd]
    width = (float(admissible.max() - admissible.min()) / p.k_p
             if len(admissible) else 0.0)
    censored = bool(len(admissible) and (rmse[-1] <= noise_sd))
    # The ridge spans decades, so the scale-free measure is the FOLD range of
    # admissible k_p, not its width in multiples of nominal. Both are returned:
    # the fold range is what the prose should quote, the width is kept because
    # V12 and the existing manuscript inserts are written against it.
    nz = admissible[admissible > 0.0]
    fold = float(nz.max() / nz.min()) if len(nz) else 0.0
    return {
        "admissible": admissible,
        "kp_ridge_width": width,
        "kp_ridge_fold": fold,
        # True means the ridge ran off the top of the grid, so the ridge
        # measures are LOWER BOUNDS and must not be quoted as an interval.
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
    """Profile k_p, refitting k_S and gamma at each fixed value.

    ``noise_sd`` is the assumed measurement precision on a senescent fraction --
    2 percentage points is optimistic for a marker-based assay. A k_p is counted
    as admissible if its best refit stays within the noise floor, which is the
    practical rather than structural notion of identifiability.

    The returned ``kp_ridge_censored`` says whether the admissible set ran off
    the top of the grid. When it is True the ridge width is a lower bound and
    the admissible range must not be quoted as an interval, because its upper
    endpoint is wherever the grid happened to stop.
    """
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
    """Profile k_p against a WITHDRAWAL-ONLY design. Not an "added data" test.

    This was called ``withdrawal_breaks_degeneracy`` and was described, in the
    Methods and in this docstring, as repeating the profile "with the withdrawal
    arm included in the fit target". It never was. The target here is built
    fresh from the withdrawal design (81 points over 0-400 d) and the
    continuous-exposure residuals from :func:`profile_kp` (61 points over
    0-60 d) are not retained or concatenated. The two profiles therefore compare
    two DIFFERENT EXPERIMENTS with different sampling densities; neither is the
    other plus more data.

    Use :func:`profile_kp_joint` for the added-information question.
    """
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
    """Profile k_p against BOTH arms at once -- the real "added data" test.

    Continuous-exposure residuals and withdrawal residuals are concatenated
    into one objective, so this is the continuous-only design plus the
    withdrawal arm, which is what an experimenter who ran both would have.

    Two admissibility criteria are returned, and the difference between them is
    the point:

    ``kp_ridge_*``       pooled average RMSE <= noise_sd. This is the criterion
                         the rest of the module uses, and it is NOT
                         observation-count invariant: appending 81 points that
                         fit well pulls the average down and can ADMIT a k_p
                         whose continuous-exposure fit is plainly bad. At
                         k_p = 5 the withdrawal-only fit scores 0.0144 and
                         passes, while its continuous-exposure residual is
                         0.0421; the pooled joint fit scores 0.0267.

    ``kp_ridge_*_perarm`` EVERY arm within noise_sd separately. This is the
                         criterion that actually means "consistent with both
                         experiments", and it is what the manuscript should
                         quote. On the nominal parameters it returns exactly the
                         continuous-only admissible set (3.57-fold,
                         k_p in [0.208, 0.743]): the withdrawal arm neither
                         narrows the ridge nor widens it.

    So the qualitative conclusion -- the withdrawal arm does not resolve
    feedback STRENGTH -- survives. What does not survive is the claim that it
    widens the ridge, and the "dense sampling of the plateau dilutes
    information" explanation for that widening. The widening was an artefact of
    averaging error over a longer, easier record.

    None of this is a profile-likelihood confidence interval. It is a heuristic
    discrepancy criterion with an assumed measurement precision, and only k_S
    and total clearance are refitted -- every other parameter is held at truth,
    which makes these ridges NARROWER than a real experiment would see.
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
