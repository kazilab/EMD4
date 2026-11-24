"""Feedback-free models of senescence during continuous exposure and after withdrawal.

The simple null relaxes toward a saturating dose response with time constant
tau_S. The cascade null adds a delayed intermediate. Both are fitted to
synthetic continuous-exposure data before predicting the withdrawal response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .model import IDX, Intervention, Params, exposure, simulate, trace


@dataclass(frozen=True)
class NullParams:
    S_max: float = 0.95
    h: float = 6.0
    K_E: float = 0.3
    tau_S: float = 20.0


def null_sinf(E: float | np.ndarray, q: NullParams) -> float | np.ndarray:
    Ep = np.maximum(np.asarray(E, dtype=float), 0.0) ** q.h
    return q.S_max * Ep / (q.K_E ** q.h + Ep)


def simulate_null(q: NullParams, E: float, t: np.ndarray,
                  t_off: float = np.inf) -> np.ndarray:
    def f(tt, y):
        return (null_sinf(exposure(tt, E, t_off), q) - y[0]) / q.tau_S

    sol = solve_ivp(f, (float(t[0]), float(t[-1])), [0.0], t_eval=t,
                    method="LSODA", rtol=1e-9, atol=1e-11, max_step=1.0)
    if not sol.success:
        raise RuntimeError(f"null integration failed: {sol.message}")
    return sol.y[0]


def forward_dose_response(p: Params, iv: Intervention, doses: np.ndarray,
                          t_end: float = 60.0) -> np.ndarray:
    """Endpoint senescent fraction over a range of continuous exposures."""
    t = np.linspace(0.0, t_end, int(t_end * 10) + 1)
    out = np.zeros_like(doses, dtype=float)
    for i, E in enumerate(doses):
        y = simulate(p, iv, float(E), t, t_off=np.inf)
        out[i] = y[IDX["S1"], -1] + y[IDX["S2"], -1]
    return out


def fit_null(doses: np.ndarray, target: np.ndarray,
             t_end: float = 60.0) -> tuple[NullParams, float]:
    """Fit the simple null to the synthetic endpoint dose response and return its R²."""
    t = np.linspace(0.0, t_end, int(t_end * 10) + 1)

    def resid(v):
        q = NullParams(S_max=v[0], h=v[1], K_E=v[2], tau_S=v[3])
        pred = np.array([simulate_null(q, float(E), t)[-1] for E in doses])
        return pred - target

    v0 = [max(target.max(), 1e-3), 6.0,
          float(np.interp(0.5 * target.max(), target, doses)), 20.0]
    sol = least_squares(resid, v0, bounds=([0.0, 0.5, 1e-4, 0.5],
                                           [1.0, 60.0, 10.0, 500.0]),
                        xtol=1e-12, ftol=1e-12)
    q = NullParams(S_max=sol.x[0], h=sol.x[1], K_E=sol.x[2], tau_S=sol.x[3])
    ss_res = float(np.sum(sol.fun ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return q, 1.0 - ss_res / max(ss_tot, 1e-30)


def null_recovery_time(q: NullParams) -> float:
    """The simple null has a constant recovery time tau_S at every dose."""
    return q.tau_S


def compare_withdrawal(p: Params, iv: Intervention, q: NullParams,
                       E: float, t_off: float, t_end: float = 400.0) -> dict:
    """Predict both trajectories after withdrawal using the continuous-exposure fit."""
    t = np.linspace(0.0, t_end, int(t_end * 4) + 1)
    y = simulate(p, iv, E, t, t_off=t_off)
    S_mech = y[IDX["S1"]] + y[IDX["S2"]]
    S_null = simulate_null(q, E, t, t_off=t_off)
    return {
        "t": t,
        "mech": S_mech,
        "null": S_null,
        "mech_end": float(S_mech[-1]),
        "null_end": float(S_null[-1]),
        "mech_at_off": float(S_mech[np.searchsorted(t, t_off)]),
        "null_at_off": float(S_null[np.searchsorted(t, t_off)]),
    }


def forward_timecourse(p: Params, iv: Intervention, doses: np.ndarray,
                       t: np.ndarray) -> np.ndarray:
    """Simulate the senescent fraction at each dose and time under continuous exposure."""
    out = np.zeros((len(doses), len(t)))
    for i, E in enumerate(doses):
        y = simulate(p, iv, float(E), t, t_off=np.inf)
        out[i] = y[IDX["S1"]] + y[IDX["S2"]]
    return out


def fit_null_timecourse(doses: np.ndarray, target: np.ndarray,
                        t: np.ndarray) -> tuple[NullParams, float]:
    """Fit the simple null to all doses and time points under continuous exposure."""
    def resid(v):
        q = NullParams(S_max=v[0], h=v[1], K_E=v[2], tau_S=v[3])
        pred = np.array([simulate_null(q, float(E), t) for E in doses])
        return (pred - target).ravel()

    endpoint = target[:, -1]
    v0 = [max(endpoint.max(), 1e-3), 6.0,
          float(np.interp(0.5 * endpoint.max(), endpoint, doses)), 20.0]
    sol = least_squares(resid, v0, bounds=([0.0, 0.5, 1e-4, 0.5],
                                           [1.0, 60.0, 10.0, 500.0]),
                        xtol=1e-12, ftol=1e-12)
    q = NullParams(S_max=sol.x[0], h=sol.x[1], K_E=sol.x[2], tau_S=sol.x[3])
    ss_res = float(np.sum(sol.fun ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return q, 1.0 - ss_res / max(ss_tot, 1e-30)


@dataclass(frozen=True)
class CascadeNull:
    """Two first-order stages produce a delayed, sigmoidal response without feedback."""
    S_max: float = 0.95
    h: float = 6.0
    K_E: float = 0.15
    tau_1: float = 10.0
    tau_2: float = 10.0


def cascade_sinf(E: float | np.ndarray, q: CascadeNull) -> float | np.ndarray:
    Ep = np.maximum(np.asarray(E, dtype=float), 0.0) ** q.h
    return q.S_max * Ep / (q.K_E ** q.h + Ep)


def simulate_cascade(q: CascadeNull, E: float, t: np.ndarray,
                     t_off: float = np.inf) -> np.ndarray:
    def f(tt, y):
        drive = cascade_sinf(exposure(tt, E, t_off), q)
        return [(drive - y[0]) / q.tau_1, (y[0] - y[1]) / q.tau_2]

    sol = solve_ivp(f, (float(t[0]), float(t[-1])), [0.0, 0.0], t_eval=t,
                    method="LSODA", rtol=1e-9, atol=1e-11, max_step=1.0)
    if not sol.success:
        raise RuntimeError(f"cascade-null integration failed: {sol.message}")
    return sol.y[1]


def fit_cascade(doses: np.ndarray, target: np.ndarray,
                t: np.ndarray) -> tuple[CascadeNull, float]:
    def resid(v):
        q = CascadeNull(S_max=v[0], h=v[1], K_E=v[2], tau_1=v[3], tau_2=v[4])
        pred = np.array([simulate_cascade(q, float(E), t) for E in doses])
        return (pred - target).ravel()

    endpoint = target[:, -1]
    v0 = [max(endpoint.max(), 1e-3), 6.0,
          float(np.interp(0.5 * endpoint.max(), endpoint, doses)), 10.0, 10.0]
    sol = least_squares(resid, v0,
                        bounds=([0.0, 0.5, 1e-4, 0.2, 0.2],
                                [1.0, 60.0, 10.0, 300.0, 300.0]),
                        xtol=1e-12, ftol=1e-12)
    q = CascadeNull(S_max=sol.x[0], h=sol.x[1], K_E=sol.x[2],
                    tau_1=sol.x[3], tau_2=sol.x[4])
    ss_res = float(np.sum(sol.fun ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return q, 1.0 - ss_res / max(ss_tot, 1e-30)


def cascade_withdrawal(p: Params, iv: Intervention, q: CascadeNull,
                       E: float, t_off: float, t_end: float = 400.0) -> dict:
    t = np.linspace(0.0, t_end, int(t_end * 4) + 1)
    y = simulate(p, iv, E, t, t_off=t_off)
    S_mech = y[IDX["S1"]] + y[IDX["S2"]]
    S_null = simulate_cascade(q, E, t, t_off=t_off)
    i_off = int(np.searchsorted(t, t_off))
    return {"t": t, "mech": S_mech, "null": S_null,
            "mech_end": float(S_mech[-1]), "null_end": float(S_null[-1]),
            "mech_at_off": float(S_mech[i_off]),
            "null_at_off": float(S_null[i_off]),
            "null_peak_after": float(S_null[i_off:].max())}
