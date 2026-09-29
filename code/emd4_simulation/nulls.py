"""The flat null: a steep sigmoid with no feedback, fitted just as hard.

This is the EMD4 counterpart of EMD1's global-m6A scalar. The claim under test
is not "senescence accumulates nonlinearly with dose" -- a monotone saturating
model does that too, and does it with fewer assumptions. The claim is that the
accumulation is SELF-SUSTAINING, and the null exists to show which measurements
can tell those apart and which cannot.

    null:  dS/dt = (S_inf(E) - S) / tau_S ,   S_inf(E) = S_max * E^h/(K^h + E^h)

Four free parameters against the mechanistic model's paracrine block -- the null
is not handicapped. It is fitted to the forward dose-response an experimenter
would actually run, then confronted with withdrawal and recovery-time data it
never saw.
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
    """Endpoint senescent fraction after continuous exposure -- the experiment
    almost every published dose-response actually is."""
    t = np.linspace(0.0, t_end, int(t_end * 10) + 1)
    out = np.zeros_like(doses, dtype=float)
    for i, E in enumerate(doses):
        y = simulate(p, iv, float(E), t, t_off=np.inf)
        out[i] = y[IDX["S1"], -1] + y[IDX["S2"], -1]
    return out


def fit_null(doses: np.ndarray, target: np.ndarray,
             t_end: float = 60.0) -> tuple[NullParams, float]:
    """Least-squares fit of the null to a forward dose-response.

    Returns the fitted null and its R^2. The null is fitted to the same
    endpoint-vs-dose curve the mechanistic model produces, at the same
    timepoint, so any difference later is a difference in dynamics and not in
    calibration.
    """
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
    """The null's recovery time is tau_S at every exposure, by construction.

    It cannot slow down near a threshold because it has no threshold -- which is
    what makes the recovery-time measurement discriminating rather than merely
    corroborating.
    """
    return q.tau_S


def compare_withdrawal(p: Params, iv: Intervention, q: NullParams,
                       E: float, t_off: float, t_end: float = 400.0) -> dict:
    """Run both models through the withdrawal protocol they were never fitted to."""
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
    """Full S(t) under CONTINUOUS exposure, every dose, every timepoint.

    This is everything an experimenter has before anyone runs a withdrawal arm,
    and it is what the null must be fitted to if the comparison is to be fair.
    """
    out = np.zeros((len(doses), len(t)))
    for i, E in enumerate(doses):
        y = simulate(p, iv, float(E), t, t_off=np.inf)
        out[i] = y[IDX["S1"]] + y[IDX["S2"]]
    return out


def fit_null_timecourse(doses: np.ndarray, target: np.ndarray,
                        t: np.ndarray) -> tuple[NullParams, float]:
    """Fit the null to the whole continuous-exposure time course.

    Strictly harder than fitting endpoints, and the version the null deserves:
    if it still fails the withdrawal arm after seeing every timepoint at every
    dose, the failure is structural rather than a calibration artefact.
    """
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
    """A no-feedback model that CAN produce a sigmoidal rise in time.

    E -> U -> S, two first-order stages. The lag makes S(t) S-shaped without any
    self-amplification, which removes the cheapest objection to the simple null:
    that it was rejected for lacking a shape rather than for lacking feedback.

    Five free parameters against the simple null's four. If this still fails the
    withdrawal arm, no monotone single-valued dose-response reproduces
    persistence, and the failure is structural.
    """
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
