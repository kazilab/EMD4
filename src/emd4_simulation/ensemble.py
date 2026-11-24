"""Sensitivity of senescence predictions to assigned model parameters.

Positive rate parameters vary independently on the log scale. Every draw
assumes paracrine feedback, weaker secretion by secondary cells, fixed marker
properties and terminal escape from arrest. The ensemble measures how often
the conclusions hold under these assumptions. It distinguishes persistence at
a finite time, two stable equilibria at zero exposure, and coexistence of an
exactly empty state with a persistent state.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .conditions import BY_KEY, PERSIST, T_END, T_OFF
from .bifurcation import (count_stable_at_zero, equilibria_at_zero,
                          q_sec_bistable_intervals)
from .model import IDX, Intervention, Params, simulate, stress_of, trace

# Log-scale standard deviations describe assumed parameter variability.
FREE_SIGMA = {
    # exposure -> ROS
    "k_E": 0.30, "r_basal": 0.25, "k_R": 0.25,
    # ROS -> commitment
    "K_R": 0.30, "h_R": 0.20, "k_S": 0.40, "R_ref": 0.20,
    # Paracrine feedback
    "k_p": 0.40, "K_P": 0.30, "n_P": 0.25,
    # SASP production and turnover
    "tau_C": 0.35, "d_P": 0.30, "q_sec": 0.30,
    # clearance
    "gam_immune": 0.40, "gam_intrinsic": 0.40,
    # KCC9 escape, KCC10 neighbour response
    "k_esc": 0.50, "k_prol": 0.30, "K_prol": 0.30, "d_N": 0.30,
}

# Secondary cells secrete less than primary cells in every draw.
Q_SEC_MAX = 0.95
N_T = 401


def sample_params(rng: np.random.Generator, base: Params | None = None) -> Params:
    base = base or Params()
    draw = {}
    for name, sigma in FREE_SIGMA.items():
        draw[name] = getattr(base, name) * float(np.exp(rng.normal(0.0, sigma)))
    draw["q_sec"] = float(np.clip(draw["q_sec"], 0.02, Q_SEC_MAX))
    draw["n_P"] = float(np.clip(draw["n_P"], 1.05, 12.0))   # retain a cooperative paracrine response
    draw["h_R"] = float(np.clip(draw["h_R"], 1.0, 8.0))
    return replace(base, **draw)


def has_empty_state(p: Params) -> bool:
    """Return whether basal ROS permits an exactly empty equilibrium.

    In the sampled range, a basal ROS level above R_ref drives commitment in the
    unexposed control. A positive low equilibrium may still coexist with a high one.
    """
    return bool(stress_of(p.r_basal / p.k_R, p) <= 0.0)


def high_state_retained(p: Params, iv: Intervention | None = None,
                        t_probe: float = 600.0) -> bool:
    """Test whether a high initial burden remains above PERSIST at t_probe without exposure."""
    iv = iv or Intervention()
    y0 = np.zeros(len(IDX))
    y0[IDX["R"]] = p.r_basal / p.k_R
    y0[IDX["S2"]] = 0.95
    y0[IDX["C2"]] = 0.95
    y0[IDX["P"]] = p.k_sasp * p.q_sec * 0.95 / p.d_P
    for m_key in ("m_p16", "m_p21", "m_lmnb1", "m_sabg", "m_ddr", "m_sasp"):
        y0[IDX[m_key]] = 0.95
    t = np.linspace(0.0, t_probe, 3)
    try:
        y = simulate(p, iv, 0.0, t, t_off=np.inf, y0=y0)
    except Exception:
        return False
    return bool(y[IDX["S1"], -1] + y[IDX["S2"], -1] > PERSIST)


def two_stable_at_zero(p: Params, iv: Intervention | None = None) -> bool:
    """Test for two or more stable equilibria at zero exposure."""
    try:
        return count_stable_at_zero(p, iv or Intervention()) >= 2
    except Exception:
        return False


def threshold_dose(p: Params, iv: Intervention | None = None,
                   t_off: float = T_OFF, lo: float = 0.0, hi: float = 4.0,
                   tol: float = 5e-3) -> float | None:
    """Lowest dose in the search range giving endpoint persistence after t_off exposure days."""
    iv = iv or Intervention()
    t = np.linspace(0.0, T_END, N_T)

    def persists(E: float) -> bool:
        try:
            y = simulate(p, iv, float(E), t, t_off=t_off)
        except Exception:
            return False
        return bool(y[IDX["S1"], -1] + y[IDX["S2"], -1] > PERSIST)

    if not persists(hi):
        return None
    if persists(lo):
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if persists(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def threshold_duration(p: Params, iv: Intervention | None = None,
                       E: float = 1.0, lo: float = 0.0, hi: float = 2 * T_OFF,
                       tol: float = 5e-3) -> float | None:
    """Shortest exposure duration in the search range giving endpoint persistence at dose E."""
    iv = iv or Intervention()
    t = np.linspace(0.0, T_END, N_T)

    def persists(t_off: float) -> bool:
        try:
            y = simulate(p, iv, float(E), t, t_off=float(t_off))
        except Exception:
            return False
        return bool(y[IDX["S1"], -1] + y[IDX["S2"], -1] > PERSIST)

    if not persists(hi):
        return None
    if persists(lo):
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if persists(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def q_sec_critical(p: Params, lo: float = 0.01, hi: float = 0.99) -> float | None:
    """Lower edge of the first bistable interval intersecting [lo, hi].

    The bounds come from numerical extrema of the analytic equilibrium curve.
    Return None when the interval search finds no bistability.
    """
    iv = q_sec_bistable_intervals(p, lo=lo, hi=hi)
    return float(iv[0][0]) if iv else None


def q_sec_bistable_window(p: Params, lo: float = 0.01,
                          hi: float = 0.99) -> tuple[bool, float | None]:
    """Return whether the bistable region closes within the range and its highest upper edge.

    At the upper fold the low-burden equilibrium is lost and high burden persists.
    """
    ivs = q_sec_bistable_intervals(p, lo=lo, hi=hi)
    if not ivs:
        return False, None
    upper = float(ivs[-1][1])
    return bool(upper < hi - 1e-9), upper


OBSERVABLES = ["S", "S1", "S2", "P", "X", "secondary_fraction",
               "m_p16", "m_lmnb1", "m_sabg", "m_ddr", "m_sasp"]
ARMS = ["control", "as", "as_low", "as_sasp", "as_seno", "as_qsec_lo"]


def run_ensemble(n: int = 300, seed: int = 4, base: Params | None = None) -> dict:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T_END, N_T)
    i14 = int(np.searchsorted(t, T_OFF))

    end = {k: {o: [] for o in OBSERVABLES} for k in ARMS}
    d14 = {k: {o: [] for o in OBSERVABLES} for k in ARMS}
    bistable, thresholds, q_crits = [], [], []
    empty_state, bistable_two_state = [], []
    two_stable, bistable_no_empty, reentrant = [], [], []
    near_fold = []
    ctrl_persistent, attributable, lowdose_baseline = [], [], []
    v3_discordant, v3_overstate, v3_understate, kS_draw = [], [], [], []
    v3_overstate_net = []
    v7_contrast, v7_sasp_collapses, v7_seno_survives = [], [], []
    ok = 0

    for _ in range(n):
        p = sample_params(rng, base)
        try:
            runs = {}
            for key in ARMS:
                c = BY_KEY[key]
                runs[key] = trace(simulate(p, c.iv, c.E, t, t_off=c.t_off),
                                  p, c.iv, t)
        except Exception:
            continue

        bs = high_state_retained(p)
        bistable.append(bs)
        empty = has_empty_state(p)
        empty_state.append(empty)
        # Classify all equilibrium-based outcomes from the same zero-exposure roots.
        try:
            eq0 = equilibria_at_zero(p)
        except Exception:
            eq0 = []
        stable_S = [e.S for e in eq0 if e.stable]
        two_st = len(stable_S) >= 2
        two_stable.append(two_st)
        # Coexistence requires an exactly empty stable equilibrium and a second stable state.
        bistable_two_state.append(
            bool(two_st and any(abs(x) < 1e-12 for x in stable_S)))
        bistable_no_empty.append(bool(two_st and not empty))
        thresholds.append(threshold_dose(p))
        qc = q_sec_critical(p)
        q_crits.append(qc)
        reentrant.append(q_sec_bistable_window(p)[0])
        near_fold.append(False)

        # Compare paired exposed and unexposed endpoints for each parameter draw.
        s_ctrl = runs["control"]["S"][-1]
        s_ref = runs["as"]["S"][-1]
        ctrl_persistent.append(bool(s_ctrl > PERSIST))
        attributable.append(bool(s_ref > PERSIST and s_ctrl <= PERSIST))
        # Near-baseline burden is a stricter threshold than persistence.
        lowdose_baseline.append(bool(runs["as_low"]["S"][-1] < 0.05))

        for key in ARMS:
            for o in OBSERVABLES:
                end[key][o].append(float(runs[key][o][-1]))
                d14[key][o].append(float(runs[key][o][i14]))

        # Use percentage-point differences because S can approach zero in the low-dose arm.
        # The control-adjusted gap removes the unexposed assay background.
        over = runs["as_low"]["m_sabg"][i14] - runs["as_low"]["S"][i14]
        over_ctrl = runs["control"]["m_sabg"][i14] - runs["control"]["S"][i14]
        v3_overstate.append(float(over))
        v3_overstate_net.append(float(over - over_ctrl))
        v3_discordant.append(bool(over > 0.10))
        v3_understate.append(bool(
            runs["as"]["m_sabg"][-1] < runs["as"]["m_sabg"][i14] - 0.02
            and runs["as"]["S"][-1] >= runs["as"]["S"][i14] - 0.02))
        kS_draw.append(float(p.k_S))

        # Restrict the paired comparison to draws with residual burden after senolysis.
        seno_end = runs["as_seno"]["S"][-1]
        sasp_end = runs["as_sasp"]["S"][-1]
        v7_sasp_collapses.append(bool(sasp_end < 0.05))
        v7_seno_survives.append(bool(seno_end > 0.20))
        if seno_end > 0.05:
            v7_contrast.append(bool(sasp_end < seno_end))
        ok += 1

    def band(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"median": np.nan, "lo": np.nan, "hi": np.nan}
        return {"median": float(np.median(a)),
                "lo": float(np.percentile(a, 5)),
                "hi": float(np.percentile(a, 95))}

    thr = [x for x in thresholds if x is not None]
    qcs = [x for x in q_crits if x is not None]

    return {
        "n_requested": n,
        "n_ok": ok,
        "end": {k: {o: band(v) for o, v in d.items()} for k, d in end.items()},
        "d14": {k: {o: band(v) for o, v in d.items()} for k, d in d14.items()},
        "frac_high_state_retained": float(np.mean(bistable)) if bistable else np.nan,
        "frac_two_stable_equilibria": float(np.mean(two_stable)) if two_stable else np.nan,
        "frac_empty_plus_persistent": float(np.mean(bistable_two_state)) if bistable_two_state else np.nan,
        "frac_empty_state_lost": float(1.0 - np.mean(empty_state)) if empty_state else np.nan,
        "frac_bistable_without_empty_state": float(
            np.mean(bistable_no_empty)) if bistable_no_empty else np.nan,
        "frac_persist_at_reference": float(
            np.mean([s > PERSIST for s in end["as"]["S"]])) if ok else np.nan,
        "frac_control_persistent": float(
            np.mean(ctrl_persistent)) if ctrl_persistent else np.nan,
        "frac_persist_exposure_attributable": float(
            np.mean(attributable)) if attributable else np.nan,
        # An endpoint below PERSIST can still exceed baseline.
        "frac_lowdose_resolves": float(
            np.mean([s < PERSIST for s in end["as_low"]["S"]])) if ok else np.nan,
        "frac_lowdose_near_baseline": float(
            np.mean(lowdose_baseline)) if lowdose_baseline else np.nan,
        "threshold_dose": band(thr),
        "frac_with_threshold": len(thr) / ok if ok else np.nan,
        # Threshold summaries are conditional on a crossing within the search range.
        "q_sec_critical": band(qcs),
        "n_with_threshold": len(thr),
        "n_with_q_sec_critical": len(qcs),
        "frac_with_q_sec_critical": len(qcs) / ok if ok else np.nan,
        "frac_q_sec_reentrant": float(np.mean(reentrant)) if reentrant else np.nan,
        "n_q_sec_reentrant": int(np.sum(reentrant)) if reentrant else 0,
        "n_q_sec_near_fold_undetermined": int(np.sum(near_fold)) if near_fold else 0,
        "frac_v3_discordant": float(np.mean(v3_discordant)) if v3_discordant else np.nan,
        "v3_overstate": band(v3_overstate),
        "v3_overstate_net": band(v3_overstate_net),
        "frac_v3_overstate_net_positive": (
            float(np.mean([x > 0.0 for x in v3_overstate_net]))
            if v3_overstate_net else np.nan),
        "v3_overstate_background": band(
            [a - b for a, b in zip(v3_overstate, v3_overstate_net)]),
        "frac_v3_understate": float(np.mean(v3_understate)) if v3_understate else np.nan,
        "frac_v7_contrast": float(np.mean(v7_contrast)) if v7_contrast else np.nan,
        "n_v7_comparable": len(v7_contrast),
        "frac_v7_sasp_collapses": float(np.mean(v7_sasp_collapses)) if v7_sasp_collapses else np.nan,
        "frac_v7_seno_survives": float(np.mean(v7_seno_survives)) if v7_seno_survives else np.nan,
        "raw_end": end,
    }
