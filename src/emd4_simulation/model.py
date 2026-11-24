"""Senescence persistence and paracrine signalling after arsenite withdrawal.

Oxidative stress induces primary senescence. Secreted factors of the
senescence-associated secretory phenotype (SASP) recruit secondary senescent
cells, which secrete at a fraction of the primary-cell rate. Immune and intrinsic
clearance remove senescent cells. Escape from arrest accumulates in a terminal
state, and SASP activity drives expansion of a neighbouring cell compartment.

Time is measured in days. The assigned parameters describe relative effects.
Each marker has its own sensitivity, background positivity, response to acute
stress and delay relative to the latent senescent fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields

import numpy as np
from scipy.integrate import solve_ivp

IDX = {
    "R":    0,   # oxidative stress (KCC5)
    "S1":   1,   # directly induced senescent fraction
    "S2":   2,   # paracrine-induced senescent fraction
    "C1":   3,   # primary secretion drive
    "C2":   4,   # secondary secretion drive
    "P":    5,   # extracellular SASP activity (KCC6)
    "X":    6,   # cumulative escape events per initial population, which can exceed one
    "lnN":  7,   # log neighbour expansion (KCC10)
    # Marker lag states follow the order of MARKERS.
    "m_p16":   8,
    "m_p21":   9,
    "m_lmnb1": 10,
    "m_sabg":  11,
    "m_ddr":   12,
    "m_sasp":  13,
}
N_STATE = len(IDX)


@dataclass(frozen=True)
class Marker:
    """Parameters linking latent senescence to marker positivity.

    sens is sensitivity, fp is baseline positivity in non-senescent cells, stress
    is the added positivity per unit of stress, and tau is the response lag in days.
    """
    key: str
    label: str
    sens: float
    fp: float
    stress: float
    tau: float


# Relative marker responses follow Ogrodnik et al. (2024) and SenNet (2024).
# Background positivity and stress sensitivity are assigned model parameters.
MARKERS = [
    Marker("m_p16",   "p16INK4a",       0.85, 0.03, 0.10, 3.0),
    Marker("m_p21",   "p21",            0.80, 0.06, 0.55, 1.0),
    Marker("m_lmnb1", "lamin B1 loss",  0.90, 0.05, 0.05, 5.0),
    Marker("m_sabg",  "SA-beta-gal",    0.95, 0.08, 0.90, 2.0),
    Marker("m_ddr",   "DDR foci",       0.75, 0.04, 1.20, 0.5),
    Marker("m_sasp",  "SASP protein",   0.90, 0.02, 0.05, 8.0),
]
MARKER_BY_KEY = {m.key: m for m in MARKERS}


@dataclass(frozen=True)
class Params:
    """Assigned relative effect sizes for the population model."""

    # Exposure -> ROS (KCC5, upstream)
    r_basal: float = 0.30      # basal ROS production
    k_E: float = 1.20          # arsenite -> ROS gain
    k_R: float = 1.00          # ROS clearance

    # ROS -> senescence commitment
    R_ref: float = 0.30        # reference ROS; commitment responds to (R-R_ref)+
    K_R: float = 0.60          # half-max of the ROS->commitment response
    h_R: float = 2.00          # steepness of that response
    k_S: float = 0.060         # max direct commitment rate, /day

    # Paracrine secondary senescence
    k_p: float = 0.350         # max paracrine induction rate, /day
    K_P: float = 0.250         # SASP activity at half-max paracrine induction
    n_P: float = 4.00          # cooperativity of the paracrine response

    # SASP production
    tau_C: float = 4.00        # arrest -> SASP competence delay, days
    k_sasp: float = 1.00       # secretion rate per competent primary cell
    q_sec: float = 0.50        # ratio of secondary to primary SASP secretion
    d_P: float = 1.00          # SASP clearance, /day

    # Clearance, split for KCC7 direction sensitivity
    gam_immune: float = 0.020      # immune-mediated clearance per day
    gam_intrinsic: float = 0.005   # resolution / intrinsic loss

    # KCC9, opposing polarity
    k_esc: float = 0.0015      # escape rate from senescence per day

    # KCC10, neighbour response
    k_prol: float = 0.030      # max SASP-driven neighbour expansion, /day
    K_prol: float = 0.200      # SASP activity at half-max neighbour response
    d_N: float = 0.004         # baseline neighbour turnover per day

    @property
    def gamma(self) -> float:
        return self.gam_immune + self.gam_intrinsic


@dataclass(frozen=True)
class Intervention:
    """Rate and activity multipliers applied from t_start onward.

    A value of one preserves the baseline rate. Delayed treatment probes the
    response of an established senescent population.
    """
    senolytic: float = 1.0      # multiplies clearance of existing senescent cells
    sasp_neut: float = 1.0      # multiplies SASP activity (neutralising antibody)
    immune: float = 1.0         # multiplies gam_immune  (KCC7 arm, both directions)
    paracrine: float = 1.0      # multiplies k_p (mechanism probe, not an experiment)
    q_sec: float = 1.0          # multiplies q_sec (the containment parameter)
    escape: float = 1.0         # multiplies k_esc (KCC9 arm)
    t_start: float = 0.0        # day the intervention begins

    def at(self, t: float) -> "Intervention":
        """Return the intervention in effect at time t."""
        if t >= self.t_start:
            return self
        return Intervention(t_start=self.t_start)


def hill(x: np.ndarray | float, K: float, n: float) -> np.ndarray | float:
    """Hill response to a nonnegative input."""
    xp = np.maximum(x, 0.0) ** n
    return xp / (K ** n + xp)


def stress_of(R: np.ndarray | float, p: Params) -> np.ndarray | float:
    """Normalised oxidative stress above the reference ROS level."""
    return hill(np.asarray(R) - p.R_ref, p.K_R, p.h_R)


def exposure(t: float, E: float, t_off: float) -> float:
    """Constant arsenite exposure from day zero until withdrawal at t_off."""
    return E if t < t_off else 0.0


def rhs(t: float, y: np.ndarray, p: Params, iv: Intervention,
        E: float, t_off: float) -> np.ndarray:
    R, S1, S2, C1, C2, P, X, lnN = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]

    dy = np.zeros_like(y)
    Et = exposure(t, E, t_off)

    # KCC5: exposure -> ROS
    dy[IDX["R"]] = p.k_E * Et + p.r_basal - p.k_R * R

    # Commitment
    S = S1 + S2
    free = max(0.0, 1.0 - S)                  # only non-senescent cells recruit

    a = iv.at(t)                              # active intervention

    induce_direct = p.k_S * stress_of(R, p)
    P_eff = P * a.sasp_neut                   # neutralisation acts on activity
    induce_para = p.k_p * a.paracrine * hill(P_eff, p.K_P, p.n_P)

    gam = (p.gam_immune * a.immune + p.gam_intrinsic) * a.senolytic
    k_esc = p.k_esc * a.escape

    dy[IDX["S1"]] = induce_direct * free - gam * S1 - k_esc * S1
    dy[IDX["S2"]] = induce_para * free - gam * S2 - k_esc * S2

    # C1 and C2 describe delayed secretion. After clearance they can temporarily
    # exceed S1 and S2, which affects the early response to senolytics.
    dy[IDX["C1"]] = (S1 - C1) / p.tau_C
    dy[IDX["C2"]] = (S2 - C2) / p.tau_C
    q2 = p.q_sec * a.q_sec
    dy[IDX["P"]] = p.k_sasp * (C1 + q2 * C2) - p.d_P * P

    # Escape events accumulate without feedback to the other states.
    dy[IDX["X"]] = k_esc * S

    # Time-matched control subtraction removes baseline neighbour turnover.
    dy[IDX["lnN"]] = p.k_prol * hill(P_eff, p.K_prol, 1.0) - p.d_N

    # Marker lag states
    for m in MARKERS:
        i = IDX[m.key]
        dy[i] = (S - y[i]) / m.tau

    return dy


def initial_state(p: Params) -> np.ndarray:
    """Start with zero senescence and the unexposed steady-state ROS level.

    This initial condition is an equilibrium when basal ROS is at or below R_ref.
    At higher basal ROS the unexposed control can accumulate senescent cells.
    """
    y0 = np.zeros(N_STATE)
    y0[IDX["R"]] = p.r_basal / p.k_R
    return y0


def simulate(p: Params, iv: Intervention, E: float, t: np.ndarray,
             t_off: float = np.inf, y0: np.ndarray | None = None) -> np.ndarray:
    y0 = initial_state(p) if y0 is None else y0
    sol = solve_ivp(
        rhs, (float(t[0]), float(t[-1])), y0, t_eval=t,
        args=(p, iv, E, t_off), method="LSODA",
        rtol=1e-8, atol=1e-10, max_step=1.0,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y


def trace(y: np.ndarray, p: Params, iv: Intervention,
          t: np.ndarray | None = None) -> dict:
    """Return latent states and simulated marker positivity.

    Pass the trajectory time grid to apply SASP neutralisation from t_start.
    Omitting the grid applies the endpoint multiplier to all columns.
    """
    S1, S2 = y[IDX["S1"]], y[IDX["S2"]]
    S = S1 + S2
    if t is None:
        P_eff = y[IDX["P"]] * iv.sasp_neut          # end-of-run handles only
    else:
        gate = np.array([iv.at(float(tt)).sasp_neut for tt in np.atleast_1d(t)])
        P_eff = y[IDX["P"]] * gate
    stress = stress_of(y[IDX["R"]], p)

    out = {
        # latent
        "R": y[IDX["R"]],
        "S": S,
        "S1": S1,
        "S2": S2,
        "P": P_eff,
        "X": y[IDX["X"]],
        "lnN": y[IDX["lnN"]],  # raw log expansion before control subtraction
        "stress": stress,
        "secondary_fraction": np.divide(S2, S, out=np.zeros_like(S), where=S > 1e-9),
    }
    # Observed marker positivity
    for m in MARKERS:
        Sm = y[IDX[m.key]]
        obs = m.sens * Sm + (m.fp + m.stress * stress) * (1.0 - Sm)
        out[m.key] = np.clip(obs, 0.0, 1.0)
    return out


def apply_control_relative_lnN(runs: dict, control_key: str = "control") -> dict:
    """Return trajectories with log neighbour expansion relative to the concurrent control.

    Subtracting the control at each time point cancels baseline turnover.
    """
    if control_key not in runs:
        raise KeyError(f"control arm {control_key!r} missing from runs")
    base = np.asarray(runs[control_key]["lnN"], dtype=float)
    out = {}
    for key, tr in runs.items():
        tr = dict(tr)
        tr["lnN"] = np.asarray(tr["lnN"], dtype=float) - base
        out[key] = tr
    return out


def manuscript_rhs(S: float, E_stress: float, p: Params) -> float:
    """Reduced senescence equation with Hill feedback acting directly on S."""
    return ((p.k_S * E_stress + p.k_p * hill(S, p.K_P, p.n_P)) * (1.0 - S)
            - p.gamma * S)


def qss_state(S: float, R: float, q: Params,
              split: float = 0.6) -> np.ndarray:
    """Construct a state with C1 = S1, C2 = S2 and secretion at steady state."""
    y = np.zeros(N_STATE)
    y[IDX["R"]] = R
    y[IDX["S1"]] = split * S
    y[IDX["S2"]] = (1.0 - split) * S
    y[IDX["C1"]] = y[IDX["S1"]]
    y[IDX["C2"]] = y[IDX["S2"]]
    y[IDX["P"]] = (q.k_sasp / q.d_P) * (y[IDX["C1"]] + q.q_sec * y[IDX["C2"]])
    return y


def quasi_steady_state_check(p: Params,
                             tol: float = 1e-9) -> tuple[bool, float]:
    """Compare the full senescence flux with the reduced equation on the slow manifold.

    Setting q_sec and k_sasp/d_P to one gives P = S. The comparison assumes
    negligible secretion delay and zero escape from arrest.
    """
    q = replace(p, tau_C=1e-6, d_P=1e6, k_sasp=1e6, q_sec=1.0, k_esc=0.0)
    iv = Intervention()
    worst = 0.0
    for S in np.linspace(0.0, 0.99, 40):
        for R in (q.R_ref, 0.6, 1.5):
            y = qss_state(S, R, q)
            dy = rhs(0.0, y, q, iv, 0.0, np.inf)
            mine = dy[IDX["S1"]] + dy[IDX["S2"]]
            theirs = manuscript_rhs(S, stress_of(R, q), q)
            worst = max(worst, abs(mine - theirs))
    return worst < tol, worst


def escape_residual_check(p: Params,
                          tol: float = 1e-9) -> tuple[bool, float]:
    """Check that the residual with escape enabled equals the drain −k_esc S."""
    q = replace(p, tau_C=1e-6, d_P=1e6, k_sasp=1e6, q_sec=1.0)
    iv = Intervention()
    worst = 0.0
    for S in np.linspace(0.0, 0.99, 40):
        for R in (q.R_ref, 0.6, 1.5):
            y = qss_state(S, R, q)
            dy = rhs(0.0, y, q, iv, 0.0, np.inf)
            mine = dy[IDX["S1"]] + dy[IDX["S2"]]
            theirs = manuscript_rhs(S, stress_of(R, q), q) - q.k_esc * S
            worst = max(worst, abs(mine - theirs))
    return worst < tol, worst


def param_names() -> list[str]:
    return [f.name for f in fields(Params)]
