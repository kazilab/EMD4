"""EMD4 mechanistic simulation: KCC5 -> EMD4/KCC6 -> KCC7, KCC10; KCC9 opposing.

Public-data-constrained mechanistic proof-of-concept for the EMD4 chain, with
arsenite as the exemplar exposure. Like the EMD1 build, this is a
*hypothesis-testing* model in relative effect sizes, not a quantitatively
validated carcinogenicity-prediction model. No parameter here is a physiological
rate constant.

Design decision that drives everything else
-------------------------------------------
S is NOT a marker. S is a LATENT multiparameter senescent state, and every
observable reaches it through a marker observation model with its own
sensitivity, its own exposure-driven false-positive rate, and its own onset
delay:

    marker        sens   base FP   stress FP   lag    reads what?
    p16           0.85    0.03       0.10      3 d    the latent state
    p21           0.80    0.06       0.55      1 d    state + acute stress
    lamin B1      0.90    0.05       0.05      5 d    the latent state
    SA-beta-gal   0.95    0.08       0.90      2 d    MOSTLY stress
    DDR foci      0.75    0.04       1.20      0.5 d  MOSTLY acute stress
    SASP protein  0.90    0.02       0.05      8 d    the state, late

The ``base FP`` column is reported alongside ``stress FP`` and is never dropped
from a summary of this table. It is positivity in UNEXPOSED tissue, so it is
present in the concurrent control at the same instant and it inflates any
marker-minus-state difference read on the absolute scale. Quoting a marker
overstatement without netting it against the control is how a constant gets
reported as an exposure effect (see ``ensemble.run_ensemble``, which reports
both scales, and ``apply_control_relative_lnN``, which applies the same
correction to the KCC10 observable).

That table is the whole EMD4 evidentiary argument in six rows. SA-beta-gal and
DDR foci rise *during* exposure while S is still low, and fall on withdrawal
while S persists. A model fitted to either one alone is fitting the stress
response, not senescence -- which is precisely why the domain requires a
multi-marker panel as a rule rather than a preference.

Time is in days; arsenite is applied from t = 0 and withdrawn at t_off.

Four deliberate departures from the sketched formulation
--------------------------------------------------------
1. **The paracrine loop runs through a secreted pool P, not through S.** The
   manuscript writes the feedback as k_p * S^n/(K^n + S^n). Routing it through
   an explicit SASP variable does three things that form does not: it makes
   "SASP neutralisation" a direct intervention rather than a parameter edit, it
   forces the delay between arrest and SASP competence to be represented, and it
   makes secretion -- not marker positivity -- the quantity the feedback depends
   on. In the fast-P, no-delay limit P is proportional to S and the manuscript's
   equation is recovered exactly (see `quasi_steady_state_check`).

2. **Senescent cells are split into primary and secondary.** S1 is directly
   induced, S2 is paracrine-induced, and they secrete at different rates
   (q_sec < 1). This is the ref-57 mechanism, and it is load-bearing: the
   bistability condition is a condition on q_sec. If secondary cells are
   sufficiently poorer secretors the loop cannot self-sustain and propagation is
   contained -- no tipping point, no hysteresis. Containment and bistability are
   therefore the same parameter question asked twice.

3. **Clearance is split into immune and intrinsic.** KCC7 is direction
   sensitive: effective immune clearance is NOT a positive KCC7 observation,
   impaired clearance is. Keeping gamma as one lumped number would make that
   distinction unrepresentable, so `gam_immune` is a separate handle that
   interventions move.

4. **Escape is an explicit flux, and it is TERMINAL -- which is not the same as
   inert.** k_esc drains S into a cumulative observable X. X is *terminal*: it
   has no outgoing edge, so nothing downstream of X can feed back into S, P or
   any other state. That is the whole of what "drives nothing" means here.

   It is NOT true that raising S cannot raise X. The equation is dX/dt = k_esc*S,
   so X(t) = k_esc * integral(S) and a larger accumulated senescent burden gives
   a strictly larger X at fixed k_esc. An earlier version of this docstring
   claimed the opposite ("raising S can never raise the immortalisation
   observable through any path in the model"); that contradicted the equation
   directly and is corrected here. What the KCC9 arm (V9) actually tests is a
   different proposition -- that raising the *escape rate* k_esc raises X while
   LOWERING S -- and it should not be read as evidence for the stronger claim.

   X is also not a measurement of immortalisation. It counts cumulative escape
   events normalised to the model population (so X > 1 is possible, because it
   accumulates turnover rather than counting living cells). The model contains
   no telomere maintenance, no indefinite proliferative capacity, and no
   immortalisation assay. The evidentiary rule KCC9 encodes -- that
   exposure-induced senescence must not be scored as evidence of immortalisation
   -- is an ontology commitment, not a result this model derives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields

import numpy as np
from scipy.integrate import solve_ivp

# --- state vector layout -------------------------------------------------
# Flat float array, not a dict, so the ensemble can run thousands of
# integrations without per-step allocation.
IDX = {
    "R":    0,   # oxidative stress / ROS                       -> KCC5 (upstream)
    "S1":   1,   # primary senescent fraction   (directly induced)  -.
    "S2":   2,   # secondary senescent fraction (paracrine)          |-> EMD4 / KCC6
    "C1":   3,   # delayed secretion DRIVE, primary   (lags S1)      |
    "C2":   4,   # delayed secretion DRIVE, secondary (lags S2)      |
    "P":    5,   # functional SASP activity (secreted)             -'
    "X":    6,   # cumulative escape EVENTS (not a live-cell fraction; may
                 # exceed 1)                                   -> KCC9 (opposing)
    "lnN":  7,   # log neighbour-compartment expansion          -> KCC10
    # --- marker lag states; order must match MARKERS ---------------------
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
    """One row of the observation model.

    ``sens``   fraction of genuinely senescent cells scoring positive
    ``fp``     background positivity in non-senescent cells, unexposed
    ``stress`` ADDITIONAL non-senescent positivity per unit of exposure stress.
               This is the column that separates a senescence marker from a
               stress marker, and it is why SA-beta-gal is supportive rather
               than definitive evidence.
    ``tau``    onset lag, days, from commitment to scoring positive
    """
    key: str
    label: str
    sens: float
    fp: float
    stress: float
    tau: float


# Sensitivity/specificity magnitudes follow the consensus criteria (Ogrodnik
# 2024, Cell; SenNet 2024, Nat Rev Mol Cell Biol): SA-beta-gal is "neither
# necessary nor sufficient", p21 does not distinguish durable senescence from
# transient arrest, lamin B1 loss and p16 are the more specific state readouts,
# and a measured secretory phenotype arrives late.
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
    """Relative effect sizes.

    Provenance lives in the trailing comment on each field, which
    ``manuscript_tables.param_comments`` parses into Supplementary Table S1 --
    so the comment IS the documentation. Unlike ``emd1_simulation``, this build
    has no ``fit/`` directory and no ``fit/PROVENANCE.md``: nothing here is
    fitted study-by-study, because no published measurement constrains these
    magnitudes in the exemplar system.
    """

    # --- exposure -> ROS (KCC5, upstream) --------------------------------
    r_basal: float = 0.30      # basal ROS production
    k_E: float = 1.20          # arsenite -> ROS gain
    k_R: float = 1.00          # ROS clearance

    # --- ROS -> senescence commitment ------------------------------------
    R_ref: float = 0.30        # reference ROS; commitment responds to (R-R_ref)+
    K_R: float = 0.60          # half-max of the ROS->commitment response
    h_R: float = 2.00          # steepness of that response
    k_S: float = 0.060         # max direct commitment rate, /day

    # --- paracrine secondary senescence (the feedback under test) --------
    k_p: float = 0.350         # max paracrine induction rate, /day
    K_P: float = 0.250         # SASP activity at half-max paracrine induction
    n_P: float = 4.00          # cooperativity of the paracrine response

    # --- SASP production -------------------------------------------------
    tau_C: float = 4.00        # arrest -> SASP competence delay, days
    k_sasp: float = 1.00       # secretion rate per competent primary cell
    q_sec: float = 0.50        # secondary-cell secretion RELATIVE to primary.
                               # ref 57's containment mechanism; the bistability
                               # condition is a condition on this number.
    d_P: float = 1.00          # SASP clearance, /day

    # --- clearance, split for KCC7 direction sensitivity -----------------
    gam_immune: float = 0.020      # immune-mediated clearance (NOT KCC7-positive)
    gam_intrinsic: float = 0.005   # resolution / intrinsic loss

    # --- KCC9, opposing polarity -----------------------------------------
    k_esc: float = 0.0015      # escape from arrest; drains S into X, drives nothing

    # --- KCC10, neighbour response ---------------------------------------
    k_prol: float = 0.030      # max SASP-driven neighbour expansion, /day
    K_prol: float = 0.200      # SASP activity at half-max neighbour response
    d_N: float = 0.004         # baseline neighbour turnover. Absolute lnN
                               # drifts as -d_N*t at P=0; the REPORTED
                               # observable is ΔlnN vs concurrent control
                               # (see apply_control_relative_lnN).

    @property
    def gamma(self) -> float:
        return self.gam_immune + self.gam_intrinsic


@dataclass(frozen=True)
class Intervention:
    """Multiplicative handles. 1.0 = untouched.

    Each corresponds to an experiment someone actually ran, or to a
    direction-sensitivity test the ontology requires.

    ``t_start`` matters more than it looks. A senolytic given from t = 0 is a
    prevention experiment; a senolytic given after the senescent population has
    established is a clearance experiment, and only the second one tests whether
    an already-persistent state can be collapsed. The published ablation designs
    are the second kind, so interventions are time-gated by default rather than
    applied to the whole run.
    """
    senolytic: float = 1.0      # multiplies clearance of EXISTING senescent cells
    sasp_neut: float = 1.0      # multiplies SASP activity (neutralising antibody)
    immune: float = 1.0         # multiplies gam_immune  (KCC7 arm, both directions)
    paracrine: float = 1.0      # multiplies k_p (mechanism probe, not an experiment)
    q_sec: float = 1.0          # multiplies q_sec (the containment parameter)
    escape: float = 1.0         # multiplies k_esc (KCC9 arm)
    t_start: float = 0.0        # day the intervention begins

    def at(self, t: float) -> "Intervention":
        """The handles actually in force at time t."""
        if t >= self.t_start:
            return self
        return Intervention(t_start=self.t_start)


def hill(x: np.ndarray | float, K: float, n: float) -> np.ndarray | float:
    """Hill function, guarded at x <= 0 so the ensemble cannot raise a negative
    base to a fractional power."""
    xp = np.maximum(x, 0.0) ** n
    return xp / (K ** n + xp)


def stress_of(R: np.ndarray | float, p: Params) -> np.ndarray | float:
    """Normalised oxidative stress in [0, 1).

    Drives BOTH senescence commitment and the exposure-dependent false-positive
    term in the observation model. That shared origin is deliberate: it is why a
    stress-responsive marker mimics a dose-response for senescence.
    """
    return hill(np.asarray(R) - p.R_ref, p.K_R, p.h_R)


def exposure(t: float, E: float, t_off: float) -> float:
    """Arsenite from t = 0, withdrawn at t_off. Square, not tapered: the
    withdrawal experiments this is checked against are medium changes."""
    return E if t < t_off else 0.0


def rhs(t: float, y: np.ndarray, p: Params, iv: Intervention,
        E: float, t_off: float) -> np.ndarray:
    R, S1, S2, C1, C2, P, X, lnN = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]

    dy = np.zeros_like(y)
    Et = exposure(t, E, t_off)

    # --- KCC5: exposure -> ROS -------------------------------------------
    dy[IDX["R"]] = p.k_E * Et + p.r_basal - p.k_R * R

    # --- commitment ------------------------------------------------------
    S = S1 + S2
    free = max(0.0, 1.0 - S)                  # only non-senescent cells recruit

    a = iv.at(t)                              # handles in force right now

    induce_direct = p.k_S * stress_of(R, p)
    P_eff = P * a.sasp_neut                   # neutralisation acts on activity
    induce_para = p.k_p * a.paracrine * hill(P_eff, p.K_P, p.n_P)

    gam = (p.gam_immune * a.immune + p.gam_intrinsic) * a.senolytic
    k_esc = p.k_esc * a.escape

    dy[IDX["S1"]] = induce_direct * free - gam * S1 - k_esc * S1
    dy[IDX["S2"]] = induce_para * free - gam * S2 - k_esc * S2

    # --- delayed secretion drive and secretion ---------------------------
    # C_i is an EFFECTIVE DELAYED SECRETION DRIVE, not a live subpopulation.
    # It is a first-order lag on S_i with no matched removal term, so after a
    # sharp clearance event (e.g. the 6x senolytic arm) C can exceed S for
    # roughly tau_C: the drive relaxes towards the new, lower S rather than
    # dropping with it. Peak excess in the nominal senolytic arm is
    # C - S = 0.119 at t = 62.9 d.
    #
    # Read literally as "SASP-competent senescent cells" that is impossible --
    # it would be a subset larger than its superset. Read as secretion memory
    # (transcript and protein already committed by cells that have since been
    # removed) it is a deliberate phenomenological choice. Intervention
    # comparisons that turn on the first few days after a senolytic therefore
    # depend on this choice; a structural alternative is immature/mature
    # senescent compartments with matched maturation and clearance, which would
    # change those predictions and has NOT been implemented here.
    dy[IDX["C1"]] = (S1 - C1) / p.tau_C
    dy[IDX["C2"]] = (S2 - C2) / p.tau_C
    q2 = p.q_sec * a.q_sec
    dy[IDX["P"]] = p.k_sasp * (C1 + q2 * C2) - p.d_P * P

    # --- KCC9 (opposing polarity): cumulative escape, terminal -----------
    # Terminal = no OUTGOING edge. X is still driven BY S: this is a strictly
    # positive S -> X path, so a larger burden gives a larger X. See departure 4.
    dy[IDX["X"]] = k_esc * S

    # --- KCC10: SASP-driven neighbour expansion --------------------------
    # Absolute integrator: at P=0 this is -d_N, so control tissue shrinks
    # on the raw scale. The KCC10-facing report is ΔlnN vs concurrent
    # control (apply_control_relative_lnN), which cancels the basal drain.
    dy[IDX["lnN"]] = p.k_prol * hill(P_eff, p.K_prol, 1.0) - p.d_N

    # --- marker lag states ------------------------------------------------
    for m in MARKERS:
        i = IDX[m.key]
        dy[i] = (S - y[i]) / m.tau

    return dy


def initial_state(p: Params) -> np.ndarray:
    """Unexposed, senescence-free tissue at its ROS fixed point.

    S = 0 is an exact equilibrium at E = 0 under TWO conditions, both needed:

    1. ``n_P > 1``, so the paracrine term is o(S) at the origin; and
    2. ``stress_of(r_basal / k_R, p) == 0``, i.e. the basal ROS fixed point
       ``R* = r_basal/k_R`` does not exceed ``R_ref``. Otherwise direct
       commitment ``k_S * stress(R*)`` is strictly positive at S = 0 and the
       empty state is not an equilibrium at all: senescence accumulates with
       no exposure.

    At the nominal parameters R* = 0.30 = R_ref exactly, so condition 2 holds
    on the boundary and the low state is genuinely empty -- which is what makes
    the high state, when it exists, a second attractor rather than the tail of
    one curve.

    The ensemble does NOT inherit this. It log-samples r_basal, k_R and R_ref
    independently, so R* > R_ref in roughly half of all draws and those draws
    have no senescence-free state to be bistable *between*. That is measured
    and reported as ``frac_empty_state_lost``; see the note on ``frac_bistable``
    in ``ensemble.run_ensemble``.
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
        rtol=1e-8, atol=1e-10, max_step=1.0,   # max_step so the withdrawal step
                                               # at t_off is never stepped over
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y


def trace(y: np.ndarray, p: Params, iv: Intervention,
          t: np.ndarray | None = None) -> dict:
    """Observables. Latent quantities and observed markers are kept separate on
    purpose -- confusing the two is the error the model exists to expose.

    ``t`` is the time grid the trajectory was evaluated on. It is needed
    because SASP neutralisation is TIME-GATED: ``rhs`` applies it through
    ``iv.at(t)``, so it is in force only from ``iv.t_start``. Reporting
    ``P * iv.sasp_neut`` for the whole trajectory -- as this function did
    before ``t`` was threaded through -- scaled P down over the pre-treatment
    window too, making the reported P wrong before day ``t_start``. Endpoint
    tables were unaffected (they sample after t_start) but any time course was
    not. Passing ``t`` is strongly preferred; omitting it falls back to the
    old end-of-run behaviour for callers that only read the final column.
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
        "lnN": y[IDX["lnN"]],  # RAW integrator; convert via apply_control_relative_lnN
        "stress": stress,
        "secondary_fraction": np.divide(S2, S, out=np.zeros_like(S), where=S > 1e-9),
    }
    # observed markers -- what an experiment would actually report
    for m in MARKERS:
        Sm = y[IDX[m.key]]
        obs = m.sens * Sm + (m.fp + m.stress * stress) * (1.0 - Sm)
        out[m.key] = np.clip(obs, 0.0, 1.0)
    return out


def apply_control_relative_lnN(runs: dict, control_key: str = "control") -> dict:
    """Replace lnN with ΔlnN versus the time-matched control arm.

    The neighbour ODE carries a basal drain ``-d_N``, so raw control
    ``lnN(t) = -d_N·t``. That absolute drift is not a biological claim about
    tissue disappearing; it is an arbitrary zero of the log-count scale. An
    experiment measures fold-change against a concurrent control plate, not
    absolute cell number on that scale. Reporting
    ``ΔlnN = lnN_arm(t) − lnN_control(t)`` cancels the basal drain, puts
    control at 0 by construction, and is the KCC10-facing observable.

    The ODE itself is unchanged. Call this once on a finished ``runs`` dict
    (as ``run_nominal`` does) before any endpoint or battery consumption.
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


# --- the manuscript's own equation, for the QSS cross-check ---------------

def manuscript_rhs(S: float, E_stress: float, p: Params) -> float:
    """dS/dt = (k_S*stress(R) + k_p*S^n/(K^n+S^n))*(1-S) - gamma*S

    The sketched formulation, with the paracrine term written directly on S.
    Used by ``quasi_steady_state_check`` to confirm that departure 1 is a
    reformulation and not a different model.
    """
    return ((p.k_S * E_stress + p.k_p * hill(S, p.K_P, p.n_P)) * (1.0 - S)
            - p.gamma * S)


def qss_state(S: float, R: float, q: Params,
              split: float = 0.6) -> np.ndarray:
    """A point on the slow manifold: no competence lag, P at quasi-steady state.

    ``C_i = S_i`` makes dC/dt vanish; ``P = (k_sasp/d_P)(C1 + q_sec C2)`` makes
    dP/dt vanish. So this is a genuine QSS point of the FULL system, not an
    algebraic stand-in for one.
    """
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
    """In the fast-P, no-delay, single-population limit the P-mediated model
    must reduce to the manuscript's equation.

    Take tau_C -> 0, d_P -> large with k_sasp/d_P = 1, q_sec -> 1 and
    k_esc -> 0; then P -> S and the two right-hand sides must agree pointwise.

    **This evaluates the real ``rhs``.** An earlier version of this function
    rebuilt the reduced right-hand side from the same algebra as
    ``manuscript_rhs`` and compared the two, which agreed to 0.00e+00 by
    construction and tested nothing. Here the full 14-state ``rhs`` is
    evaluated on the slow manifold (:func:`qss_state`) and its
    d(S1+S2)/dt is compared against the manuscript form, so a change to the
    ODE that broke the correspondence would actually be caught.

    ``k_esc -> 0`` is part of the limit and not a convenience: the manuscript
    equation has no escape term, so the reformulation reduces to it only once
    the terminal drain is switched off. :func:`escape_residual_check` measures
    what that term contributes when it is left on.

    One subtlety: ``d_P`` enters only through the ratio ``k_sasp/d_P``, because
    :func:`qss_state` places P at its steady value directly. Making P *fast* is
    what justifies the manifold dynamically; the correspondence tested here is
    the algebraic one that holds on it.
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
    """With escape left ON, the residual must be exactly the escape drain.

    The companion to :func:`quasi_steady_state_check`. It pins down *what* the
    reformulation adds to the manuscript equation -- a terminal -k_esc*S flux
    and nothing else -- rather than leaving the difference unaccounted for.
    """
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
