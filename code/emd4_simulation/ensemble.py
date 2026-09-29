"""Uncertainty propagation.

None of these effect sizes is measured in the exemplar system. The published
arsenite work establishes that premature senescence occurs, that markers and
SASP factors persist after withdrawal, and that secondary senescence exists --
but not the rate at which any of it happens. Point estimates would therefore be
false precision, so the free effect sizes are treated as log-normally uncertain
and every claim is reported as an ensemble band. A prediction only counts if it
survives the spread.

What is NOT sampled is the hypothesis:

- that a paracrine loop exists at all (k_p > 0 always),
- that secondary senescent cells secrete LESS than primary ones (q_sec < 1),
- which markers are stress-driven and which read the latent state,
- that escape is terminal, i.e. has no outgoing edge (it is still driven by S).

One feature specific to EMD4 makes this ensemble do more work than EMD1's.
Sampling moves draws ACROSS the bifurcation: some parameter sets are bistable
and some are not. That is not noise to be averaged away -- it is the honest
answer to the question V6 asserts rather than measures. The headline output of
this module is therefore the *fraction of plausible parameter space that has a
tipping point at all*.

Three quantities get called "bistability" and they must be kept apart. In the
default seed-4, n = 250 ensemble:

    frac_high_state_retained        235/250   a high initial burden is still
                                              high at the finite probe time
    frac_two_stable_equilibria      170/250   two stable equilibria at E = 0
                                              (ordinary bistability)
    frac_empty_plus_persistent      115/250   an exactly EMPTY stable state
                                              plus a persistent one

Under the 90% convention this build uses for "robust", NONE of 170/250 (68%)
or 115/250 (46%) is robust; only the weakest of the three clears the bar, and
it is not a bistability count. Reporting any single one of these as "bistable"
is the error this module is now structured to prevent.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .conditions import BY_KEY, PERSIST, T_END, T_OFF
from .bifurcation import (count_stable_at_zero, equilibria_at_zero,
                          q_sec_bistable_intervals)
from .model import IDX, Intervention, Params, simulate, stress_of, trace

# Free effect sizes and their log-scale sigma. Wider than EMD1's (0.15-0.35)
# because EMD1 could anchor most of its magnitudes on published measurements
# and this build cannot: these are plausibility ranges, not measurement error.
FREE_SIGMA = {
    # exposure -> ROS
    "k_E": 0.30, "r_basal": 0.25, "k_R": 0.25,
    # ROS -> commitment
    "K_R": 0.30, "h_R": 0.20, "k_S": 0.40, "R_ref": 0.20,
    # the paracrine loop -- magnitudes uncertain, existence is not
    "k_p": 0.40, "K_P": 0.30, "n_P": 0.25,
    # SASP production and turnover
    "tau_C": 0.35, "d_P": 0.30, "q_sec": 0.30,
    # clearance
    "gam_immune": 0.40, "gam_intrinsic": 0.40,
    # KCC9 escape, KCC10 neighbour response
    "k_esc": 0.50, "k_prol": 0.30, "K_prol": 0.30, "d_N": 0.30,
}

# q_sec is a RATIO and must stay below 1: a secondary senescent cell that
# secreted more than a primary one would be a different hypothesis, not a
# wider interval on this one.
Q_SEC_MAX = 0.95
N_T = 401



def sample_params(rng: np.random.Generator, base: Params | None = None) -> Params:
    base = base or Params()
    draw = {}
    for name, sigma in FREE_SIGMA.items():
        draw[name] = getattr(base, name) * float(np.exp(rng.normal(0.0, sigma)))
    draw["q_sec"] = float(np.clip(draw["q_sec"], 0.02, Q_SEC_MAX))
    draw["n_P"] = float(np.clip(draw["n_P"], 1.05, 12.0))   # n_P <= 1 removes the
    draw["h_R"] = float(np.clip(draw["h_R"], 1.0, 8.0))     # cooperativity entirely
    return replace(base, **draw)


def has_empty_state(p: Params) -> bool:
    """Is S = 0 actually an equilibrium at zero exposure for this draw?

    Two conditions, both required (see ``model.initial_state``): the paracrine
    term must vanish faster than linearly at the origin (``n_P > 1``, which
    ``sample_params`` enforces), and basal ROS must not by itself drive
    commitment -- ``R* = r_basal/k_R <= R_ref``.

    The sampler perturbs ``r_basal``, ``k_R`` and ``R_ref`` independently, so a
    substantial minority of draws (122/250 by default) have R* > R_ref and
    accumulate senescence with no exposure at all.

    CAREFUL: losing the exactly empty state does NOT imply a single attractor.
    An earlier version of this docstring said "the tissue has one attractor, the
    senescent one", and that is false -- a small POSITIVE low-senescence
    equilibrium can coexist with a high one, and 55/250 default draws are
    bistable in exactly that way. What this function tests is the stricter
    "empty versus persistent" reading of bistability, which is a legitimate
    definition but a narrower one. Use :func:`two_stable_at_zero` for ordinary
    bistability; ``run_ensemble`` reports both.
    """
    return bool(stress_of(p.r_basal / p.k_R, p) <= 0.0)


def high_state_retained(p: Params, iv: Intervention | None = None,
                        t_probe: float = 600.0) -> bool:
    """FINITE-TIME RETENTION test -- NOT a test for bistability.

    Start from a fully senescent tissue with no exposure and ask whether S is
    still above PERSIST at ``t_probe``. That is a useful and biologically
    direct question ("a tissue that has already tipped, left alone"), but it is
    a strictly weaker statement than "two stable equilibria exist":

    - it can PASS on a long transient that is still decaying at t_probe,
      which is exactly what happens near a fold; and
    - it says nothing about what the second state is, or whether the low state
      it would be bistable *with* still exists.

    This function was previously called ``is_bistable`` and its output was
    reported, labelled and tabulated as a bistability count. It is not one. Use
    :func:`bifurcation.count_stable_at_zero` for equilibrium claims;
    ``run_ensemble`` now reports both, plus the stricter empty-state version.
    """
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
    """TRUE bistability: two or more stable equilibria at zero exposure.

    This is the mathematical statement the word "bistable" makes, and it is the
    one that must back any claim about a tipping point, a separatrix or an
    irreversible transition. It is deliberately independent of
    :func:`high_state_retained`, which only asks whether a high initial burden
    is still high at a finite probe time.

    Note that bistability does NOT require the low state to be exactly empty: a
    small positive low-senescence equilibrium can coexist with a high one. In
    the default ensemble 55/250 draws are bistable despite having no exactly
    empty state, so "the empty state is lost, therefore one attractor" is a
    false inference. :func:`has_empty_state` reports the stricter condition.
    """
    try:
        return count_stable_at_zero(p, iv or Intervention()) >= 2
    except Exception:
        return False


def threshold_dose(p: Params, iv: Intervention | None = None,
                   t_off: float = T_OFF, lo: float = 0.0, hi: float = 4.0,
                   tol: float = 5e-3) -> float | None:
    """Lowest dose that, applied for ``t_off`` days, leaves a persistent state.

    This is the *measurable* threshold -- the one a finite experiment sees --
    and it is not the quasi-static fold. Returns None if no dose in range
    persists, which is the correct answer for a monostable draw.
    """
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
    """Shortest exposure at dose ``E`` that leaves a persistent state.

    The dose partner of :func:`threshold_dose`. Together the two say that
    tipping is a dose x DURATION condition and not a static dose threshold,
    which is the claim the Results make -- so it is computed here rather than
    typed into the manuscript generator.
    """
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
    """ONSET of bistability: the lowest q_sec admitting two stable equilibria.

    Read off :func:`bifurcation.q_sec_bistable_intervals`, which locates the
    saddle-nodes analytically as the interior local extrema of the equilibrium
    curve q*(S). No search over q_sec is involved, which matters because the
    quantity being located can be extremely narrow in q_sec.

    Three successive versions of this function under-reported, and it is worth
    recording why, because each failure looked like a different bug:

    1. It bisected a finite-time RETENTION probe, which near a fold scores a
       slow transient as a second attractor and places the crossing low.
    2. It then gated on the two ENDPOINTS of the range, returning None unless
       the draw was monostable at ``lo`` and bistable at ``hi``. Bistability is
       not monotone in q_sec -- the bistable set is often a WINDOW that closes
       again -- so draws that are monostable at both ends were dropped.
    3. It then bisected a scalar "distance to fold" built from the local
       maximum of F. That is only half the condition: three equilibria need the
       local maximum above zero AND the local minimum below it. It also had to
       be scanned to find a bracket, and the narrowest window in the default
       ensemble (draw 193) is 5.7e-4 wide in q_sec -- 0.104977 to 0.105546 --
       so an 81-point scan stepped over it and the draw was wrongly written off
       as monostable everywhere.

    Returns None only when no bistable interval meets [lo, hi].
    """
    iv = q_sec_bistable_intervals(p, lo=lo, hi=hi)
    return float(iv[0][0]) if iv else None


def q_sec_bistable_window(p: Params, lo: float = 0.01,
                          hi: float = 0.99) -> tuple[bool, float | None]:
    """``(closes_inside_range, upper_edge)`` for the bistable set in q_sec.

    A closing window means bistability DISAPPEARS again at high secondary
    secretion. That is not a return to containment and must never be written up
    as one. What the upper fold destroys is the LOW branch: above it the system
    is monostable at a HIGH burden, with no healthy state left to return to.
    For draw 193 the single stable burden at q_sec = 0.99 is S = 0.812, against
    a bistable window of only (0.104977, 0.105546).

    Losing bistability is therefore ambiguous on its own, and the direction
    matters. Below the lower fold the surviving state is the low one, which is
    containment. Above the upper fold the surviving state is the high one,
    which is the worst case. Any prose reporting "no longer bistable" has to
    say which.
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

        # THREE DISTINCT QUANTITIES. They are not interchangeable and none of
        # them may be labelled "bistable" without saying which one it is.
        #   bs      finite-time retention of a high initial burden (weakest)
        #   two_st  two stable equilibria at E = 0 (ordinary bistability)
        #   bs&empty  an exactly EMPTY stable state plus a persistent one
        #             (strictest; the "empty versus persistent" reading)
        # Default seed-4, n=250 ensemble: 235, 170 and 115 respectively.
        bs = high_state_retained(p)
        bistable.append(bs)
        empty = has_empty_state(p)
        empty_state.append(empty)
        # Enumerate the zero-exposure equilibria ONCE and derive every
        # equilibrium-based statistic from that list, so the three counts
        # cannot drift apart by being computed from different criteria.
        try:
            eq0 = equilibria_at_zero(p)
        except Exception:
            eq0 = []
        stable_S = [e.S for e in eq0 if e.stable]
        two_st = len(stable_S) >= 2
        two_stable.append(two_st)
        # "Empty plus persistent" must mean an EXACTLY EMPTY STABLE EQUILIBRIUM
        # coexisting with a second stable one. It previously read
        # `high_state_retained(p) and has_empty_state(p)`, i.e. a finite-time
        # retention probe AND an analytic condition on basal ROS -- neither of
        # which checks that two equilibria exist. On the default ensemble the
        # two definitions happen to agree exactly (115/250, zero mismatches),
        # so this is correct-by-construction rather than a changed number.
        bistable_two_state.append(
            bool(two_st and any(abs(x) < 1e-12 for x in stable_S)))
        # Counterexamples to "no empty state means one attractor" (55/250).
        bistable_no_empty.append(bool(two_st and not empty))
        thresholds.append(threshold_dose(p))
        qc = q_sec_critical(p)
        q_crits.append(qc)
        reentrant.append(q_sec_bistable_window(p)[0])
        # No crossing found, yet the margin comes within a hair of zero: the
        # draw sits on a fold and the answer is resolution-dependent. Counted
        # rather than silently folded into "no threshold".
        near_fold.append(False)   # exact folds: no undetermined draws remain

        # Exposure ATTRIBUTION. A draw whose unexposed control also runs away
        # to a senescent state is not evidence that the exposure did it: the
        # sampler perturbs basal ROS independently, so R* > R_ref draws climb
        # with no exposure at all, and S = 0 is then a non-equilibrium initial
        # condition the control is simply relaxing away from.
        s_ctrl = runs["control"]["S"][-1]
        s_ref = runs["as"]["S"][-1]
        ctrl_persistent.append(bool(s_ctrl > PERSIST))
        attributable.append(bool(s_ref > PERSIST and s_ctrl <= PERSIST))
        # The low-dose arm is reported as "resolves" on S < PERSIST, which is
        # only "did not exceed the persistence threshold". Near-baseline is a
        # different and stricter statement, and the intervention arms use it.
        lowdose_baseline.append(bool(runs["as_low"]["S"][-1] < 0.05))

        for key in ARMS:
            for o in OBSERVABLES:
                end[key][o].append(float(runs[key][o][-1]))
                d14[key][o].append(float(runs[key][o][i14]))

        # V3: is the marker/state discordance robust, or an artefact of the
        # nominal magnitudes? Scale-free form -- does SA-beta-gal overstate the
        # latent state MORE during exposure than after it? The earlier fixed
        # "3x" threshold conflated the question with how fast S happened to
        # accumulate in a given draw.
        # Absolute, not ratio. A ratio m_sabg/S is numerically degenerate here:
        # in the arm that RESOLVES, S -> 0 at the endpoint, so the ratio
        # explodes and the test measures the denominator rather than any
        # discordance. Overstatement in percentage points is what an
        # experimenter would actually misread.
        #
        # TWO SCALES, and the difference between them is the finding.
        # ``over`` is the ABSOLUTE marker-minus-state gap. It cannot fall below
        # the marker's unexposed false-positive rate (m_sabg.fp = 0.08) in any
        # draw where the low dose does nothing, because ``fp`` is a structural
        # constant and is NOT in FREE_SIGMA. Reporting only ``over`` therefore
        # buys a "sign-robust, never touches zero" result for free: the lower
        # tail of its interval IS that constant, and it is present in the
        # concurrent control at the same instant.
        # ``over_net`` subtracts the concurrent control, exactly as
        # ``apply_control_relative_lnN`` does for KCC10, and is the
        # EXPOSURE-ATTRIBUTABLE overstatement. It is the number the manuscript
        # claim has to rest on; it is materially smaller and it does touch zero.
        over = runs["as_low"]["m_sabg"][i14] - runs["as_low"]["S"][i14]
        over_ctrl = runs["control"]["m_sabg"][i14] - runs["control"]["S"][i14]
        v3_overstate.append(float(over))
        v3_overstate_net.append(float(over - over_ctrl))
        v3_discordant.append(bool(over > 0.10))
        # Understatement of persistence: in the arm that PERSISTS, does the
        # marker fall while the latent state does not?
        v3_understate.append(bool(
            runs["as"]["m_sabg"][-1] < runs["as"]["m_sabg"][i14] - 0.02
            and runs["as"]["S"][-1] >= runs["as"]["S"][i14] - 0.02))
        kS_draw.append(float(p.k_S))

        # V7: the paired contrast is only meaningful where the senolytic has
        # NOT already collapsed the state -- otherwise it compares 0 with 0 and
        # reports numerical noise as a failed prediction.
        seno_end = runs["as_seno"]["S"][-1]
        sasp_end = runs["as_sasp"]["S"][-1]
        v7_sasp_collapses.append(bool(sasp_end < 0.05))
        v7_seno_survives.append(bool(seno_end > 0.20))
        if seno_end > 0.05:                      # comparison is non-degenerate
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
        # --- the three bistability quantities, never conflated -------------
        # Weakest: a high initial burden is still high at the probe time.
        "frac_high_state_retained": float(np.mean(bistable)) if bistable else np.nan,
        # Ordinary bistability: two stable equilibria at E = 0.
        "frac_two_stable_equilibria": float(np.mean(two_stable)) if two_stable else np.nan,
        # Strictest: an exactly EMPTY stable state plus a persistent one.
        "frac_empty_plus_persistent": float(np.mean(bistable_two_state)) if bistable_two_state else np.nan,
        "frac_empty_state_lost": float(1.0 - np.mean(empty_state)) if empty_state else np.nan,
        # Draws that are bistable ANYWAY, with no exactly empty state. These
        # refute "the empty state is lost, so there is one attractor".
        "frac_bistable_without_empty_state": float(
            np.mean(bistable_no_empty)) if bistable_no_empty else np.nan,
        # --- persistence, descriptive and exposure-attributable ------------
        "frac_persist_at_reference": float(
            np.mean([s > PERSIST for s in end["as"]["S"]])) if ok else np.nan,
        # The concurrent control runs away too in this many draws.
        "frac_control_persistent": float(
            np.mean(ctrl_persistent)) if ctrl_persistent else np.nan,
        # Paired: exposed arm persistent AND its matched control is not. This
        # is the count that supports an exposure claim.
        "frac_persist_exposure_attributable": float(
            np.mean(attributable)) if attributable else np.nan,
        # "Did not exceed PERSIST" -- NOT the same as resolved to baseline.
        "frac_lowdose_resolves": float(
            np.mean([s < PERSIST for s in end["as_low"]["S"]])) if ok else np.nan,
        "frac_lowdose_near_baseline": float(
            np.mean(lowdose_baseline)) if lowdose_baseline else np.nan,
        "threshold_dose": band(thr),
        "frac_with_threshold": len(thr) / ok if ok else np.nan,
        # CONDITIONAL summaries: draws with no detected crossing in the search
        # range return None and are excluded, so quote the denominator.
        "q_sec_critical": band(qcs),
        "n_with_threshold": len(thr),
        "n_with_q_sec_critical": len(qcs),
        "frac_with_q_sec_critical": len(qcs) / ok if ok else np.nan,
        # Bistability is NOT monotone in q_sec: in these draws it appears and
        # then disappears again as secondary secretion rises, so "above q_sec*
        # the system tips" is false above the window's upper edge.
        "frac_q_sec_reentrant": float(np.mean(reentrant)) if reentrant else np.nan,
        "n_q_sec_reentrant": int(np.sum(reentrant)) if reentrant else 0,
        # Draws with no detected crossing that nonetheless sit on a fold: the
        # existence of a bistable window is resolution-dependent for these and
        # is not asserted either way.
        "n_q_sec_near_fold_undetermined": int(np.sum(near_fold)) if near_fold else 0,
        "frac_v3_discordant": float(np.mean(v3_discordant)) if v3_discordant else np.nan,
        # Absolute scale: inflated by the assay's unexposed background.
        "v3_overstate": band(v3_overstate),
        # Net of the concurrent control: the exposure-attributable part, and
        # the only one of the two that is a claim about the exposure.
        "v3_overstate_net": band(v3_overstate_net),
        "frac_v3_overstate_net_positive": (
            float(np.mean([x > 0.0 for x in v3_overstate_net]))
            if v3_overstate_net else np.nan),
        # How much of the absolute gap is just the unexposed background.
        "v3_overstate_background": band(
            [a - b for a, b in zip(v3_overstate, v3_overstate_net)]),
        "frac_v3_understate": float(np.mean(v3_understate)) if v3_understate else np.nan,
        "frac_v7_contrast": float(np.mean(v7_contrast)) if v7_contrast else np.nan,
        "n_v7_comparable": len(v7_contrast),
        "frac_v7_sasp_collapses": float(np.mean(v7_sasp_collapses)) if v7_sasp_collapses else np.nan,
        "frac_v7_seno_survives": float(np.mean(v7_seno_survives)) if v7_seno_survives else np.nan,
        "raw_end": end,
    }
