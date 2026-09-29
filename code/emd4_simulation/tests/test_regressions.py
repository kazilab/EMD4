"""Regressions for the EMD4 build.

The one that matters most is `test_qss_check_is_not_a_tautology`. The published
build shipped a `quasi_steady_state_check` that rebuilt the reduced right-hand
side from the same algebra as `manuscript_rhs` and compared the two. It agreed
to 0.00e+00 in every run, and it would have agreed to 0.00e+00 no matter what
the real ODE did, because it never evaluated `rhs`. The README quoted that
0.00e+00 as evidence that the reformulation was faithful. Nothing failed; the
number was simply meaningless. These tests make that class of error loud.

The rest lock the contracts that the manuscript pipeline depends on: that the
battery cannot pass on skipped evidence, that the empty-state condition is
stated as it actually is, and that the numbers in the README and the docx come
from the summary JSON rather than from someone's memory.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from emd4_simulation.conditions import BY_KEY, CHECKS, PERSIST, T_END, T_OFF
from emd4_simulation.ensemble import (FREE_SIGMA, has_empty_state,
                                      high_state_retained, two_stable_at_zero,
                                      sample_params, threshold_dose,
                                      threshold_duration)
from emd4_simulation.model import (IDX, MARKER_BY_KEY, Intervention, Params,
                                   apply_control_relative_lnN,
                                   escape_residual_check, manuscript_rhs,
                                   qss_state, quasi_steady_state_check, rhs,
                                   simulate, stress_of, trace)
from emd4_simulation.identifiability import _summarise
from emd4_simulation.manuscript_tables import audit
from emd4_simulation.spatial import (SpatialParams, _boundary_mask, _hex_coords,
                                     escape_crossing, wilson_interval)
from emd4_simulation.run import json_safe, run_nominal, t50
import emd4_simulation.bifurcation as bf

P = Params()
OUT = Path(__file__).resolve().parents[2] / "figures" / "output"


# --- the bug that motivated this suite ------------------------------------

def test_qss_check_actually_evaluates_the_ode() -> None:
    """The QSS check must read from `rhs`, not re-derive the reduced form.

    Perturb the ODE in a way that `manuscript_rhs` cannot see. If the check is
    real the residual moves; if it is a tautology it stays at zero.
    """
    q = replace(P, tau_C=1e-6, d_P=1e6, k_sasp=1e6, q_sec=1.0, k_esc=0.0)
    iv = Intervention()
    S, R = 0.5, 0.6
    y = qss_state(S, R, q)

    dy = rhs(0.0, y, q, iv, 0.0, np.inf)
    baseline = abs((dy[IDX["S1"]] + dy[IDX["S2"]])
                   - manuscript_rhs(S, stress_of(R, q), q))
    assert baseline < 1e-9, "the limit should reduce to the manuscript form"

    # A model whose paracrine gain differs must NOT still agree.
    q_bad = replace(q, k_p=q.k_p * 1.10)
    y_bad = qss_state(S, R, q_bad)
    dy_bad = rhs(0.0, y_bad, q_bad, iv, 0.0, np.inf)
    moved = abs((dy_bad[IDX["S1"]] + dy_bad[IDX["S2"]])
                - manuscript_rhs(S, stress_of(R, q), q))
    assert moved > 1e-3, (
        "changing the ODE left the QSS residual unmoved -- the check is "
        "comparing an expression with itself again")


def test_qss_is_sensitive_to_each_limit() -> None:
    """Every clause of the stated limit must be load-bearing."""
    ok, worst = quasi_steady_state_check(P)
    assert ok and worst < 1e-9

    # escape left on: residual must appear, and must be exactly the drain
    ok_esc, worst_esc = escape_residual_check(P)
    assert ok_esc and worst_esc < 1e-9

    iv = Intervention()
    lim = dict(tau_C=1e-6, d_P=1e6, k_sasp=1e6, q_sec=1.0)

    def worst_residual(q, off_manifold=False):
        w = 0.0
        for S in np.linspace(0.0, 0.99, 20):
            y = qss_state(S, 0.6, q)
            if off_manifold:
                y[IDX["P"]] *= 0.5
            dy = rhs(0.0, y, q, iv, 0.0, np.inf)
            w = max(w, abs((dy[IDX["S1"]] + dy[IDX["S2"]])
                           - manuscript_rhs(S, stress_of(0.6, q), q)))
        return w

    assert worst_residual(replace(P, **lim)) > 1e-4, "k_esc must matter"
    assert worst_residual(replace(P, **{**lim, "q_sec": 0.5},
                                  k_esc=0.0)) > 1e-3, "q_sec=1 must matter"
    assert worst_residual(replace(P, **lim, k_esc=0.0),
                          off_manifold=True) > 1e-3, "the manifold must matter"


# --- the empty state ------------------------------------------------------

def test_empty_state_condition_is_stated_correctly() -> None:
    """S = 0 is an equilibrium only when basal ROS does not drive commitment."""
    assert has_empty_state(P), "nominal parameters must have an empty state"

    # Nudge basal ROS above the reference: the empty state must be destroyed.
    hot = replace(P, r_basal=P.R_ref * P.k_R * 1.5)
    assert not has_empty_state(hot)

    t = np.linspace(0.0, 400.0, 401)
    y = simulate(hot, Intervention(), 0.0, t)
    S_end = float(y[IDX["S1"], -1] + y[IDX["S2"], -1])
    assert S_end > 1e-3, (
        "with R* > R_ref, unexposed tissue must accumulate senescence; "
        f"got S = {S_end:.2e}")


def test_ensemble_prior_loses_the_empty_state_often_enough_to_report() -> None:
    """If this ever drops near zero the disclosure can go -- until then it stays."""
    rng = np.random.default_rng(4)
    lost = sum(not has_empty_state(sample_params(rng)) for _ in range(250))
    assert lost > 25, (
        "the empty state now survives almost every draw; the bistability "
        "counts no longer need decomposing and the manuscript should be "
        "revisited")


# --- the battery must not pass on evidence that was skipped ---------------

def _ctx_stub(**over):
    """Minimal context where every check passes, so one clause can be broken."""
    end = {k: {"S": 0.9, "X": 0.5, "m_sabg": 0.88, "secondary_fraction": 0.99,
               "R": 0.3, "stress": 0.0, "lnN": 8.0}
           for k in BY_KEY}
    end["as_qsec_lo"]["S"] = 0.002
    end["as_immune_lo"]["S"] = 0.97
    end["as_immune_hi"]["S"] = 0.82
    end["as_escape"] = {**end["as_escape"], "S": 0.88, "X": 5.2}
    end["as_sasp"] = {**end["as_sasp"], "S": 0.0001, "lnN": 1.8}
    end["as_seno"]["S"] = 0.62
    end["as_low"]["lnN"] = 0.3
    end["control"] = {**end["control"], "S": 0.0, "R": 0.3, "lnN": 0.0}
    ctx = {
        "end": end,
        "d14": {k: {"S": 0.05, "m_sabg": 0.19, "m_ddr": 0.17, "R": 1.5,
                    "stress": 0.8} for k in BY_KEY},
        # The control arm carries the assay background and nothing else. V3
        # nets this off, so a stub that gave control the same numbers as the
        # exposed arm would make the netted clause vacuously zero.
        "d14_control": {"S": 0.0, "m_sabg": 0.08, "m_ddr": 0.04, "R": 0.3,
                        "stress": 0.0},
        "d5": {k: {"S": 0.01, "stress": 0.79} for k in BY_KEY},
        "peak_t": {k: {"S1": 1.0, "S2": 5.0, "P": 9.0} for k in BY_KEY},
        "null": {"endpoint_r2": 0.996, "timecourse_r2": 0.884,
                 "cascade_r2": 0.902, "simple_end": 2.7e-9,
                 "cascade_end": 4.3e-16, "worst_dev_simple": 0.54,
                 "worst_dev_cascade": 0.53},
        "bif": {"width": 0.068, "fold": 0.070, "q_crit": 0.243},
        "spatial": {"q_contained": 0.15, "q_majority_escape": 0.30},
        "ident": {"kp_ridge_width": 1.53, "kp_ridge_fold": 4.0,
                  "kp_ridge_censored": False, "null_admissible": False},
    }
    ctx["d14"]["control"] = ctx.pop("d14_control")
    ctx.update(over)
    return ctx


def _run(cid, ctx):
    chk = next(c for c in CHECKS if c.cid == cid)
    try:
        return bool(chk.test(ctx))
    except Exception:
        return False


def test_v10_fails_when_the_spatial_layer_was_skipped() -> None:
    """--no-spatial must not yield a passing V10."""
    assert _run("V10", _ctx_stub()) is True
    assert _run("V10", _ctx_stub(spatial=None)) is False, (
        "V10 passed with no spatial evidence at all")


def test_v11_tests_every_clause_of_its_statement() -> None:
    """The statement claims the time course separates the models; assert it."""
    assert _run("V11", _ctx_stub()) is True
    ctx = _ctx_stub()
    ctx["null"] = {**ctx["null"], "timecourse_r2": 0.999,
                   "worst_dev_simple": 0.001}
    assert _run("V11", ctx) is False, (
        "V11 passed while a null also matched the time course, which its "
        "statement says is the discriminating design")


def test_v3_does_not_assert_the_fragile_direction() -> None:
    """V3 must not claim the marker falls after withdrawal (111/250 draws)."""
    ctx = _ctx_stub()
    # Marker flat after withdrawal -- the fragile clause would fail here.
    ctx["end"] = {k: {**v, "m_sabg": 0.19} for k, v in ctx["end"].items()}
    assert _run("V3", ctx) is True, (
        "V3 still depends on the post-withdrawal fall, which the ensemble "
        "rejects at 111/250")


# --- KCC10 relative neighbour expansion -----------------------------------

def test_reported_lnN_is_relative_to_concurrent_control() -> None:
    """Raw control drifts as -d_N*t; the reported observable cancels that."""
    t = np.linspace(0.0, T_END, int(T_END * 4) + 1)
    runs = run_nominal(P, t)
    assert runs["control"]["lnN"][-1] == pytest.approx(0.0, abs=1e-9)
    assert runs["as"]["lnN"][-1] == pytest.approx(8.29, abs=0.05)
    assert runs["as"]["lnN"][-1] > runs["as_low"]["lnN"][-1]
    assert runs["as"]["lnN"][-1] > runs["as_sasp"]["lnN"][-1]

    # The raw ODE still drifts: absolute control is -d_N * T_END.
    y = simulate(P, Intervention(), 0.0, t)
    assert y[IDX["lnN"], -1] == pytest.approx(-P.d_N * T_END, abs=1e-9)


def test_v13_requires_sasp_conditional_relative_expansion() -> None:
    assert _run("V13", _ctx_stub()) is True
    broken = _ctx_stub()
    broken["end"]["as"]["lnN"] = broken["end"]["as_low"]["lnN"]
    assert _run("V13", broken) is False
    broken = _ctx_stub()
    broken["end"]["control"]["lnN"] = -1.6
    assert _run("V13", broken) is False, (
        "V13 must fail if lnN was left on the absolute (drifting) scale")


def test_apply_control_relative_lnN_is_idempotent_on_already_relative() -> None:
    """Calling the helper twice must not double-subtract."""
    t = np.linspace(0.0, 10.0, 11)
    raw = {
        "control": {"lnN": -P.d_N * t},
        "as": {"lnN": 0.5 - P.d_N * t},
    }
    once = apply_control_relative_lnN(raw)
    twice = apply_control_relative_lnN(once)
    # Second call subtracts control again; control is 0 so as is unchanged.
    assert twice["control"]["lnN"][-1] == pytest.approx(0.0)
    assert twice["as"]["lnN"][-1] == pytest.approx(once["as"]["lnN"][-1])


# --- the marker observation layer -----------------------------------------

def test_sasp_neutralisation_is_time_gated_in_trace() -> None:
    """trace() must apply the handle only from t_start, exactly as rhs does."""
    c = BY_KEY["as_sasp"]
    t = np.linspace(0.0, T_END, int(T_END * 4) + 1)
    y = simulate(P, c.iv, c.E, t, t_off=c.t_off)
    tr = trace(y, P, c.iv, t)
    before = int(np.searchsorted(t, c.iv.t_start - 10.0))
    after = int(np.searchsorted(t, c.iv.t_start + 10.0))
    assert tr["P"][before] == pytest.approx(y[IDX["P"]][before], rel=1e-9), (
        "P was scaled before the intervention starts")
    assert tr["P"][after] == pytest.approx(
        y[IDX["P"]][after] * c.iv.sasp_neut, rel=1e-9)


# --- README and manuscript contracts --------------------------------------

@pytest.mark.skipif(not (OUT / "emd4_simulation_summary.json").exists(),
                    reason="run emd4_simulation.run first")
def test_readme_numbers_match_the_summary() -> None:
    """Every number the README quotes must come from the JSON, not from memory."""
    s = json.loads((OUT / "emd4_simulation_summary.json").read_text())
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    ens = s["ensemble"]
    for label, value, fmt in [
        ("fold", s["bifurcation"]["fold"], "{:.4f}"),
        ("hysteresis area", s["bifurcation"]["hysteresis_width"], "{:.4f}"),
        ("q_sec*", s["bifurcation"]["q_sec_critical"], "{:.3f}"),
        ("recovery ratio", s["bifurcation"]["recovery_time_ratio"], "{:.1f}"),
        # The quantities the audit corrected. These are the ones most likely to
        # go stale, because the README argues about them at length.
        ("k_p ridge fold, continuous",
         s["identifiability"]["kp_ridge_fold_continuous"], "{:.0f}-fold"),
        ("k_p ridge fold, withdrawal-only design",
         s["identifiability"]["kp_ridge_fold_withdrawal_only"], "{:.0f}-fold"),
        ("netted SA-beta-gal over-read",
         ens["v3_overstate_points_net"]["median"] * 100, "{:.1f}"),
        ("fraction where the netted over-read is positive",
         ens["frac_v3_overstate_net_positive"] * 100, "{:.1f}%"),
        ("spatial replicate count", s["spatial"]["n_rep"], "{:d}"),
        ("majority-escape q_sec", s["spatial"]["q_majority_escape"], "{:.2f}"),
    ]:
        assert fmt.format(value) in readme, (
            f"README does not quote the current {label} "
            f"({fmt.format(value)}); it has gone stale")


@pytest.mark.skipif(not (OUT / "emd4_simulation_summary.json").exists(),
                    reason="run emd4_simulation.run first")
def test_summary_archives_every_quantity_the_prose_quotes() -> None:
    """Nothing quoted in prose may live only in a print statement."""
    s = json.loads((OUT / "emd4_simulation_summary.json").read_text())
    for key in ("finite_threshold", "qss_reduction", "derived_quantities"):
        assert key in s, f"{key} missing from the summary JSON"
    d = s["derived_quantities"]
    for key in ("null_hill_coefficient", "normal_form_r2",
                "critical_slowing_slope_asymptotic",
                "sabg_overstatement_d14_low_dose"):
        assert d.get(key) is not None, f"derived_quantities.{key} not archived"
    e = s["ensemble"]
    for key in ("frac_empty_state_lost", "frac_empty_plus_persistent",
                "frac_high_state_retained", "frac_two_stable_equilibria",
                "frac_bistable_without_empty_state",
                "frac_control_persistent", "frac_persist_exposure_attributable",
                "frac_lowdose_near_baseline",
                "frac_v7_contrast", "n_v7_comparable",
                "v3_overstate_points_net", "v3_overstate_background",
                "frac_v3_overstate_net_positive"):
        assert key in e, f"ensemble.{key} not archived"

    # The absolute marker scale must never be archived alone: on its own it
    # licenses a "does not touch zero" claim that the fp constant guarantees.
    assert "v3_overstate_points" in e and "v3_overstate_points_net" in e

    for key in ("kp_ridge_fold_continuous", "kp_ridge_censored_continuous",
                "kp_admissible_lo_continuous", "kp_admissible_hi_continuous",
                "kp_ridge_fold_withdrawal_only", "kp_ridge_fold_joint_pooled",
                "kp_ridge_fold_joint_perarm",
                "kp_admissible_lo_joint_perarm",
                "kp_admissible_hi_joint_perarm"):
        assert key in s["identifiability"], f"identifiability.{key} not archived"
    assert s["identifiability"]["kp_ridge_censored_continuous"] is False, (
        "the k_p profile grid truncated the ridge; widen KP_GRID_HI")

    for key in ("n_rep", "ci_lo", "ci_hi", "frac_intermediate",
                "max_frac_intermediate", "q_majority_escape"):
        assert key in s["spatial"], f"spatial.{key} not archived"
    assert s["spatial"]["n_rep"] >= 40, (
        "the containment scan ran at fewer than 40 replicates; at 5 the escape "
        "fraction is quantised to 0.2 and band edges are sampling accidents")


# --- finite-experiment thresholds -----------------------------------------

def test_dose_and_duration_thresholds_bracket_the_published_claims() -> None:
    """The Results quote these; they must be derived, and they must hold."""
    td = threshold_dose(P, Intervention(), t_off=T_OFF)
    tt = threshold_duration(P, Intervention(), E=1.0)
    assert td is not None and tt is not None

    t = np.linspace(0.0, T_END, int(T_END * 4) + 1)

    def S_end(E, t_off):
        y = simulate(P, Intervention(), E, t, t_off=t_off)
        return float(y[IDX["S1"], -1] + y[IDX["S2"], -1])

    assert S_end(td * 0.9, T_OFF) < PERSIST < S_end(td * 1.1, T_OFF)
    assert S_end(1.0, tt * 0.9) < PERSIST < S_end(1.0, tt * 1.1)
    assert td > 2.0 * 0.070, (
        "the 14-day threshold dose should sit well above the quasi-static "
        "fold; if it does not, the dose x duration claim is wrong")


# --- A1: the marker over-read must survive control subtraction ------------

def test_v3_requires_the_over_read_to_survive_control_subtraction() -> None:
    """The absolute gap includes the assay background; V3 must not accept it.

    ``m - S`` cannot fall below the marker's unexposed false-positive rate in
    an arm where nothing happens, and ``fp`` is a structural constant the
    ensemble does not sample. A check written on the absolute scale alone
    therefore passes on the assay's background rather than on the exposure.
    """
    assert _run("V3", _ctx_stub()) is True

    # Exposure adds NOTHING beyond what the control already reads: the
    # absolute over-read is still large (0.19 - 0.05), but all of it is
    # background. V3 must fail.
    ctx = _ctx_stub()
    ctx["d14"] = {**ctx["d14"]}
    ctx["d14"]["control"] = {"S": 0.0, "m_sabg": 0.14, "m_ddr": 0.12,
                             "R": 0.3, "stress": 0.0}
    assert _run("V3", ctx) is False, (
        "V3 passed on an over-read entirely explained by the unexposed "
        "background the concurrent control carries at the same instant")


def test_ensemble_reports_the_marker_over_read_on_both_scales() -> None:
    """The netted scale must exist, and must be smaller than the absolute one."""
    from emd4_simulation.ensemble import run_ensemble
    ens = run_ensemble(n=30, seed=4)
    for key in ("v3_overstate", "v3_overstate_net", "v3_overstate_background",
                "frac_v3_overstate_net_positive"):
        assert key in ens, f"{key} missing from the ensemble result"
    assert ens["v3_overstate_net"]["median"] < ens["v3_overstate"]["median"], (
        "netting the concurrent control off did not reduce the over-read; "
        "the background term has gone missing")
    # The control arm reads the marker's unexposed false-positive rate and
    # nothing else whenever basal ROS is below the commitment threshold, so
    # the background term is centred on m_sabg.fp. That constant is what the
    # absolute scale carries into every draw, and it is not sampled.
    fp = MARKER_BY_KEY["m_sabg"].fp
    assert ens["v3_overstate_background"]["median"] == pytest.approx(fp, abs=5e-3), (
        "the background term is no longer the marker fp; if the observation "
        "model changed, the disclosure text needs revisiting")
    assert ens["v3_overstate_background"]["lo"] > 0.0, (
        "the sampled absolute scale no longer carries the expected positive "
        "background contribution")


def test_readme_contains_no_superseded_identifiability_claims() -> None:
    """Withdrawn interpretations must not reappear below their correction."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    stale = (
        "Adding the withdrawal arm makes the ridge",
        "range once the withdrawal arm is added",
        "supported by ordinary data",
        "reduces to one measurable quantity",
    )
    assert not [phrase for phrase in stale if phrase in readme]


# --- A2/A3/B3/B4: the spatial layer ---------------------------------------

def test_escape_crossing_returns_none_rather_than_the_smallest_q() -> None:
    """A scan that never reaches 50% escape has no crossing.

    ``q[argmax(escaped > 0.5)]`` returns index 0 on an all-False array, so a
    fully contained scan reported its SMALLEST q as the 50%-escape point.
    """
    q = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    assert escape_crossing(q, np.array([0.0, 0.0, 0.0, 0.2, 0.4])) is None
    # and when it does cross, it interpolates rather than snapping to the grid
    x = escape_crossing(q, np.array([0.0, 0.0, 0.25, 0.75, 1.0]))
    assert x == pytest.approx(0.175, abs=1e-9)


def test_v10_fails_when_a_band_edge_was_not_determined() -> None:
    """The band edges used to default to 0.0 and 1.0, which nothing can fail."""
    assert _run("V10", _ctx_stub()) is True
    assert _run("V10", _ctx_stub(
        spatial={"q_contained": None, "q_majority_escape": 0.30})) is False
    assert _run("V10", _ctx_stub(
        spatial={"q_contained": 0.15, "q_majority_escape": None})) is False


def test_reached_edge_is_ring_membership_not_a_radius_threshold() -> None:
    """The hex embedding is anisotropic; a radius threshold mis-scores it.

    Row spacing is sqrt(3)/2 of column spacing, so on a 41x41 lattice the
    half-height is 17.32 while the old criterion fired at 0.45*41 = 18.45. A
    lesion spanning top to bottom scored as contained.
    """
    nx = ny = 41
    co = _hex_coords(nx, ny)
    ring = _boundary_mask(nx, ny)
    centre = int(np.argmin(np.linalg.norm(co - co.mean(axis=0), axis=1)))
    d = np.linalg.norm(co - co[centre], axis=1)

    half_height = np.abs(co[:, 1] - co[centre, 1]).max()
    assert half_height < 0.45 * nx, (
        "the old radius threshold no longer exceeds the half-height; this "
        "regression is only meaningful while it does")

    # A top-row cell is on the ring, but its radius is below the old threshold.
    top = np.flatnonzero(ring & (d < 0.45 * nx))
    assert len(top) > 0
    assert ring[top[0]], "a boundary cell must count as having reached the edge"


def test_wilson_interval_has_width_at_the_extremes() -> None:
    """5/5 is not certainty. The normal approximation would call it that."""
    lo, hi = wilson_interval(5, 5)
    assert lo < 0.99 and hi == pytest.approx(1.0)
    assert hi - lo > 0.3, "a five-for-five result must not read as certain"
    lo40, hi40 = wilson_interval(32, 40)
    assert lo40 > lo, "more replicates must tighten the interval"


# --- B1: the identifiability profile grid ---------------------------------

def test_censored_profile_grid_is_flagged_and_gated() -> None:
    """A ridge that runs off the top of the grid is a lower bound, not a range."""
    p = Params()
    grid = np.linspace(0.0, 2.0 * p.k_p, 21)

    # rmse still inside the noise floor at the top of the grid -> censored
    censored = _summarise(grid, np.full_like(grid, 0.001), 0.02, p)
    assert censored["kp_ridge_censored"] is True

    # rmse leaves the floor before the top -> not censored
    rmse = np.linspace(0.001, 0.05, len(grid))
    assert _summarise(grid, rmse, 0.02, p)["kp_ridge_censored"] is False

    # and the manuscript gate refuses a censored profile
    tables = {"checks": [], "identifiability": {
        "kp_ridge_censored_continuous": True,
        "kp_ridge_censored_with_withdrawal": False}}
    problems = audit(tables)
    assert any("kp_ridge_censored_continuous" in why for why in problems)


def test_manuscript_gate_requires_the_netted_marker_scale() -> None:
    tables = {"checks": [], "identifiability": {},
              "ensemble": {"frac_empty_state_lost": 0.488,
                           "frac_empty_plus_persistent": 0.46,
                           "frac_high_state_retained": 0.94,
                           "frac_two_stable_equilibria": 0.68,
                           "frac_bistable_without_empty_state": 0.22,
                           "frac_control_persistent": 0.256,
                           "frac_persist_exposure_attributable": 0.668}}
    problems = audit(tables)
    assert any("v3_overstate_points_net" in why for why in problems)


# --- B2: t50 is a half-rise time ------------------------------------------

def test_t50_is_a_half_rise_time_on_a_rise_then_decay_trajectory() -> None:
    """Half of the PEAK, not half of a decay remnant."""
    t = np.linspace(0.0, 100.0, 1001)
    # rises to 1.0 at t = 20, then decays to ~0
    v = np.where(t <= 20.0, t / 20.0, np.exp(-(t - 20.0) / 5.0))
    assert t50(t, v) == pytest.approx(10.0, abs=0.2), (
        "t50 must report the half-rise (t = 10), not the time to cross half "
        "of the decayed final value")

    # on a plateau the two definitions agree, which is the case it was for
    plateau = np.clip(t / 20.0, 0.0, 1.0)
    assert t50(t, plateau) == pytest.approx(10.0, abs=0.2)


def test_t50_on_the_real_withdrawal_arm_reports_the_rise() -> None:
    t = np.linspace(0.0, T_END, int(T_END * 4) + 1)
    runs = run_nominal(P, t)
    s1 = np.asarray(runs["as"]["S1"])
    assert s1[-1] < 1e-4 < s1.max(), "S1 must rise and then decay in this arm"
    assert t50(t, s1) > 2.0, (
        "t50 collapsed to ~0 on a decaying arm, which is the final-value bug")


# --- B5: the hysteresis area ----------------------------------------------

def test_hysteresis_area_is_grid_converged() -> None:
    """The integrand jumps at the fold, so the area needs a fine enough grid."""
    iv = Intervention()
    a121 = bf.hysteresis_loop(P, iv, E_max=0.15, n=121, dwell=600.0)["area"]
    a181 = bf.hysteresis_loop(P, iv, E_max=0.15, n=181, dwell=600.0)["area"]
    assert abs(a121 - a181) / a181 < 0.02, (
        f"hysteresis area not converged: n=121 gives {a121:.5f}, "
        f"n=181 gives {a181:.5f}")
    # and it is near the analytic estimate S_upper * E_fold
    assert a121 == pytest.approx(0.9240 * 0.07018, rel=0.05)


def test_sweep_returns_the_final_state_so_the_up_sweep_runs_once() -> None:
    E = np.linspace(0.0, 0.15, 5)
    S, y_end = bf.sweep(P, Intervention(), E, dwell=100.0)
    assert S.shape == E.shape
    assert y_end.shape == (len(IDX),)


# --- the summary JSON must be readable by the docx generators -------------

def test_summary_json_is_valid_json_for_javascript() -> None:
    """A bare NaN is not valid JSON and takes the whole manuscript build down.

    ``json.dumps`` writes NaN and Infinity happily and Python reads them back,
    so this fails only in the docx generators, which use ``JSON.parse``. One
    undefined quantity anywhere in the summary -- an occupied fraction where no
    lesion survived, for instance -- would reject the entire file.
    """
    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe({"a": [np.nan, 2.0]}) == {"a": [None, 2.0]}

    # allow_nan=False is what makes it a hard error rather than a silent one
    with pytest.raises(ValueError):
        json.dumps({"x": float("nan")}, allow_nan=False)
    json.dumps(json_safe({"x": float("nan")}), allow_nan=False)


@pytest.mark.skipif(not (OUT / "emd4_simulation_summary.json").exists(),
                    reason="run emd4_simulation.run first")
def test_written_summaries_contain_no_bare_nan() -> None:
    for name in ("emd4_simulation_summary.json", "emd4_manuscript_tables.json"):
        path = OUT / name
        if not path.exists():
            continue
        text = path.read_text()
        assert "NaN" not in text and "Infinity" not in text, (
            f"{name} contains a bare NaN/Infinity; JSON.parse in the docx "
            "generators will reject the whole file")
        json.loads(text, parse_constant=_reject_constant)


def _reject_constant(name):
    raise AssertionError(f"summary JSON contains the constant {name}")
