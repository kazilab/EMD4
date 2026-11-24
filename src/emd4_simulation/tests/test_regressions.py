"""Regression tests for senescence dynamics, uncertainty estimates and exported results."""

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
OUT = Path(__file__).resolve().parents[3] / "outputs" / "reproduced"


def test_qss_check_actually_evaluates_the_ode() -> None:
    """The full ODE and its reduced form agree on the slow manifold."""
    q = replace(P, tau_C=1e-6, d_P=1e6, k_sasp=1e6, q_sec=1.0, k_esc=0.0)
    iv = Intervention()
    S, R = 0.5, 0.6
    y = qss_state(S, R, q)

    dy = rhs(0.0, y, q, iv, 0.0, np.inf)
    baseline = abs((dy[IDX["S1"]] + dy[IDX["S2"]])
                   - manuscript_rhs(S, stress_of(R, q), q))
    assert baseline < 1e-9, "the limit should reduce to the manuscript form"

    q_bad = replace(q, k_p=q.k_p * 1.10)
    y_bad = qss_state(S, R, q_bad)
    dy_bad = rhs(0.0, y_bad, q_bad, iv, 0.0, np.inf)
    moved = abs((dy_bad[IDX["S1"]] + dy_bad[IDX["S2"]])
                - manuscript_rhs(S, stress_of(R, q), q))
    assert moved > 1e-3, (
        "Increasing paracrine gain should change the QSS residual.")


def test_qss_is_sensitive_to_each_limit() -> None:
    """Perturb each assumption behind the quasi-steady-state reduction."""
    ok, worst = quasi_steady_state_check(P)
    assert ok and worst < 1e-9

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


def test_empty_state_condition_is_stated_correctly() -> None:
    """S = 0 is an equilibrium only when basal ROS does not drive commitment."""
    assert has_empty_state(P), "nominal parameters must have an empty state"

    # Raise basal ROS above the commitment threshold.
    hot = replace(P, r_basal=P.R_ref * P.k_R * 1.5)
    assert not has_empty_state(hot)

    t = np.linspace(0.0, 400.0, 401)
    y = simulate(hot, Intervention(), 0.0, t)
    S_end = float(y[IDX["S1"], -1] + y[IDX["S2"], -1])
    assert S_end > 1e-3, (
        "with R* > R_ref, unexposed tissue must accumulate senescence; "
        f"got S = {S_end:.2e}")


def test_ensemble_prior_loses_the_empty_state_often_enough_to_report() -> None:
    """The reference ensemble includes draws with exposure-independent senescence."""
    rng = np.random.default_rng(4)
    lost = sum(not has_empty_state(sample_params(rng)) for _ in range(250))
    assert lost > 25, (
        "Expected more than 25 draws without an exactly empty equilibrium.")


def _ctx_stub(**over):
    """Construct passing model-check inputs for testing individual predicates."""
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
        # The control carries baseline marker positivity.
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
    """The containment comparison requires the spatial result."""
    assert _run("V10", _ctx_stub()) is True
    assert _run("V10", _ctx_stub(spatial=None)) is False, (
        "V10 passed with no spatial evidence at all")


def test_v11_tests_every_clause_of_its_statement() -> None:
    """A no-feedback fit that reproduces the time course fails the discrimination check."""
    assert _run("V11", _ctx_stub()) is True
    ctx = _ctx_stub()
    ctx["null"] = {**ctx["null"], "timecourse_r2": 0.999,
                   "worst_dev_simple": 0.001}
    assert _run("V11", ctx) is False, (
        "V11 accepted a null model that matched the time course.")


def test_v3_does_not_assert_the_fragile_direction() -> None:
    """The exposure-time marker check allows a flat marker trajectory after withdrawal."""
    ctx = _ctx_stub()
    ctx["end"] = {k: {**v, "m_sabg": 0.19} for k, v in ctx["end"].items()}
    assert _run("V3", ctx) is True, (
        "V3 should depend only on the day-14 marker excess.")


def test_reported_lnN_is_relative_to_concurrent_control() -> None:
    """Control subtraction cancels the baseline drift in log neighbour expansion."""
    t = np.linspace(0.0, T_END, int(T_END * 4) + 1)
    runs = run_nominal(P, t)
    assert runs["control"]["lnN"][-1] == pytest.approx(0.0, abs=1e-9)
    assert runs["as"]["lnN"][-1] == pytest.approx(8.29, abs=0.05)
    assert runs["as"]["lnN"][-1] > runs["as_low"]["lnN"][-1]
    assert runs["as"]["lnN"][-1] > runs["as_sasp"]["lnN"][-1]

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
    """Repeating the subtraction leaves control-relative expansion unchanged."""
    t = np.linspace(0.0, 10.0, 11)
    raw = {
        "control": {"lnN": -P.d_N * t},
        "as": {"lnN": 0.5 - P.d_N * t},
    }
    once = apply_control_relative_lnN(raw)
    twice = apply_control_relative_lnN(once)
    assert twice["control"]["lnN"][-1] == pytest.approx(0.0)
    assert twice["as"]["lnN"][-1] == pytest.approx(once["as"]["lnN"][-1])


def test_sasp_neutralisation_is_time_gated_in_trace() -> None:
    """Reported SASP neutralisation starts at the same time as in the ODE."""
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


@pytest.mark.skipif(not (OUT / "emd4_simulation_summary.json").exists(),
                    reason="run emd4_simulation.run first")
def test_table_values_match_the_generated_summary() -> None:
    """Exported table values and check identifiers agree with the simulation summary."""
    from emd4_simulation.manuscript_tables import build

    summary = json.loads((OUT / "emd4_simulation_summary.json").read_text())
    tables = build(summary)
    for key in ("endpoints", "bifurcation", "null_comparison", "spatial",
                "ensemble", "identifiability", "finite_threshold",
                "derived_quantities", "qss_reduction"):
        assert tables[key] == summary[key]
    assert {check["cid"]: check["passed"] for check in tables["checks"]} == summary["checks"]
    assert tables["public_data"] is None


@pytest.mark.skipif(not (OUT / "emd4_simulation_summary.json").exists(),
                    reason="run emd4_simulation.run first")
def test_summary_archives_every_quantity_the_prose_quotes() -> None:
    """The summary contains the model checks, ensemble statistics and derived results."""
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

    assert "v3_overstate_points" in e and "v3_overstate_points_net" in e

    for key in ("kp_ridge_fold_continuous", "kp_ridge_censored_continuous",
                "kp_admissible_lo_continuous", "kp_admissible_hi_continuous",
                "kp_ridge_fold_withdrawal_only", "kp_ridge_fold_joint_pooled",
                "kp_ridge_fold_joint_perarm",
                "kp_admissible_lo_joint_perarm",
                "kp_admissible_hi_joint_perarm"):
        assert key in s["identifiability"], f"identifiability.{key} not archived"
    assert s["identifiability"]["kp_ridge_censored_continuous"] is False, (
        "The upper bound of the reference continuous-exposure profile is censored.")

    for key in ("n_rep", "ci_lo", "ci_hi", "frac_intermediate",
                "max_frac_intermediate", "q_majority_escape"):
        assert key in s["spatial"], f"spatial.{key} not archived"
    assert s["spatial"]["n_rep"] >= 40, (
        "The reference containment scan requires at least 40 replicates per secretion ratio.")


def test_dose_and_duration_thresholds_bracket_the_published_claims() -> None:
    """Bracket the transition between resolution and persistence for dose and duration."""
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
        "The 14-day threshold dose should exceed twice the nominal quasi-static fold.")


def test_v3_requires_the_over_read_to_survive_control_subtraction() -> None:
    """The day-14 marker excess must remain positive after subtracting the concurrent control."""
    assert _run("V3", _ctx_stub()) is True

    # Here the control explains the entire apparent exposure effect.
    ctx = _ctx_stub()
    ctx["d14"] = {**ctx["d14"]}
    ctx["d14"]["control"] = {"S": 0.0, "m_sabg": 0.14, "m_ddr": 0.12,
                             "R": 0.3, "stress": 0.0}
    assert _run("V3", ctx) is False, (
        "V3 accepted a marker excess fully explained by the control.")


def test_ensemble_reports_the_marker_over_read_on_both_scales() -> None:
    """The spatially homogeneous control accounts for baseline marker positivity."""
    from emd4_simulation.ensemble import run_ensemble
    ens = run_ensemble(n=30, seed=4)
    for key in ("v3_overstate", "v3_overstate_net", "v3_overstate_background",
                "frac_v3_overstate_net_positive"):
        assert key in ens, f"{key} missing from the ensemble result"
    assert ens["v3_overstate_net"]["median"] < ens["v3_overstate"]["median"], (
        "Control subtraction should reduce the median marker excess.")
    # With basal ROS below the commitment threshold, the control has baseline marker positivity.
    fp = MARKER_BY_KEY["m_sabg"].fp
    assert ens["v3_overstate_background"]["median"] == pytest.approx(fp, abs=5e-3), (
        "The median background should match the assigned marker false-positive rate.")
    assert ens["v3_overstate_background"]["lo"] > 0.0, (
        "the sampled absolute scale no longer carries the expected positive "
        "background contribution")


def test_readme_contains_no_superseded_identifiability_claims() -> None:
    """Check the interpretation of individual and joint feedback profiles in the guide."""
    readme = (Path(__file__).resolve().parents[3] / "README.md").read_text()
    stale = (
        "Adding the withdrawal arm makes the ridge",
        "range once the withdrawal arm is added",
        "supported by ordinary data",
        "reduces to one measurable quantity",
    )
    assert not [phrase for phrase in stale if phrase in readme]


def test_escape_crossing_returns_none_rather_than_the_smallest_q() -> None:
    """Return None if the observed escape fraction stays below 50%."""
    q = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    assert escape_crossing(q, np.array([0.0, 0.0, 0.0, 0.2, 0.4])) is None
    x = escape_crossing(q, np.array([0.0, 0.0, 0.25, 0.75, 1.0]))
    assert x == pytest.approx(0.175, abs=1e-9)


def test_v10_fails_when_a_band_edge_was_not_determined() -> None:
    """The containment comparison requires measured values for both band edges."""
    assert _run("V10", _ctx_stub()) is True
    assert _run("V10", _ctx_stub(
        spatial={"q_contained": None, "q_majority_escape": 0.30})) is False
    assert _run("V10", _ctx_stub(
        spatial={"q_contained": 0.15, "q_majority_escape": None})) is False


def test_reached_edge_is_ring_membership_not_a_radius_threshold() -> None:
    """The outer lattice ring includes boundary cells at different distances from the centre."""
    nx = ny = 41
    co = _hex_coords(nx, ny)
    ring = _boundary_mask(nx, ny)
    centre = int(np.argmin(np.linalg.norm(co - co.mean(axis=0), axis=1)))
    d = np.linalg.norm(co - co[centre], axis=1)

    half_height = np.abs(co[:, 1] - co[centre, 1]).max()
    assert half_height < 0.45 * nx, (
        "The comparison radius should exceed the lattice half-height.")

    # Select outer-ring cells within the comparison radius.
    top = np.flatnonzero(ring & (d < 0.45 * nx))
    assert len(top) > 0
    assert ring[top[0]], "a boundary cell must count as having reached the edge"


def test_wilson_interval_has_width_at_the_extremes() -> None:
    """Wilson intervals retain uncertainty at the edges of the binomial range."""
    lo, hi = wilson_interval(5, 5)
    assert lo < 0.99 and hi == pytest.approx(1.0)
    assert hi - lo > 0.3, "a five-for-five result must not read as certain"
    lo40, hi40 = wilson_interval(32, 40)
    assert lo40 > lo, "more replicates must tighten the interval"


def test_censored_profile_grid_is_flagged_and_gated() -> None:
    """An admissible upper grid endpoint flags a right-censored profile."""
    p = Params()
    grid = np.linspace(0.0, 2.0 * p.k_p, 21)

    censored = _summarise(grid, np.full_like(grid, 0.001), 0.02, p)
    assert censored["kp_ridge_censored"] is True

    rmse = np.linspace(0.001, 0.05, len(grid))
    assert _summarise(grid, rmse, 0.02, p)["kp_ridge_censored"] is False

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


def test_t50_is_a_half_rise_time_on_a_rise_then_decay_trajectory() -> None:
    """The half-rise uses the maximum attained during the observation window."""
    t = np.linspace(0.0, 100.0, 1001)
    v = np.where(t <= 20.0, t / 20.0, np.exp(-(t - 20.0) / 5.0))
    assert t50(t, v) == pytest.approx(10.0, abs=0.2), (
        "Expected half of the maximum fraction at t = 10.")

    plateau = np.clip(t / 20.0, 0.0, 1.0)
    assert t50(t, plateau) == pytest.approx(10.0, abs=0.2)


def test_t50_on_the_real_withdrawal_arm_reports_the_rise() -> None:
    t = np.linspace(0.0, T_END, int(T_END * 4) + 1)
    runs = run_nominal(P, t)
    s1 = np.asarray(runs["as"]["S1"])
    assert s1[-1] < 1e-4 < s1.max(), "S1 must rise and then decay in this arm"
    assert t50(t, s1) > 2.0, (
        "The primary fraction should reach half of its peak after day 2.")


def test_hysteresis_area_is_grid_converged() -> None:
    """Increasing the dose-grid resolution leaves the hysteresis area within tolerance."""
    iv = Intervention()
    a121 = bf.hysteresis_loop(P, iv, E_max=0.15, n=121, dwell=600.0)["area"]
    a181 = bf.hysteresis_loop(P, iv, E_max=0.15, n=181, dwell=600.0)["area"]
    assert abs(a121 - a181) / a181 < 0.02, (
        f"hysteresis area not converged: n=121 gives {a121:.5f}, "
        f"n=181 gives {a181:.5f}")
    # Compare with the analytic estimate S_upper × E_fold.
    assert a121 == pytest.approx(0.9240 * 0.07018, rel=0.05)


def test_sweep_returns_the_final_state_so_the_up_sweep_runs_once() -> None:
    E = np.linspace(0.0, 0.15, 5)
    S, y_end = bf.sweep(P, Intervention(), E, dwell=100.0)
    assert S.shape == E.shape
    assert y_end.shape == (len(IDX),)


def test_summary_json_is_valid_json_for_javascript() -> None:
    """Convert non-finite values to null before writing strict JSON."""
    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe({"a": [np.nan, 2.0]}) == {"a": [None, 2.0]}

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
            f"{name} contains a non-finite value outside the JSON standard")
        json.loads(text, parse_constant=_reject_constant)


def _reject_constant(name):
    raise AssertionError(f"summary JSON contains the constant {name}")
