"""Run the EMD4 simulation and its model checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import FIGURE_FORMATS, FIGURE_STEM, remove_previous_figure
from .conditions import CHECKS, CONDITIONS, DOSE, PERSIST, T_END, T_OFF
from . import bifurcation as bf
from . import identifiability as idf
from . import nulls as nl
from .model import (MARKERS, Intervention, Params, apply_control_relative_lnN,
                    simulate, trace)

N_T = int(T_END * 4) + 1
LATENT = ["R", "S", "S1", "S2", "P", "X", "lnN", "secondary_fraction"]
OBSERVED = [m.key for m in MARKERS]


def t50(t: np.ndarray, v: np.ndarray) -> float:
    """First time the trajectory reaches half its peak, including trajectories that decay."""
    peak = float(np.max(v))
    if abs(peak) < 1e-6:
        return np.inf
    idx = int(np.argmax(v >= 0.5 * peak))
    return float(t[idx]) if v[idx] >= 0.5 * peak else np.inf


def json_safe(obj):
    """Represent missing or non-finite values as null in JSON outputs."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_nominal(p: Params, t: np.ndarray) -> dict:
    out = {}
    for c in CONDITIONS:
        y = simulate(p, c.iv, c.E, t, t_off=c.t_off)
        out[c.key] = trace(y, p, c.iv, t)
    # KCC10 is reported as ΔlnN vs concurrent control (cancels basal -d_N).
    return apply_control_relative_lnN(out)


def build_context(p: Params, runs: dict, t: np.ndarray,
                  do_ident: bool, do_spatial: bool = True) -> dict:
    i14 = int(np.searchsorted(t, T_OFF))
    i5 = int(np.searchsorted(t, 5.0))
    end = {k: {o: float(v[-1]) for o, v in tr.items()} for k, tr in runs.items()}
    d14 = {k: {o: float(v[i14]) for o, v in tr.items()} for k, tr in runs.items()}
    d5 = {k: {o: float(v[i5]) for o, v in tr.items()} for k, tr in runs.items()}
    t50s = {k: {o: t50(t, v) for o, v in tr.items()} for k, tr in runs.items()}

    doses = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10,
                      0.12, 0.15, 0.20, 0.25, 0.5, 1.0])
    t_fit = np.linspace(0.0, 60.0, 121)
    tc = nl.forward_timecourse(p, Intervention(), doses, t_fit)

    q_end, r2_end = nl.fit_null(doses, tc[:, -1], t_end=60.0)
    q_tc, r2_tc = nl.fit_null_timecourse(doses, tc, t_fit)
    q_cas, r2_cas = nl.fit_cascade(doses, tc, t_fit)

    w_simple = nl.compare_withdrawal(p, Intervention(), q_tc, 1.0, T_OFF, T_END)
    w_cascade = nl.cascade_withdrawal(p, Intervention(), q_cas, 1.0, T_OFF, T_END)

    null = {
        "endpoint_r2": r2_end, "timecourse_r2": r2_tc, "cascade_r2": r2_cas,
        "endpoint_fit": q_end, "timecourse_fit": q_tc, "cascade_fit": q_cas,
        "simple_end": w_simple["null_end"], "cascade_end": w_cascade["null_end"],
        "worst_dev_simple": float(max(
            abs(nl.simulate_null(q_tc, float(E), t_fit) - tc[i]).max()
            for i, E in enumerate(doses))),
        "worst_dev_cascade": float(max(
            abs(nl.simulate_cascade(q_cas, float(E), t_fit) - tc[i]).max()
            for i, E in enumerate(doses))),
    }

    fold = bf.find_fold(p, Intervention())
    hyst = bf.hysteresis_loop(p, Intervention(), E_max=0.15, n=121, dwell=600.0)
    q_crit = bf.containment_threshold(p, Intervention())
    eq0 = bf.equilibria(p, Intervention(), 0.0)
    Es_rt, taus = bf.recovery_times(p, Intervention(),
                                    bf.approach_grid(fold or 0.07, 14))
    bif = {"fold": fold, "width": hyst["width"], "q_crit": q_crit,
           "eq0": eq0, "hyst": hyst,
           "rt_E": Es_rt, "rt_tau": taus,
           "rt_ratio": float(taus[-1] / taus[0]) if len(taus) > 1 else 1.0}
    # The inverse-square-root law applies close to the fold.
    if fold is not None and len(taus) > 2:
        d_ = np.asarray(fold) - np.asarray(Es_rt)
        tau_ = np.asarray(taus)
        m_ = (d_ > 0) & (d_ < 1e-2)
        bif["rt_slope_full"] = float(np.polyfit(np.log(d_[d_ > 0]),
                                                np.log(tau_[d_ > 0]), 1)[0])
        bif["rt_slope_asymptotic"] = (float(np.polyfit(np.log(d_[m_]),
                                                       np.log(tau_[m_]), 1)[0])
                                      if m_.sum() > 2 else None)
    if fold is not None:
        bif["normal_form"] = bf.normal_form(p, Intervention(), fold,
                                            span=0.05, n=9)

    from .ensemble import threshold_dose, threshold_duration
    finite = {
        "threshold_dose_at_t_off": threshold_dose(p, Intervention(), t_off=T_OFF),
        "threshold_duration_at_reference": threshold_duration(
            p, Intervention(), E=DOSE),
        "t_off": T_OFF,
        "reference_dose": DOSE,
    }

    if do_ident:
        ident = idf.profile_kp(p)
        ident_w = idf.profile_kp_withdrawal_only(p)
        ident_j = idf.profile_kp_joint(p)
    else:
        _stub = {"kp_ridge_width": np.nan, "kp_ridge_fold": np.nan,
                 "null_admissible": None, "kp_ridge_censored": None,
                 "admissible": np.array([]), "skipped": True}
        ident = dict(_stub)
        ident_w = dict(_stub)
        ident_j = dict(_stub, kp_ridge_fold_perarm=np.nan,
                       kp_ridge_width_perarm=np.nan,
                       kp_ridge_censored_perarm=None,
                       admissible_perarm=np.array([]))

    spatial = None
    if do_spatial:
        from .spatial import (SpatialParams, containment_scan, escape_crossing,
                              wilson_interval)
        qs = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65])
        sc = containment_scan(qs, n_rep=40, t_end=300.0,
                              base=SpatialParams(nx=41, ny=41))
        ci = [wilson_interval(int(k), sc["n_rep"]) for k in sc["n_escaped"]]
        sc["ci_lo"] = np.array([a for a, _ in ci])
        sc["ci_hi"] = np.array([b for _, b in ci])
        contained = sc["q"][sc["escaped"] == 0.0]
        majority = sc["q"][sc["escaped"] > 0.5]
        spatial = {
            "scan": sc,
            "q_contained": float(contained.max()) if len(contained) else None,
            "q_majority_escape": float(majority.min()) if len(majority) else None,
            # Interpolated, and None when the scan never crosses 50%.
            "q_half": escape_crossing(sc["q"], sc["escaped"], 0.5),
            "max_frac_intermediate": float(sc["frac_intermediate"].max()),
        }

    return {"end": end, "d14": d14, "d5": d5, "t50": t50s, "peak_t": t50s,
            "spatial": spatial, "finite": finite,
            "null": null, "bif": bif, "ident": ident, "ident_w": ident_w, "ident_j": ident_j}


def validate(ctx: dict) -> list[tuple]:
    results = []
    for chk in CHECKS:
        try:
            passed = bool(chk.test(ctx))
        except Exception as exc:
            passed = False
            print(f"{chk.cid} raised {exc}")
        results.append((chk, passed))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-identifiability", action="store_true",
                    help="skip the k_p profile required by check V12")
    ap.add_argument("--n-ensemble", type=int, default=250,
                    help="number of uncertainty draws (0 to skip)")
    ap.add_argument("--no-spatial", action="store_true",
                    help="skip the spatial scan required by check V10")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "reproduced"))
    args = ap.parse_args()

    p = Params()
    t = np.linspace(0.0, T_END, N_T)

    print("Simulating senescence during and after exposure", flush=True)
    from .model import escape_residual_check, quasi_steady_state_check
    ok, worst = quasi_steady_state_check(p)
    ok_esc, worst_esc = escape_residual_check(p)
    if not (ok and ok_esc):
        print(f"Quasi-steady-state reduction failed (residuals {worst:.2e}, {worst_esc:.2e})")

    runs = run_nominal(p, t)
    ctx = build_context(p, runs, t,
                        do_ident=not args.no_identifiability,
                        do_spatial=not args.no_spatial)
    b = ctx["bif"]
    n = ctx["null"]
    results = validate(ctx)
    for chk, passed in results:
        if not passed:
            print(f"Failed {chk.cid}: {chk.statement}")
    n_pass = sum(1 for _, passed in results if passed)
    print(f"{n_pass}/{len(results)} model checks passed", flush=True)

    ens = None
    if args.n_ensemble > 0:
        from .ensemble import run_ensemble
        print(f"Running {args.n_ensemble} parameter draws", flush=True)
        ens = run_ensemble(n=args.n_ensemble)
        print(f"Completed {ens['n_ok']}/{ens['n_requested']} parameter draws")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "endpoints": {k: {o: v for o, v in d.items()} for k, d in ctx["end"].items()},
        "bifurcation": {
            "fold": b["fold"], "hysteresis_width": b["width"],
            "q_sec_critical": b["q_crit"], "recovery_time_ratio": b["rt_ratio"],
            "equilibria_at_zero": [
                {"S": e.S, "stable": e.stable, "lambda": e.lam_max}
                for e in b["eq0"]],
        },
        "null_comparison": {
            "endpoint_r2": n["endpoint_r2"],
            "timecourse_r2": n["timecourse_r2"],
            "cascade_r2": n["cascade_r2"],
            "mechanistic_after_withdrawal": ctx["end"]["as"]["S"],
            "simple_null_after_withdrawal": n["simple_end"],
            "cascade_null_after_withdrawal": n["cascade_end"],
        },
        "identifiability": {
            "kp_ridge_width_continuous": ctx["ident"]["kp_ridge_width"],
            "kp_ridge_width_with_withdrawal": ctx["ident_w"]["kp_ridge_width"],
            "kp_ridge_fold_continuous": ctx["ident"].get("kp_ridge_fold"),
            # The with_withdrawal and withdrawal_only keys describe the withdrawal arm alone.
            "kp_ridge_fold_withdrawal_only": ctx["ident_w"].get("kp_ridge_fold"),
            "kp_ridge_width_withdrawal_only": ctx["ident_w"]["kp_ridge_width"],
            "kp_ridge_fold_joint_pooled": ctx["ident_j"].get("kp_ridge_fold"),
            "kp_ridge_width_joint_pooled": ctx["ident_j"].get("kp_ridge_width"),
            # The per-arm criterion requires every arm to fit within its noise floor.
            "kp_ridge_fold_joint_perarm": ctx["ident_j"].get("kp_ridge_fold_perarm"),
            "kp_ridge_width_joint_perarm": ctx["ident_j"].get("kp_ridge_width_perarm"),
            "kp_admissible_lo_joint_perarm": (
                float(ctx["ident_j"]["admissible_perarm"].min())
                if len(ctx["ident_j"].get("admissible_perarm", [])) else None),
            "kp_admissible_hi_joint_perarm": (
                float(ctx["ident_j"]["admissible_perarm"].max())
                if len(ctx["ident_j"].get("admissible_perarm", [])) else None),
            "kp_admissible_lo_continuous": (
                float(ctx["ident"]["admissible"].min())
                if len(ctx["ident"]["admissible"]) else None),
            "kp_admissible_hi_continuous": (
                float(ctx["ident"]["admissible"].max())
                if len(ctx["ident"]["admissible"]) else None),
            # A profile that reaches the grid edge gives a lower bound on width.
            "kp_ridge_censored_continuous": ctx["ident"].get("kp_ridge_censored"),
            "kp_ridge_censored_with_withdrawal":
                ctx["ident_w"].get("kp_ridge_censored"),
            "kp_ridge_censored_joint_perarm":
                ctx["ident_j"].get("kp_ridge_censored_perarm"),
            "null_admissible": ctx["ident"]["null_admissible"],
        },
        "finite_threshold": ctx["finite"],
        "derived_quantities": {
            "null_hill_coefficient": float(getattr(n["endpoint_fit"], "h",
                                                    float("nan"))),
            "normal_form_r2": (b.get("normal_form") or {}).get("r2"),
            "critical_slowing_slope_asymptotic": b.get("rt_slope_asymptotic"),
            "critical_slowing_slope_full": b.get("rt_slope_full"),
            "sabg_overstatement_d14_low_dose":
                ctx["d14"]["as_low"]["m_sabg"] - ctx["d14"]["as_low"]["S"],
            "sabg_overstatement_d14_low_dose_net_of_control":
                (ctx["d14"]["as_low"]["m_sabg"] - ctx["d14"]["as_low"]["S"])
                - (ctx["d14"]["control"]["m_sabg"] - ctx["d14"]["control"]["S"]),
            "sabg_unexposed_background_d14":
                ctx["d14"]["control"]["m_sabg"] - ctx["d14"]["control"]["S"],
            "worst_dev_simple_null": n["worst_dev_simple"],
            "worst_dev_cascade_null": n["worst_dev_cascade"],
        },
        "qss_reduction": {
            "worst_abs_diff_manuscript_form": worst,
            "worst_abs_diff_with_escape_term": worst_esc,
            "passed": bool(ok and ok_esc),
        },
        "checks": {chk.cid: bool(ok_) for chk, ok_ in results},
    }
    if ctx["spatial"] is not None:
        sp = ctx["spatial"]
        summary["spatial"] = {
            "q": sp["scan"]["q"].tolist(),
            "n_rep": sp["scan"]["n_rep"],
            # Escape records boundary contact at any time during the run.
            "n_escaped": sp["scan"]["n_escaped"].tolist(),
            "frac_escaping": sp["scan"]["escaped"].tolist(),
            "n_occupied_at_end": sp["scan"]["n_occupied_at_end"].tolist(),
            "frac_occupied_at_end": sp["scan"]["occupied_at_end"].tolist(),
            "frac_extinct": sp["scan"]["frac_extinct"].tolist(),
            "ci_lo": sp["scan"]["ci_lo"].tolist(),
            "ci_hi": sp["scan"]["ci_hi"].tolist(),
            "frac_intermediate": sp["scan"]["frac_intermediate"].tolist(),
            "max_frac_intermediate": sp["max_frac_intermediate"],
            # Cluster size and radius depend on the escape fraction.
            "cluster_size_diagnostic": sp["scan"]["size"].tolist(),
            "radius_diagnostic": sp["scan"]["radius"].tolist(),
            "occupied_fraction_given_escape":
                sp["scan"]["size_given_escape"].tolist(),
            "q_contained": sp["q_contained"],
            "q_majority_escape": sp["q_majority_escape"],
            "q_half_escape": sp["q_half"],
        }
    if ens is not None:
        summary["ensemble"] = {
            "n": ens["n_ok"],
            "frac_high_state_retained": ens["frac_high_state_retained"],
            "frac_two_stable_equilibria": ens["frac_two_stable_equilibria"],
            "frac_empty_plus_persistent": ens["frac_empty_plus_persistent"],
            "frac_empty_state_lost": ens["frac_empty_state_lost"],
            "frac_bistable_without_empty_state":
                ens["frac_bistable_without_empty_state"],
            "frac_v7_contrast": ens["frac_v7_contrast"],
            "n_v7_comparable": ens["n_v7_comparable"],
            "threshold_dose": ens["threshold_dose"],
            "q_sec_critical": ens["q_sec_critical"],
            "frac_persist_at_reference": ens["frac_persist_at_reference"],
            "frac_control_persistent": ens["frac_control_persistent"],
            "frac_persist_exposure_attributable":
                ens["frac_persist_exposure_attributable"],
            "frac_lowdose_resolves": ens["frac_lowdose_resolves"],
            "frac_lowdose_near_baseline": ens["frac_lowdose_near_baseline"],
            "n_with_threshold": ens["n_with_threshold"],
            "n_with_q_sec_critical": ens["n_with_q_sec_critical"],
            "frac_with_q_sec_critical": ens["frac_with_q_sec_critical"],
            # A reentrant window ends when increasing secondary secretion removes bistability.
            "frac_q_sec_reentrant": ens["frac_q_sec_reentrant"],
            "n_q_sec_reentrant": ens["n_q_sec_reentrant"],
            "n_q_sec_near_fold_undetermined":
                ens["n_q_sec_near_fold_undetermined"],
            "v3_overstate_points": ens["v3_overstate"],
            "v3_overstate_points_net": ens["v3_overstate_net"],
            "v3_overstate_background": ens["v3_overstate_background"],
            "frac_v3_overstate_net_positive": ens["frac_v3_overstate_net_positive"],
            "frac_v3_understate": ens["frac_v3_understate"],
            "frac_v7_sasp_collapses": ens["frac_v7_sasp_collapses"],
            "frac_v7_seno_survives": ens["frac_v7_seno_survives"],
            "endpoint_bands": ens["end"],
        }
    path = outdir / "emd4_simulation_summary.json"
    path.write_text(json.dumps(json_safe(summary), indent=2, default=float,
                               allow_nan=False))
    print(f"Simulation summary saved to {path}")

    (outdir / "emd4_manuscript_tables.json").unlink(missing_ok=True)

    if not args.no_figure:
        from .figure import make_figure
        ctx["ensemble"] = summary.get("ensemble")
        make_figure(runs, t, ctx, outdir)
    else:
        for ext in FIGURE_FORMATS:
            (outdir / f"{FIGURE_STEM}.{ext}").unlink(missing_ok=True)
        (outdir / f"{FIGURE_STEM}_legend.md").unlink(missing_ok=True)
    remove_previous_figure(outdir)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
