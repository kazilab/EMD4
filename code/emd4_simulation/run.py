"""Run the EMD4 simulation: trajectories, bifurcation, nulls, validation.

    python -m emd4_simulation.run [--no-identifiability] [--no-figure]

Exit status is non-zero if any validation check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
    """First time a trajectory reaches half its own PEAK value.

    Used instead of peak time for every lag comparison: on a plateau the argmax
    is set by numerical noise, whereas the half-rise time is stable and is what
    a delay actually means.

    Half of the PEAK, not half of the final value. On a plateau the two agree,
    which is the case the original definition was written for. On a trajectory
    that rises and then decays they do not, and the final-value form stops being
    a rise time at all: in the withdrawal arm S1 peaks at 0.366 on day 14.5 and
    decays to a remnant of 1.4e-5, so "half the final value" is 6.8e-6 and is
    crossed on day 0.25 -- reported as an onset 21 days EARLIER than the true
    half-rise at day 5.25. Every lag comparison in the battery (V4, V5) is a
    comparison of these numbers, so the statistic has to mean what its name says
    on a decaying arm as well as on a plateau.
    """
    peak = float(np.max(v))
    if abs(peak) < 1e-6:
        return np.inf
    idx = int(np.argmax(v >= 0.5 * peak))
    return float(t[idx]) if v[idx] >= 0.5 * peak else np.inf


def json_safe(obj):
    """Recursively replace non-finite floats with None.

    ``json.dumps`` emits bare ``NaN`` and ``Infinity`` by default. Python reads
    those back, but they are not valid JSON and ``JSON.parse`` rejects the whole
    file -- which means one NaN anywhere in the summary silently breaks every
    docx generator that reads it. NaN is a real answer here (an occupied
    fraction is undefined when no lesion survived), so it is written as null
    rather than suppressed.
    """
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

    # --- nulls, fitted to what an experimenter would have -----------------
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

    # --- bifurcation structure -------------------------------------------
    fold = bf.find_fold(p, Intervention())
    # n = 121: the integrand jumps at the fold, and n = 16 put the area 5% high.
    hyst = bf.hysteresis_loop(p, Intervention(), E_max=0.15, n=121, dwell=600.0)
    q_crit = bf.containment_threshold(p, Intervention())
    eq0 = bf.equilibria(p, Intervention(), 0.0)
    Es_rt, taus = bf.recovery_times(p, Intervention(),
                                    bf.approach_grid(fold or 0.07, 14))
    bif = {"fold": fold, "width": hyst["width"], "q_crit": q_crit,
           "eq0": eq0, "hyst": hyst,
           "rt_E": Es_rt, "rt_tau": taus,
           "rt_ratio": float(taus[-1] / taus[0]) if len(taus) > 1 else 1.0}
    # The (E*-E)^(-1/2) law holds only in the asymptotic regime. Quoting a
    # single ratio across the whole sweep implies a scaling that is not there,
    # so both slopes are recorded and the README states both.
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

    # --- the threshold a finite experiment actually sees --------------------
    # Tipping is a dose x DURATION condition, not a static dose threshold. Both
    # halves are computed rather than asserted, because the Results quote them.
    from .ensemble import threshold_dose, threshold_duration
    finite = {
        "threshold_dose_at_t_off": threshold_dose(p, Intervention(), t_off=T_OFF),
        "threshold_duration_at_reference": threshold_duration(
            p, Intervention(), E=DOSE),
        "t_off": T_OFF,
        "reference_dose": DOSE,
    }

    # --- identifiability --------------------------------------------------
    if do_ident:
        ident = idf.profile_kp(p)
        ident_w = idf.profile_kp_withdrawal_only(p)
        # The actual added-information test: both arms in one objective.
        ident_j = idf.profile_kp_joint(p)
    else:                                   # keep the battery runnable fast
        _stub = {"kp_ridge_width": np.nan, "kp_ridge_fold": np.nan,
                 "null_admissible": None, "kp_ridge_censored": None,
                 "admissible": np.array([]), "skipped": True}
        ident = dict(_stub)
        ident_w = dict(_stub)
        ident_j = dict(_stub, kp_ridge_fold_perarm=np.nan,
                       kp_ridge_width_perarm=np.nan,
                       kp_ridge_censored_perarm=None,
                       admissible_perarm=np.array([]))

    # --- tier 3: spatial ---------------------------------------------------
    # n_rep = 40, not 5. At five replicates the escape fraction is quantised to
    # steps of 0.2 and "escape is certain" means five coin flips landing the
    # same way: q_sec = 0.50 scored 5/5 and was reported as certain escape,
    # where forty replicates put it at 0.80 [0.68, 0.92]. Both the band edge
    # V10 tests and the 50%-crossing the Results quote were set by that noise.
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
        # Band edges are None when undetermined rather than 0.0/1.0. The old
        # fallbacks turned V10 into `0 <= q_crit <= 1`, which nothing can fail.
        contained = sc["q"][sc["escaped"] == 0.0]
        majority = sc["q"][sc["escaped"] > 0.5]
        spatial = {
            "scan": sc,
            "q_contained": float(contained.max()) if len(contained) else None,
            "q_majority_escape": float(majority.min()) if len(majority) else None,
            # Interpolated, and None when the scan never crosses 50%.
            "q_half": escape_crossing(sc["q"], sc["escaped"], 0.5),
            # Diagnostic for the claim this tier is allowed to make at all: if
            # bounded lesions are never seen, the lattice measures extinction
            # probability and not propagation extent.
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
        except Exception as exc:            # a malformed check must not
            passed = False                  # masquerade as a failed model
            print(f"  !! {chk.cid} raised: {exc}")
        results.append((chk, passed))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-identifiability", action="store_true",
                    help="skip the k_p profile (slow); V12 will not run")
    ap.add_argument("--n-ensemble", type=int, default=250,
                    help="uncertainty draws; 0 disables")
    ap.add_argument("--no-spatial", action="store_true",
                    help="skip the tier-3 spatial scan")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--outdir", default="figures/output")
    args = ap.parse_args()

    p = Params()
    t = np.linspace(0.0, T_END, N_T)

    bar = "=" * 78
    print(bar)
    print("EMD4 mechanistic simulation   KCC5 -> EMD4/KCC6 -> KCC7, KCC10")
    print("                              KCC9 opposing polarity")
    print("public-data-constrained proof of concept; arsenite as exemplar exposure")
    print(bar)

    from .model import escape_residual_check, quasi_steady_state_check
    ok, worst = quasi_steady_state_check(p)
    ok_esc, worst_esc = escape_residual_check(p)
    print(f"\nQSS reduction to the manuscript's equation "
          f"(full rhs evaluated on the slow manifold):")
    print(f"  with k_esc -> 0, matches the manuscript form:   "
          f"{'OK' if ok else 'FAILED'} (worst |diff| = {worst:.2e})")
    print(f"  with k_esc on, residual is exactly -k_esc*S:    "
          f"{'OK' if ok_esc else 'FAILED'} (worst |diff| = {worst_esc:.2e})")

    runs = run_nominal(p, t)
    ctx = build_context(p, runs, t,
                        do_ident=not args.no_identifiability,
                        do_spatial=not args.no_spatial)

    # --- endpoint table ---------------------------------------------------
    print(f"\n{'-' * 78}\nLatent state and observed markers at day {T_END:.0f}\n{'-' * 78}")
    hdr = f"  {'condition':<34}" + "".join(f"{o:>9}" for o in
                                           ["S", "S1", "S2", "P", "X"])
    print(hdr)
    for c in CONDITIONS:
        e = ctx["end"][c.key]
        row = "".join(f"{e[o]:>9.3f}" for o in ["S", "S1", "S2", "P", "X"])
        print(f"  {c.label:<34}{row}")

    print(f"\n  {'condition':<34}" + "".join(f"{m.label[:8]:>10}" for m in MARKERS))
    for c in CONDITIONS:
        e = ctx["end"][c.key]
        row = "".join(f"{e[m.key]:>10.3f}" for m in MARKERS)
        print(f"  {c.label:<34}{row}")

    # --- equilibrium structure --------------------------------------------
    b = ctx["bif"]
    print(f"\n{'-' * 78}\nEquilibrium structure\n{'-' * 78}")
    print("  equilibria at E = 0 (unexposed):")
    for e in b["eq0"]:
        kind = "stable  " if e.stable else "UNSTABLE"
        rt = f"{e.recovery_time:8.1f} d" if np.isfinite(e.recovery_time) else "       -"
        print(f"    S = {e.S:.4f}   {kind}   lambda = {e.lam_max:+.4f}   "
              f"recovery {rt}")
    print(f"  up-sweep fold:            E* = {b['fold']:.4f}"
          if b["fold"] is not None else "  up-sweep fold:            none (monostable)")
    print(f"  hysteresis loop AREA:     {b['width']:.4f} "
          f"(units of S x dose, not a width in E)")
    print(f"  containment threshold:    q_sec* = {b['q_crit']:.4f} "
          f"(nominal q_sec = {p.q_sec})" if b["q_crit"] is not None
          else "  containment threshold:    none found")
    print(f"  critical slowing:         recovery time x{b['rt_ratio']:.1f} "
          f"approaching the fold")
    if "normal_form" in b and b["normal_form"].get("ok"):
        print(f"  normal form:              gap^2 linear in (E*-E), "
              f"r2 = {b['normal_form']['r2']:.4f}")
    fin = ctx["finite"]
    print(f"  finite-experiment threshold (tipping is dose x DURATION):")
    print(f"    at E = {fin['reference_dose']:.2f}, exposure must exceed "
          f"{fin['threshold_duration_at_reference']:.2f} d to persist")
    print(f"    at a {fin['t_off']:.0f}-day exposure, dose must exceed "
          f"E = {fin['threshold_dose_at_t_off']:.3f}")
    print(f"    -- neither equals the quasi-static fold E* = {b['fold']:.4f}")

    # --- model comparison --------------------------------------------------
    n = ctx["null"]
    print(f"\n{'-' * 78}\nModel comparison: can a no-feedback model do this?\n{'-' * 78}")
    print(f"  forward ENDPOINT dose-response (day 60):")
    print(f"    simple null                       R2 = {n['endpoint_r2']:.5f}   "
          f"<- indistinguishable")
    print(f"  full continuous-exposure TIME COURSE (12 doses x 121 points):")
    print(f"    simple null   (4 par)             R2 = {n['timecourse_r2']:.5f}   "
          f"worst dev {n['worst_dev_simple']:.3f}")
    print(f"    cascade null  (5 par, sigmoidal)  R2 = {n['cascade_r2']:.5f}   "
          f"worst dev {n['worst_dev_cascade']:.3f}")
    print(f"  WITHDRAWAL at day {T_OFF:.0f}, senescent fraction at day {T_END:.0f}:")
    print(f"    mechanistic                       S = {ctx['end']['as']['S']:.3f}")
    print(f"    simple null                       S = {n['simple_end']:.3f}")
    print(f"    cascade null                      S = {n['cascade_end']:.3f}")

    # --- identifiability ---------------------------------------------------
    if not args.no_identifiability:
        i1, i2, i3 = ctx["ident"], ctx["ident_w"], ctx["ident_j"]
        print(f"\n{'-' * 78}\nIdentifiability of the paracrine strength k_p\n{'-' * 78}")
        print("  Each row is a DESIGN, not a nested amount of data. The first two")
        print("  are separate experiments; only 'both arms jointly' adds the")
        print("  withdrawal arm to the continuous-exposure record.")
        for name, d in (("continuous exposure, one dose", i1),
                        ("withdrawal design only", i2),
                        ("both arms jointly (pooled RMSE)", i3)):
            adm = d["admissible"]
            rng = (f"{adm.min():.3f} - {adm.max():.3f}" if len(adm) else "empty")
            flag = "  [CENSORED BY GRID]" if d.get("kp_ridge_censored") else ""
            print(f"  {name:<32} admissible k_p {rng:<16} "
                  f"{d['kp_ridge_fold']:>5.0f}-fold   "
                  f"k_p=0 admissible: {d['null_admissible']}{flag}")
        adm_pa = i3.get("admissible_perarm", np.array([]))
        if len(adm_pa):
            print(f"  {'both arms, EACH within noise':<32} admissible k_p "
                  f"{adm_pa.min():.3f} - {adm_pa.max():.3f}      "
                  f"{i3['kp_ridge_fold_perarm']:>5.0f}-fold   "
                  f"k_p=0 admissible: {i3['null_admissible_perarm']}")
        print("  NOTE: the pooled row uses an AVERAGE RMSE threshold, so a longer,")
        print("  easier record lowers the average and can admit a k_p that fits the")
        print("  continuous arm badly. The per-arm row is the count-invariant one,")
        print("  and on it the withdrawal arm neither narrows nor widens the ridge.")
        print("  (the profile grid is log-spaced to "
              f"{i1['kp_grid_top'] / Params().k_p:.0f}x nominal; the previous "
              "linear grid stopped at 2x,")
        print("   where the fit was still inside the noise floor, so both "
              "ranges were grid artefacts)")

    # --- validation --------------------------------------------------------
    print(f"\n{bar}\nValidation battery\n{bar}")
    results = validate(ctx)
    for chk, passed in results:
        print(f"\n[{'PASS' if passed else 'FAIL'}] {chk.cid}  {chk.edge}")
        print(f"       {chk.statement}")
        print(f"       basis: {chk.source}")
    n_pass = sum(1 for _, ok_ in results if ok_)
    print(f"\n  {n_pass}/{len(results)} checks passed")

    # --- tier 3 --------------------------------------------------------------
    if ctx["spatial"] is not None:
        sp = ctx["spatial"]
        print(f"\n{'-' * 78}\nTier 3: spatial stochastic propagation "
              f"(41x41 hex lattice, {sp['scan']['n_rep']} reps)\n{'-' * 78}")
        print("    q_sec   escaping   95% CI (Wilson)   bounded lesions")
        for q, e, lo, hi, fi in zip(sp["scan"]["q"], sp["scan"]["escaped"],
                                    sp["scan"]["ci_lo"], sp["scan"]["ci_hi"],
                                    sp["scan"]["frac_intermediate"]):
            print(f"    {q:5.2f}     {e * 100:5.0f}%     "
                  f"[{lo * 100:4.0f}%, {hi * 100:4.0f}%]         {fi * 100:4.0f}%")
        qc = sp["q_contained"]
        qm = sp["q_majority_escape"]
        print(f"\n    every replicate contained up to q_sec = "
              f"{qc:.2f}" if qc is not None else
              "\n    no q_sec contained every replicate")
        print(f"    majority escape from q_sec = {qm:.2f}" if qm is not None
              else "    no q_sec reached majority escape")
        print(f"    mean-field threshold q_sec* = {b['q_crit']:.3f}")
        if sp["q_half"] is not None:
            gap = sp["q_half"] - b["q_crit"]
            print(f"    50% escape at q_sec = {sp['q_half']:.3f}, i.e. "
                  f"{gap:+.3f} from the mean-field threshold")
            print(f"    -- at {sp['scan']['n_rep']} replicates the escape "
                  f"probabilities either side of the crossing carry Wilson "
                  f"intervals\n       ~0.15 wide, so this gap is NOT resolved: "
                  f"the scan does not establish\n       that spatial structure "
                  f"makes propagation harder than well-mixed.")
        print(f"    bounded (non-extinct, non-saturating) lesions seen in at "
              f"most {sp['max_frac_intermediate'] * 100:.0f}% of replicates:")
        print(f"       outcomes are near-binary, so this lattice measures "
              f"EXTINCTION PROBABILITY,\n       not propagation extent -- cluster "
              f"size and radius are the saturation\n       value times the escape "
              f"probability and carry no independent information.")

    # --- uncertainty --------------------------------------------------------
    ens = None
    if args.n_ensemble > 0:
        from .ensemble import run_ensemble
        print(f"\n{'-' * 78}\nUncertainty propagation ({args.n_ensemble} draws over free "
              f"effect sizes)\n{'-' * 78}")
        ens = run_ensemble(n=args.n_ensemble)
        print(f"  {ens['n_ok']}/{ens['n_requested']} draws integrated successfully\n")
        print("  Does the tipping point survive the spread?")
        print("  THREE different questions -- do not quote one as another:")
        print(f"    high burden still high at day 600      "
              f"{ens['frac_high_state_retained'] * 100:5.1f}%   "
              f"(retention, NOT bistability)")
        print(f"    TWO STABLE EQUILIBRIA at E = 0         "
              f"{ens['frac_two_stable_equilibria'] * 100:5.1f}%   "
              f"<- ordinary bistability")
        print(f"    empty state AND persistent state       "
              f"{ens['frac_empty_plus_persistent'] * 100:5.1f}%   "
              f"<- strictest reading")
        print(f"    senescence-free state LOST             "
              f"{ens['frac_empty_state_lost'] * 100:5.1f}%   "
              f"(basal ROS alone commits cells)")
        print(f"      ...yet still bistable anyway         "
              f"{ens['frac_bistable_without_empty_state'] * 100:5.1f}%   "
              f"(so losing it does NOT imply one attractor)")
        tb, qb = ens["threshold_dose"], ens["q_sec_critical"]
        print(f"    threshold dose (14 d)   median {tb['median']:.3f}  "
              f"[{tb['lo']:.3f}, {tb['hi']:.3f}]")
        print(f"    critical q_sec*         median {qb['median']:.3f}  "
              f"[{qb['lo']:.3f}, {qb['hi']:.3f}]   "
              f"(over {ens['n_with_q_sec_critical']}/{ens['n_ok']} draws with a crossing)")
        print(f"      of those, bistable WINDOW not half-line "
              f"{ens['n_q_sec_reentrant']:>3d}   "
              f"(bistability reappears then vanishes as q_sec rises)")
        print("\n  Sign robustness of the headline claims")
        ov, ovn = ens["v3_overstate"], ens["v3_overstate_net"]
        bg = ens["v3_overstate_background"]
        rows = [
            ("reference dose persists after withdrawal", ens["frac_persist_at_reference"]),
            ("  ...but its own control is persistent too", ens["frac_control_persistent"]),
            ("  EXPOSURE-ATTRIBUTABLE (paired contrast)", ens["frac_persist_exposure_attributable"]),
            ("low dose stays below persistence (S<0.30)", ens["frac_lowdose_resolves"]),
            ("  low dose near baseline (S<0.05)", ens["frac_lowdose_near_baseline"]),
            ("SA-beta-gal over-read, ABSOLUTE scale", None),
            ("  of which unexposed assay background", None),
            ("  SA-beta-gal over-read, NET of control", ens["frac_v3_overstate_net_positive"]),
            ("  marker falls while latent state persists", ens["frac_v3_understate"]),
            ("SASP neutralisation collapses the state", ens["frac_v7_sasp_collapses"]),
            ("senolytic leaves >0.20 standing", ens["frac_v7_seno_survives"]),
        ]
        for label, frac in rows:
            if frac is None:
                d = bg if "background" in label else ov
                note = ("not a claim about exposure" if "background" in label
                        else "NOT a robustness result -- see below")
                print(f"    {label:<42} {d['median'] * 100:4.1f} pts "
                      f"[{d['lo'] * 100:.1f}, {d['hi'] * 100:.1f}]  {note}")
            else:
                verdict = "robust" if frac >= 0.90 else ("fragile" if frac >= 0.60 else "NOT robust")
                extra = ""
                if "NET of control" in label:
                    extra = (f"  {ovn['median'] * 100:+.1f} pts "
                             f"[{ovn['lo'] * 100:+.1f}, {ovn['hi'] * 100:+.1f}]")
                print(f"    {label:<42} {frac * 100:5.1f}%   {verdict}{extra}")
        print("\n    The sampled ABSOLUTE over-read includes the marker's own "
              "unexposed false-positive\n    rate (m_sabg.fp = 0.08), a structural "
              "constant the ensemble does not sample.\n    It is not a universal "
              "positive lower bound: at zero stress and marker steady\n    state, "
              "the nominal y - S becomes negative above S = 0.615. The NET "
              "figure is\n    the exposure-attributable part and is the one the "
              "manuscript claim rests on.")

    # --- summary -----------------------------------------------------------
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
            # NB "with_withdrawal" is a WITHDRAWAL-ONLY design, not the
            # continuous record plus the withdrawal arm. The joint keys below
            # are the added-data comparison.
            "kp_ridge_fold_withdrawal_only": ctx["ident_w"].get("kp_ridge_fold"),
            "kp_ridge_width_withdrawal_only": ctx["ident_w"]["kp_ridge_width"],
            "kp_ridge_fold_joint_pooled": ctx["ident_j"].get("kp_ridge_fold"),
            "kp_ridge_width_joint_pooled": ctx["ident_j"].get("kp_ridge_width"),
            # Count-invariant criterion: every arm within the noise floor.
            # This is the one the manuscript should quote.
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
            # True would mean the admissible set ran off the top of the profile
            # grid, making the width a lower bound rather than an interval.
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
            # Absolute: includes the marker's unexposed false-positive rate,
            # which the concurrent control carries at the same instant.
            "sabg_overstatement_d14_low_dose":
                ctx["d14"]["as_low"]["m_sabg"] - ctx["d14"]["as_low"]["S"],
            # Net of that control: the exposure-attributable part, and the only
            # one of the two that is a statement about the exposure.
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
            # ESCAPE = boundary reached at ANY time during the run.
            "n_escaped": sp["scan"]["n_escaped"].tolist(),
            "frac_escaping": sp["scan"]["escaped"].tolist(),
            # Edge OCCUPANCY at t_end, the older snapshot criterion. Reported
            # so the two definitions can be compared; on the nominal 41x41,
            # 40-replicate scan they agree at every q, because a front that
            # reaches the boundary does not subsequently clear off it.
            "n_occupied_at_end": sp["scan"]["n_occupied_at_end"].tolist(),
            "frac_occupied_at_end": sp["scan"]["occupied_at_end"].tolist(),
            # Containment has TWO mechanisms and they are not the same claim.
            "frac_extinct": sp["scan"]["frac_extinct"].tolist(),
            "ci_lo": sp["scan"]["ci_lo"].tolist(),
            "ci_hi": sp["scan"]["ci_hi"].tolist(),
            "frac_intermediate": sp["scan"]["frac_intermediate"].tolist(),
            "max_frac_intermediate": sp["max_frac_intermediate"],
            # Diagnostics only. They are the saturation value times the escape
            # probability (r = 1.000 against frac_escaping), so they are NOT
            # independent measurements and must not be tabulated as such.
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
            # Bistability is NOT monotone in q_sec. In these draws it appears
            # and then disappears again as secondary secretion rises, so the
            # threshold is the lower edge of a WINDOW, not of a half-line.
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
    print(f"\n  wrote {path}")

    if not args.no_figure:
        from .figure import make_figure
        ctx["ensemble"] = summary.get("ensemble")
        make_figure(runs, t, ctx, outdir)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
