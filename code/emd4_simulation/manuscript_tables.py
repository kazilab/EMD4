"""Emit every number the EMD4 manuscript inserts need, as one JSON file.

    python -m emd4_simulation.manuscript_tables

Writes `figures/output/emd4_manuscript_tables.json`. The docx generators read
ONLY from this file, so main text and supplementary cannot drift apart, and
neither can drift from the simulation: rerun `emd4_simulation.run` first, then
this.

Mirrors `emd2_simulation/manuscript_tables.py`. As in the EMD3 build, parameter
units and meanings are read from the trailing comments in `model.py` rather than
restated here, so the comment IS the documentation and the two cannot drift.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path

from .conditions import (CONDITIONS, CHECKS, DOSE, DOSE_LOW, PERSIST, T_END,
                         T_IV, T_OFF)
from .model import IDX, MARKERS, Params
from .run import json_safe

SRC = Path(__file__).parent / "model.py"

# Which parameters carry the load, and what each one rests on. Everything not
# named here is reported with its source comment and marked chosen.
PARAM_BASIS = {
    "q_sec": "THE CONTAINMENT PARAMETER, and the one the interventions move. "
             "Secondary-cell secretion relative to primary. Holding every "
             "other parameter fixed, varying q_sec crosses the model's "
             "saddle-node -- but it is NOT the single biological quantity "
             "deciding bistability. The analytic threshold is q* = "
             "(K_P d_P / k_sasp) [n / (r S*^(n-1))]^(1/n) with r = k_p/(gamma "
             "+ k_esc), so it moves with receptor sensitivity K_P (doubling "
             "K_P doubles q* to 0.486), with SASP clearance, with paracrine "
             "gain and cooperativity, and with senescent-cell removal "
             "(doubling both clearance terms raises q* to 0.301). A measured "
             "secretome ratio cannot place a real system relative to the "
             "threshold while those quantities are unknown, and a scalar "
             "protein-output ratio need not equal the ratio of functional "
             "senescence-inducing activity. Nothing in the source data "
             "measures any of it.",
    "k_p": "CHOSEN. Maximum paracrine induction rate — the amplification edge "
           "the model is built to interrogate.",
    "n_P": "CHOSEN. Cooperativity of the paracrine response. Together with K_P "
           "this sets whether the response is switch-like; a value of 1 removes "
           "the bistability entirely.",
    "gam_immune": "CHOSEN. Immune-mediated clearance. NOT scored as KCC7: the "
                  "edge is direction-sensitive and this build reports both.",
    "k_esc": "CHOSEN. Escape from arrest. Drains S into X and drives nothing "
             "downstream — deliberately inert, so KCC9 cannot be scored by "
             "accident.",
    "tau_C": "CHOSEN. Arrest → SASP competence delay. The delay is what makes "
             "SASP a secretion phenotype rather than a transcript readout.",
}

STATE_GROUPS = [
    ("Redox and exposure",
     [("R", "ROS level"),
      ("stress", "continuing exposure stress term")]),
    ("Senescent compartments",
     [("S₁", "PRIMARY senescent cells — induced directly by exposure"),
      ("S₂", "SECONDARY senescent cells — induced paracrinely by SASP"),
      ("S = S₁ + S₂", "total senescent burden, the measurable quantity"),
      ("P", "SASP activity in the medium — a SECRETION, not a transcript"),
      ("X", "cells escaped from arrest; a sink that drives nothing")]),
    ("Neighbour population",
     [("Δln N", "log neighbour-cell count relative to concurrent control "
       "(raw ODE drifts as −d_N·t at P=0; the reported KCC10 observable "
       "cancels that basal drain)")]),
    ("Marker panel (derived)",
     [("m_p16, m_p21, m_lmnb1, m_sabg, m_ddr, m_sasp",
       "six markers computed from S with per-marker sensitivity, false-positive "
       "rate and lag; construct validity is tested on the PANEL, not on any "
       "single marker")]),
]


def param_comments() -> dict:
    """Unit and meaning for each Params field, from its trailing comment."""
    out: dict[str, str] = {}
    for line in SRC.read_text().splitlines():
        m = re.match(r"\s*([a-zA-Z_]\w*)\s*:\s*float\s*=\s*[^#]*#\s*(.+?)\s*$",
                     line)
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


PUBLIC_DATA_DIR = Path(__file__).parent / "public_data_results"


def public_data_block(results_dir: Path | None = None) -> dict | None:
    """Condense the public-data workflow into the few numbers the prose needs.

    Returns None when the workflow has not been run, so the manuscript can be
    built without it. Every number here is copied from
    ``complete_summary.json`` / ``summary.json``; nothing is recomputed, and
    nothing is pooled across cell systems.

    The three ``*_established`` flags are carried through deliberately. They
    are the workflow's own verdict that it does NOT establish EMD4, feedback or
    a biological q_sec_critical, and :func:`audit` refuses to write tables if
    any of them ever flips to True without the prose being rewritten.
    """
    d = Path(results_dir) if results_dir else PUBLIC_DATA_DIR
    complete, core = d / "complete_summary.json", d / "summary.json"
    if not complete.exists() or not core.exists():
        return None
    C = json.loads(complete.read_text())
    B = json.loads(core.read_text())

    # Held-out withdrawal prediction, per exposure arm.
    arms = []
    for arm, rec in C["withdrawal_comparisons"].items():
        fits = rec["fits"]
        fb = [f for f in fits if f["model"] == "feedback"]
        best = min(fits, key=lambda f: f["holdout_whitened_MSE"])
        best_fb = min(fb, key=lambda f: f["holdout_whitened_MSE"])
        last_obs = rec["forecast_baselines"]["last_observation"]["holdout_whitened_MSE"]
        best_train = min(fits, key=lambda f: f["train_whitened_SSE"])
        arms.append({
            "arm": arm,
            "holdout_day": rec["holdout_time_days"],
            "best_model": best["model"],
            "best_structure": best["structure"],
            "best_holdout_mse": best["holdout_whitened_MSE"],
            "feedback_holdout_mse": best_fb["holdout_whitened_MSE"],
            "last_observation_mse": last_obs,
            "last_observation_beats_feedback": bool(last_obs < best_fb["holdout_whitened_MSE"]),
            "best_train_model": best_train["model"],
            "feedback_is_best_train": bool(best_train["model"] == "feedback"),
            "feedback_kinetics_identified": bool(best_fb.get("kinetic_parameters_identified")),
            "feedback_boundary_parameters": best_fb.get("boundary_parameters", []),
            "n_fits_on_a_bound": sum(1 for f in fits if f.get("boundary_parameters")),
            "n_fits": len(fits),
        })
    arms.sort(key=lambda a: a["arm"])

    bench = B["synthetic_benchmark"]
    sabg = C["arsenite"]["SA_beta_gal_control_adjusted"]

    return {
        "EMD4_established": C["EMD4_established"],
        "feedback_established": C["feedback_established"],
        "q_sec_critical_calibrated": C["q_sec_critical_calibrated"],
        "withdrawal_arms": arms,
        "n_withdrawal_arms": len(arms),
        "n_arms_last_obs_beats_feedback": sum(
            1 for a in arms if a["last_observation_beats_feedback"]),
        "n_arms_feedback_best_train": sum(
            1 for a in arms if a["feedback_is_best_train"]),
        "benchmark": {
            "trials_per_truth": bench["trials_per_truth"],
            "false_feedback_selections": bench["false_feedback_selections"],
            "nonfeedback_trials": bench["nonfeedback_trials_with_a_winner"],
            "false_selection_fraction": bench["false_selection_fraction"],
            "wilson95": bench["wilson95"],
            "feedback_recovered_when_true":
                bench["confusion"]["feedback"]["feedback"],
            "feedback_trials": sum(bench["confusion"]["feedback"].values()),
        },
        "arsenite_sabg": sabg,
        "secretome_rows": C["secretome"]["reported_protein_rows"],
        "secretome_panel_rows": C["secretome"]["panel_rows"],
        "single_cell": C["single_cell"],
        "external_validation": C["external_validation"],
    }


def build(summary: dict) -> dict:
    p = Params()
    comments = param_comments()

    parameters = [
        {"name": f.name,
         "value": getattr(p, f.name),
         "meaning": comments.get(f.name, ""),
         "basis": PARAM_BASIS.get(f.name, "Chosen.")}
        for f in fields(Params)
    ]

    conditions = [
        {"key": c.key,
         "label": getattr(c, "label", c.key),
         "plotted": bool(getattr(c, "plotted", False)),
         **{k: v for k, v in summary["endpoints"].get(c.key, {}).items()}}
        for c in CONDITIONS
    ]

    checks = [{"cid": c.cid, "edge": c.edge, "statement": c.statement,
               "source": c.source, "passed": summary["checks"].get(c.cid)}
              for c in CHECKS]

    markers = [{"key": m.key,
                "sens": m.sens, "fp": m.fp, "stress": m.stress, "tau": m.tau}
               for m in MARKERS]

    return {
        "parameters": parameters,
        "state_groups": STATE_GROUPS,
        "state_index": {k: v for k, v in IDX.items()},
        "markers": markers,
        "conditions": conditions,
        "endpoints": summary["endpoints"],
        "checks": checks,
        "bifurcation": summary["bifurcation"],
        "null_comparison": summary["null_comparison"],
        "identifiability": summary["identifiability"],
        "spatial": summary.get("spatial"),
        "ensemble": summary.get("ensemble"),
        "finite_threshold": summary.get("finite_threshold"),
        "derived_quantities": summary.get("derived_quantities"),
        "qss_reduction": summary.get("qss_reduction"),
        # None when the public-data workflow has not been run; the generators
        # omit the corresponding sections in that case rather than inventing
        # numbers.
        "public_data": public_data_block(),
        "protocol": {"dose": DOSE, "dose_low": DOSE_LOW, "t_off": T_OFF,
                     "t_iv": T_IV, "t_end": T_END, "persist_threshold": PERSIST},
        "n_states": len(IDX),
        "n_checks": len(CHECKS),
        "n_checks_passed": sum(1 for c in CHECKS
                               if summary["checks"].get(c.cid)),
    }


def audit(tables: dict) -> list[str]:
    """Reasons this summary must not be turned into manuscript text.

    A partial run (`--no-identifiability`, `--n-ensemble 0`, `--no-spatial`)
    writes a perfectly well-formed summary JSON in which checks have silently
    failed and quantities are NaN. Nothing downstream would notice: the docx
    generators read this file and `build_docx.sh` does not rerun the simulation
    unless asked. So the gate belongs here, between the run and the manuscript.
    """
    problems = []

    failed = [c["cid"] for c in tables["checks"] if not c["passed"]]
    if failed:
        problems.append(
            f"{len(failed)} validation check(s) did not pass: {', '.join(failed)}")

    for key, val in tables["identifiability"].items():
        if val is None or (isinstance(val, float) and val != val):
            problems.append(f"identifiability.{key} is {val!r} "
                            "(run without --no-identifiability)")

    for name in ("spatial", "ensemble", "finite_threshold",
                 "derived_quantities", "qss_reduction"):
        if tables.get(name) is None:
            problems.append(f"{name} block missing from the summary")

    qss = tables.get("qss_reduction") or {}
    if not qss.get("passed", False):
        problems.append(
            "the QSS reduction to the manuscript equation did not hold "
            "(worst |diff| "
            f"{qss.get('worst_abs_diff_manuscript_form')!r}) -- the docx "
            "asserts this reduction in equation (6)")

    ens = tables.get("ensemble") or {}
    # All three bistability quantities must be present, because quoting any
    # one of them alone is how the retention count came to be tabulated as
    # "Bistable at E = 0, robust".
    for name in ("frac_empty_state_lost", "frac_high_state_retained",
                 "frac_two_stable_equilibria", "frac_empty_plus_persistent",
                 "frac_bistable_without_empty_state"):
        if ens.get(name) is None:
            problems.append(f"ensemble.{name} missing; the bistability "
                            "figure cannot be reported honestly without it")

    # Persistence must be reportable as a PAIRED contrast. Without it the
    # headline count includes draws whose unexposed control also ran away,
    # which is not an exposure effect.
    for name in ("frac_control_persistent", "frac_persist_exposure_attributable"):
        if ens.get(name) is None:
            problems.append(
                f"ensemble.{name} missing; reference-arm persistence would be "
                "reportable only as an unpaired count, which does not "
                "establish that the exposure caused it")

    # The marker over-read must be available on BOTH scales. The absolute one
    # alone cannot be written up: its lower tail is the marker's unexposed
    # false-positive rate, a structural constant. Its sampled interval is not a
    # universal lower bound, so the control-adjusted scale carries the claim.
    for name in ("v3_overstate_points_net", "frac_v3_overstate_net_positive"):
        if ens.get(name) is None:
            problems.append(
                f"ensemble.{name} missing; the SA-beta-gal over-read would be "
                "reportable only on the absolute scale, which includes the "
                "assay background the concurrent control also carries")

    # The public-data workflow reports its own verdict on what it does and does
    # not establish. The manuscript prose is written against all three being
    # False. If the workflow is ever extended so that one flips, the prose is
    # stale by definition and must be rewritten before tables are regenerated.
    pub = tables.get("public_data")
    if pub is not None:
        for name in ("EMD4_established", "feedback_established",
                     "q_sec_critical_calibrated"):
            if pub.get(name):
                problems.append(
                    f"public_data.{name} is True, but the manuscript text is "
                    "written for the case where the public data do NOT "
                    "establish it. Rewrite those sections before regenerating")
        arms = pub.get("withdrawal_arms") or []
        if arms and not any(a["last_observation_beats_feedback"] for a in arms):
            problems.append(
                "no withdrawal arm has a last-observation forecast beating "
                "feedback any more, but the Results say two of three do; "
                "recheck the prose against the current numbers")

    # A censored profile grid means the admissible k_p range ran off the top of
    # the grid, so the ridge is a lower bound and its upper endpoint is an
    # artefact of where the grid stopped. That must not reach a table.
    ident = tables.get("identifiability") or {}
    for name in ("kp_ridge_censored_continuous",
                 "kp_ridge_censored_with_withdrawal"):
        if ident.get(name):
            problems.append(
                f"identifiability.{name} is True: the admissible k_p set "
                "extends past the top of the profile grid, so the ridge width "
                "is a LOWER BOUND and the range must not be quoted as an "
                "interval. Widen identifiability.KP_GRID_HI and rerun")

    # Spatial band edges of None mean the scan never fully contained or never
    # reached majority escape; V10's band is then undetermined, not wide.
    sp = tables.get("spatial") or {}
    for name in ("q_contained", "q_majority_escape"):
        if sp.get(name) is None:
            problems.append(
                f"spatial.{name} is None: the containment band V10 tests was "
                "not determined by this scan")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures/output")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write the tables even if the run was partial or "
                         "failed; for debugging only, never for a manuscript")
    args = ap.parse_args()

    out = Path(args.outdir)
    summary = json.loads((out / "emd4_simulation_summary.json").read_text())
    tables = build(summary)

    problems = audit(tables)
    if problems:
        print("REFUSING to write manuscript tables from this summary:")
        for why in problems:
            print(f"  - {why}")
        print("\n  The summary in "
              f"{out / 'emd4_simulation_summary.json'} is from a partial or "
              "failed run.\n  Rerun `python -m emd4_simulation.run` with no "
              "skip flags, then retry.")
        if not args.allow_incomplete:
            return 1
        print("\n  --allow-incomplete given; writing anyway. DO NOT PUBLISH THIS.")

    path = out / "emd4_manuscript_tables.json"
    # allow_nan=False so a NaN can never reach the docx generators: bare NaN is
    # not valid JSON and JSON.parse rejects the entire file, so one undefined
    # quantity would take the whole manuscript build down with it.
    path.write_text(json.dumps(json_safe(tables), indent=2, default=float,
                               allow_nan=False))
    print(f"wrote {path}  ({len(tables['parameters'])} parameters, "
          f"{tables['n_checks']} checks, {tables['n_states']} states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
