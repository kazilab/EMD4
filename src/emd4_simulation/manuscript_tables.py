"""Export model parameters, checks and results from the EMD4 simulation."""

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

PARAM_BASIS = {
    "q_sec": "Assigned ratio of secondary to primary functional SASP secretion. "
             "At fixed parameters it controls whether two stable equilibria "
             "coexist. The critical ratio also depends on receptor sensitivity, "
             "SASP clearance, paracrine gain and cooperativity, and removal of "
             "senescent cells. Protein abundance alone cannot locate this threshold.",
    "k_p": "Assigned maximum rate of paracrine induction.",
    "n_P": "Assigned cooperativity of the paracrine response. At the nominal "
           "parameters, setting this exponent to one removes bistability.",
    "gam_immune": "Assigned rate of immune-mediated clearance of senescent cells.",
    "k_esc": "Assigned rate of escape from arrest into the terminal compartment X.",
    "tau_C": "Assigned delay between arrest and SASP secretion.",
}

STATE_GROUPS = [
    ("Redox and exposure",
     [("R", "reactive oxygen species level"),
      ("stress", "stress from continuing exposure")]),
    ("Senescent compartments",
     [("S₁", "primary senescent cells induced by exposure"),
      ("S₂", "secondary senescent cells induced by paracrine SASP activity"),
      ("S = S₁ + S₂", "total senescent fraction"),
      ("P", "secreted SASP activity in the medium"),
      ("X", "cells that have escaped arrest")]),
    ("Neighbour population",
     [("Δln N", "log neighbour-cell count relative to the concurrent control")]),
    ("Marker panel (derived)",
     [("m_p16, m_p21, m_lmnb1, m_sabg, m_ddr, m_sasp",
       "six marker readouts with marker-specific sensitivity, false-positive "
       "rate, stress response and lag")]),
]


def param_comments() -> dict:
    """Read parameter definitions and units from the model."""
    out: dict[str, str] = {}
    for line in SRC.read_text().splitlines():
        m = re.match(r"\s*([a-zA-Z_]\w*)\s*:\s*float\s*=\s*[^#]*#\s*(.+?)\s*$",
                     line)
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


def public_data_block(results_dir: Path | None = None) -> dict | None:
    """Read empirical comparisons and the synthetic benchmark from a completed data analysis."""
    if results_dir is None:
        return None
    d = Path(results_dir)
    complete, core = d / "complete_summary.json", d / "summary.json"
    if not complete.is_file() or not core.is_file():
        raise FileNotFoundError(f"Run the complete public-data workflow in {d} first.")
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


def build(summary: dict, public_results: Path | None = None) -> dict:
    p = Params()
    comments = param_comments()

    parameters = [
        {"name": f.name,
         "value": getattr(p, f.name),
         "meaning": comments.get(f.name, ""),
         "basis": PARAM_BASIS.get(f.name, "Assigned.")}
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
        "public_data": public_data_block(public_results),
        "protocol": {"dose": DOSE, "dose_low": DOSE_LOW, "t_off": T_OFF,
                     "t_iv": T_IV, "t_end": T_END, "persist_threshold": PERSIST},
        "n_states": len(IDX),
        "n_checks": len(CHECKS),
        "n_checks_passed": sum(1 for c in CHECKS
                               if summary["checks"].get(c.cid)),
    }


def audit(tables: dict) -> list[str]:
    """List incomplete results and departures from the reference model checks."""
    problems = []
    failed = [c["cid"] for c in tables["checks"] if not c["passed"]]
    if failed:
        problems.append(f"Failed model checks: {', '.join(failed)}")

    for key, val in tables["identifiability"].items():
        if val is None or (isinstance(val, float) and val != val):
            problems.append(f"Missing identifiability result: {key}")

    for name in ("spatial", "ensemble", "finite_threshold",
                 "derived_quantities", "qss_reduction"):
        if tables.get(name) is None:
            problems.append(f"Missing simulation results: {name}")

    qss = tables.get("qss_reduction") or {}
    if not qss.get("passed", False):
        problems.append("Quasi-steady-state reduction failed "
                        f"(maximum difference {qss.get('worst_abs_diff_manuscript_form')!r})")

    ens = tables.get("ensemble") or {}
    for name in ("frac_empty_state_lost", "frac_high_state_retained",
                 "frac_two_stable_equilibria", "frac_empty_plus_persistent",
                 "frac_bistable_without_empty_state", "frac_control_persistent",
                 "frac_persist_exposure_attributable", "v3_overstate_points_net",
                 "frac_v3_overstate_net_positive"):
        if ens.get(name) is None:
            problems.append(f"Missing ensemble result: {name}")

    pub = tables.get("public_data")
    if pub is not None:
        for name in ("EMD4_established", "feedback_established",
                     "q_sec_critical_calibrated"):
            if pub.get(name):
                problems.append(f"Public-data result differs from the reference: {name}")
        arms = pub.get("withdrawal_arms") or []
        if arms and not any(a["last_observation_beats_feedback"] for a in arms):
            problems.append("The last-observation forecast did not outperform "
                            "feedback in any withdrawal arm")

    ident = tables.get("identifiability") or {}
    for name in ("kp_ridge_censored_continuous",
                 "kp_ridge_censored_with_withdrawal"):
        if ident.get(name):
            problems.append(f"{name}: the admissible range reached the upper "
                            "edge of the paracrine rate grid")

    sp = tables.get("spatial") or {}
    for name in ("q_contained", "q_majority_escape"):
        if sp.get(name) is None:
            problems.append(f"The spatial scan did not resolve {name}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "reproduced")
    ap.add_argument("--public-results", type=Path, help="optional complete public-data output directory")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="export tables even when checks fail or results are missing")
    args = ap.parse_args()

    out = Path(args.outdir)
    summary = json.loads((out / "emd4_simulation_summary.json").read_text())
    tables = build(summary, args.public_results)

    problems = audit(tables)
    if problems:
        print("Table validation found the following issues:")
        for why in problems:
            print(f"  - {why}")
        if not args.allow_incomplete:
            return 1
        print("Exporting partial results as requested.")

    path = out / "emd4_manuscript_tables.json"
    path.write_text(json.dumps(json_safe(tables), indent=2, default=float,
                               allow_nan=False))
    print(f"wrote {path}  ({len(tables['parameters'])} parameters, "
          f"{tables['n_checks']} checks, {tables['n_states']} states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
