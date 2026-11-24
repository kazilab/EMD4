"""Exposure conditions, intervention predictions and model checks for senescence persistence.

The arsenite withdrawal study provides qualitative evidence for persistence.
The dose scale and timing used here are assigned. Senolytic treatment, SASP
neutralisation and altered immune clearance are model predictions for this
cell system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Intervention, Params

DOSE = 1.0          # dimensionless reference arsenite exposure
DOSE_LOW = 0.15     # below the finite-duration threshold at 14 days
T_OFF = 14.0        # assigned withdrawal day
T_IV = 60.0         # interventions begin once the state has established
T_END = 400.0

# The persistence threshold lies between the nominal unstable and high stable states.
PERSIST = 0.30


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    E: float
    iv: Intervention
    t_off: float = T_OFF
    plotted: bool = False


CONDITIONS = [
    Condition("control", "Control", 0.0, Intervention(), t_off=np.inf, plotted=True),
    Condition("as", "Arsenite, withdrawn d14", DOSE, Intervention(), plotted=True),
    Condition("as_low", "Arsenite low dose, withdrawn d14", DOSE_LOW,
              Intervention(), plotted=True),
    Condition("as_sasp", "Arsenite + SASP neutralisation d60", DOSE,
              Intervention(sasp_neut=0.25, t_start=T_IV), plotted=True),
    Condition("as_cont", "Arsenite, continuous", DOSE, Intervention(), t_off=np.inf),
    Condition("as_seno", "Arsenite + senolytic d60", DOSE,
              Intervention(senolytic=6.0, t_start=T_IV)),
    Condition("as_immune_lo", "Arsenite + impaired immune clearance", DOSE,
              Intervention(immune=0.25)),
    Condition("as_immune_hi", "Arsenite + enhanced immune clearance", DOSE,
              Intervention(immune=3.0)),
    Condition("as_qsec_lo", "Arsenite + poor secondary secretion", DOSE,
              Intervention(q_sec=0.40)),      # q_sec 0.50 -> 0.20, below q_sec*
    Condition("as_escape", "Arsenite + elevated escape", DOSE,
              Intervention(escape=10.0)),
]

BY_KEY = {c.key: c for c in CONDITIONS}
PLOTTED = [c for c in CONDITIONS if c.plotted]


@dataclass(frozen=True)
class Check:
    cid: str
    edge: str
    statement: str
    test: object
    source: str


def _net_over(c: dict, marker: str) -> float:
    """Day-14 marker excess over the latent state after subtracting the same gap in control."""
    low = c["d14"]["as_low"]
    ctrl = c["d14"]["control"]
    return (low[marker] - low["S"]) - (ctrl[marker] - ctrl["S"])


def _gt(a, b, margin=1.05):
    return a > b * margin


def _lt(a, b, margin=0.95):
    return a < b * margin


CHECKS = [
    Check("V1", "KCC5 to EMD4",
          "Under arsenite exposure, oxidative stress rises before senescent burden. "
          "After withdrawal, ROS returns to its control level while senescence "
          "persists.",
          lambda c: (_gt(c["d14"]["as"]["R"], c["d14"]["control"]["R"])
                     and c["d5"]["as"]["stress"] > 0.90 * c["d14"]["as"]["stress"]
                     and c["d5"]["as"]["S"] < 0.50 * c["end"]["as"]["S"]
                     and c["end"]["as"]["S"] > PERSIST
                     and abs(c["end"]["as"]["R"] - c["end"]["control"]["R"]) < 1e-3),
          "Okamura et al. reported premature senescence in Huh-7 cells (2024, EHPM "
          "29:74) and hepatic stellate cells (2022, TAP 454:116231) after arsenite "
          "exposure."),

    Check("V2", "Persistence after withdrawal",
          "Senescent burden exceeds the persistence threshold on day 400 after "
          "withdrawal on day 14. Both fitted feedback-free models return near baseline.",
          lambda c: (c["end"]["as"]["S"] > PERSIST
                     and c["null"]["simple_end"] < 0.05
                     and c["null"]["cascade_end"] < 0.05),
          "Okamura et al. (2024) observed senescence features and SASP factors after "
          "arsenite withdrawal in Huh-7 cells. The simulated follow-up tests "
          "persistence over an assigned duration."),

    Check("V3", "Senescence marker specificity",
          "In the low-dose arm on day 14, SA-beta-gal and DDR foci each exceed three "
          "times the latent senescent fraction. Each marker retains a positive excess "
          "after subtracting the concurrent control.",
          lambda c: (c["d14"]["as_low"]["m_sabg"] > 3.0 * c["d14"]["as_low"]["S"]
                     and c["d14"]["as_low"]["m_ddr"] > 3.0 * c["d14"]["as_low"]["S"]
                     and _net_over(c, "m_sabg") > 0.0
                     and _net_over(c, "m_ddr") > 0.0),
          "Ogrodnik et al. (2024, Cell 187:4150) and SenNet (2024, NRMCB 25:1001) "
          "recommend measuring distinct senescence hallmarks. The absolute marker gap "
          "includes baseline assay positivity. The ensemble evaluates the sign of the "
          "control-adjusted gap and the marker response after withdrawal."),

    Check("V4", "EMD4 to KCC6",
          "Under continuous exposure, SASP activity reaches its half-rise later than "
          "the primary senescent fraction.",
          lambda c: c["peak_t"]["as_cont"]["P"] > c["peak_t"]["as_cont"]["S1"],
          "The model represents a delay between primary senescence and extracellular "
          "secretion. Transcript measurements alone cannot establish secreted SASP "
          "activity."),

    Check("V5", "Paracrine amplification",
          "After withdrawal, secondary senescent cells account for most of the "
          "persistent burden and reach their half-rise later than primary cells.",
          lambda c: (c["end"]["as"]["secondary_fraction"] > 0.5
                     and c["peak_t"]["as"]["S2"] > c["peak_t"]["as"]["S1"]),
          "SASP factors can induce senescence in neighbouring cells (Coppé et al., 2010 "
          "and Martin et al., 2023, Aging Cell 22:e13892)."),

    Check("V6", "Bistability",
          "The nominal model has a positive hysteresis area and loses its low stable "
          "branch at a finite exposure.",
          lambda c: (c["bif"]["width"] > 0.01
                     and c["bif"]["fold"] is not None),
          "The fold and hysteresis are predictions conditional on the assigned "
          "parameters."),

    Check("V7", "Treatment of established senescence",
          "SASP neutralisation reduces the established senescent fraction below 0.05. "
          "Senolytic treatment at the same time reduces the burden but leaves more than "
          "0.20 senescent.",
          lambda c: (c["end"]["as_sasp"]["S"] < 0.05
                     and c["end"]["as_seno"]["S"] < c["end"]["as"]["S"]
                     and c["end"]["as_seno"]["S"] > 0.20),
          "These interventions predict responses in the simulated cell system. Alimirah "
          "et al. (2020, Cancer Res 80:3606) studied p16-positive cell ablation in skin "
          "tumour progression."),

    Check("V8", "EMD4 to KCC7",
          "Impaired immune clearance increases the senescent burden, and enhanced "
          "clearance decreases it.",
          lambda c: (c["end"]["as_immune_lo"]["S"] > c["end"]["as"]["S"]
                     and c["end"]["as_immune_hi"]["S"] < c["end"]["as"]["S"]),
          "The KCC7 interpretation applies to impaired immune clearance."),

    Check("V9", "EMD4 and KCC9",
          "Increasing the rate of escape from arrest raises cumulative escape and "
          "lowers the senescent burden.",
          lambda c: (c["end"]["as_escape"]["X"] > c["end"]["as"]["X"]
                     and c["end"]["as_escape"]["S"] < c["end"]["as"]["S"]),
          "Escape is a terminal flux driven by senescent burden. Its cumulative event "
          "count can exceed one and does not measure acquisition of an immortal "
          "phenotype."),

    Check("V10", "Containment threshold",
          "At fixed nominal parameters, the mean-field secretion threshold lies between "
          "the largest ratio with no observed spatial escape and the smallest ratio "
          "with majority escape. A lower secondary secretion ratio reduces the final "
          "burden.",
          lambda c: (c["bif"]["q_crit"] is not None
                     and c["end"]["as_qsec_lo"]["S"] < 0.05
                     and c["end"]["as"]["S"] > PERSIST
                     and c["spatial"] is not None
                     and c["spatial"]["q_contained"] is not None
                     and c["spatial"]["q_majority_escape"] is not None
                     and (c["spatial"]["q_contained"]
                          <= c["bif"]["q_crit"]
                          <= c["spatial"]["q_majority_escape"])),
          "Martin et al. (2023, Aging Cell 22:e13892) found that differences in "
          "signalling by primary and secondary senescent cells limit spread. The "
          "threshold also depends on paracrine gain, receptor sensitivity, "
          "cooperativity and rates of clearance. The model combines secondary-cell "
          "subtypes."),

    Check("V11", "Comparison with feedback-free models",
          "The simple null reproduces the forward endpoint dose response. Both "
          "feedback-free models deviate from the continuous time course and predict "
          "less persistence after withdrawal.",
          lambda c: (c["null"]["endpoint_r2"] > 0.99
                     and c["null"]["timecourse_r2"] < 0.99
                     and c["null"]["cascade_r2"] < 0.99
                     and c["null"]["worst_dev_simple"] > 0.05
                     and c["null"]["worst_dev_cascade"] > 0.05
                     and c["end"]["as"]["S"] - c["null"]["simple_end"] > 0.5
                     and c["end"]["as"]["S"] - c["null"]["cascade_end"] > 0.5),
          "The models are fitted to synthetic continuous-exposure data. The withdrawal "
          "prediction tests their dynamics under a different exposure schedule."),

    Check("V12", "Feedback identifiability",
          "A range of feedback strengths fits the continuous-exposure trajectory within "
          "the assigned noise threshold. The fit excludes zero feedback.",
          lambda c: (c["ident"]["kp_ridge_width"] > 0.20
                     and not c["ident"]["null_admissible"]),
          "The profile refits direct commitment and total clearance with the other "
          "parameters fixed. The admissible range describes practical identifiability "
          "under this simulated design."),

    Check("V13", "EMD4 to KCC10",
          "Relative to the time-matched control, neighbour expansion is positive after "
          "reference-dose withdrawal. Expansion is smaller after low-dose exposure or "
          "SASP neutralisation.",
          lambda c: (abs(c["end"]["control"]["lnN"]) < 1e-9
                     and c["end"]["as"]["lnN"] > 1.0
                     and c["end"]["as"]["lnN"] > c["end"]["as_low"]["lnN"]
                     and c["end"]["as"]["lnN"] > c["end"]["as_sasp"]["lnN"]),
          "Subtracting the concurrent control removes baseline neighbour turnover from "
          "log expansion. The model attributes the remaining expansion to SASP "
          "activity."),
]
