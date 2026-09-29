"""Simulated conditions and the published results each one is checked against.

Four conditions are plotted:

    control -> arsenite (persistent) -> arsenite (contained) -> + SASP neutralisation

The rest are held out of the figure and used as sign-test validation. As in the
EMD1 build, several arms tune no parameter at all, so agreement there is
prediction rather than fit.

A note on what "held out" can mean here. EMD1 could hold out whole published
experiments. EMD4's withdrawal arm is published (Okamura 2024) but its
senolytic, SASP-neutralisation and immune-clearance arms are not available for
this cell system, so those are *predictions awaiting data* rather than
retrospective validation. They are labelled as such in the battery and must not
be reported as agreement with measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Intervention, Params

DOSE = 1.0          # dimensionless; E = 1 is the reference arsenite exposure
DOSE_LOW = 0.15     # below the finite-duration threshold at 14 days
T_OFF = 14.0        # arsenite withdrawn on day 14 (the Okamura protocol)
T_IV = 60.0         # interventions begin once the state has established
T_END = 400.0

# The senescent fraction above which the state is called persistent. Set well
# above the unstable separatrix (~0.18) and well below the upper branch (~0.92)
# so the classification is not sensitive to where exactly it sits.
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
    # --- held out of the figure ------------------------------------------
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


# --- validation battery ---------------------------------------------------
# Each check is (id, edge, statement, callable(ctx) -> bool, source).
# ``ctx`` carries endpoint dicts plus the derived analyses, so a check can ask
# about a bifurcation or a model comparison and not only about a trajectory.

@dataclass(frozen=True)
class Check:
    cid: str
    edge: str
    statement: str
    test: object
    source: str


def _net_over(c: dict, marker: str) -> float:
    """Marker-minus-latent-state gap at day 14, NET of the concurrent control.

    ``m - S`` on the absolute scale includes the marker's unexposed
    false-positive rate, which the control arm carries at the same instant. The
    netted quantity is the part attributable to the exposure, and it is the one
    a construct-validity claim has to rest on.
    """
    low = c["d14"]["as_low"]
    ctrl = c["d14"]["control"]
    return (low[marker] - low["S"]) - (ctrl[marker] - ctrl["S"])


def _gt(a, b, margin=1.05):
    return a > b * margin


def _lt(a, b, margin=0.95):
    return a < b * margin


CHECKS = [
    Check("V1", "KCC5 -> EMD4",
          "Arsenite raises ROS DURING exposure and commits cells to "
          "senescence, and the commitment lags the ROS rise. ROS returns to "
          "baseline after withdrawal while the senescent state does not -- the "
          "upstream driver is transient, the domain event is not",
          lambda c: (_gt(c["d14"]["as"]["R"], c["d14"]["control"]["R"])
                     and c["d5"]["as"]["stress"] > 0.90 * c["d14"]["as"]["stress"]
                     and c["d5"]["as"]["S"] < 0.50 * c["end"]["as"]["S"]
                     and c["end"]["as"]["S"] > PERSIST
                     and abs(c["end"]["as"]["R"] - c["end"]["control"]["R"]) < 1e-3),
          "Arsenite-induced ROS; Okamura 2024 (EHPM 29:74) Huh-7 premature "
          "senescence; Okamura 2022 (TAP 454:116231) hepatic stellate cells"),

    Check("V2", "persistence (CENTRAL REQUIREMENT)",
          "The senescent state persists to day 400 after withdrawal on day 14, "
          "while the fitted no-feedback null relaxes to baseline",
          lambda c: (c["end"]["as"]["S"] > PERSIST
                     and c["null"]["simple_end"] < 0.05
                     and c["null"]["cascade_end"] < 0.05),
          "Okamura 2024: senescence features and SASP factors persist after "
          "arsenite withdrawal. This is the observation the whole domain rests "
          "on, and the one a single-valued dose-response cannot produce."),

    Check("V3", "multi-marker (CONSTRUCT VALIDITY)",
          "SA-beta-gal and DDR foci OVERSTATE the latent state during exposure: "
          "both read high in the low-dose arm while S is still low. The claim "
          "is deliberately ONE-DIRECTIONAL",
          # Two clauses per marker, and the second one is the load-bearing one.
          # The first says the marker reads far above the latent state on the
          # ABSOLUTE scale -- which is what an experimenter sees, but which any
          # marker with a non-zero unexposed false-positive rate gets partly for
          # free. The second nets off the concurrent control at the same
          # instant, exactly as apply_control_relative_lnN does for KCC10, and
          # so asserts that the over-read is attributable to the EXPOSURE and
          # not to the assay's own background.
          lambda c: (c["d14"]["as_low"]["m_sabg"] > 3.0 * c["d14"]["as_low"]["S"]
                     and c["d14"]["as_low"]["m_ddr"] > 3.0 * c["d14"]["as_low"]["S"]
                     and _net_over(c, "m_sabg") > 0.0
                     and _net_over(c, "m_ddr") > 0.0),
          "Ogrodnik 2024 (Cell 187:4150) and SenNet 2024 (NRMCB 25:1001): "
          "SA-beta-gal is neither necessary nor sufficient; consensus requires "
          "markers spanning distinct senescence hallmarks. "
          "NARROWED DELIBERATELY: this check previously also asserted that the "
          "marker FALLS after withdrawal while S persists. At nominal "
          "parameters that clause passed only on a 0.001 margin (m_sabg 0.8849 "
          "-> 0.8839 while S rose 0.719 -> 0.924), and the ensemble finds it in "
          "just 111/250 draws. Asserting it here would have recorded a 'pass' "
          "for a claim the uncertainty layer rejects, so the battery now tests "
          "only the overstatement. On the ABSOLUTE scale that reads +12.2 pts "
          "[8.0, 28.3], but its apparent sign-robustness is an artefact and is "
          "not claimed -- see the second narrowing below. The discarded direction is reported as "
          "NOT robust in the ensemble table and is not claimed anywhere in the "
          "manuscript. "
          "SECOND NARROWING: the absolute over-read is inflated by the marker's "
          "unexposed false-positive rate (m_sabg.fp = 0.08), which is present "
          "in the concurrent control at the same instant and is a structural "
          "constant the ensemble does not sample. It helps explain why the "
          "sampled absolute interval stays positive, but it is not a universal "
          "positive lower bound: at zero stress and marker steady state, the "
          "nominal y - S becomes negative above S = 0.615. "
          "Netting the control off leaves +5.8 pts at nominal parameters and a "
          "median +3.5 pts [-1.7, +15.9] across draws, positive in 84% of them: "
          "FRAGILE, not sign-robust. Both scales are reported and the check now "
          "asserts the netted one."),

    Check("V4", "EMD4 -> KCC6 (SASP as secretion, not transcript)",
          "SASP activity is a distinct delayed state, not a relabelling of S: "
          "it lags the senescent fraction on establishment",
          lambda c: c["peak_t"]["as_cont"]["P"] > c["peak_t"]["as_cont"]["S1"],
          "Transcript-level SASP increases do not establish secretion or "
          "extracellular concentration; the delay between arrest and a "
          "functionally active SASP is a required model feature."),

    Check("V5", "paracrine amplification",
          "Most of the persistent burden is SECONDARY senescence, and it "
          "arrives after the primary population",
          lambda c: (c["end"]["as"]["secondary_fraction"] > 0.5
                     and c["peak_t"]["as"]["S2"] > c["peak_t"]["as"]["S1"]),
          "SASP components released by senescent cells induce senescence in "
          "neighbours (Coppe 2010; Martin 2023, Aging Cell 22:e13892)."),

    Check("V6", "bistability",
          "A hysteresis loop of non-zero AREA exists, and the low branch is "
          "destroyed at a finite fold exposure",
          lambda c: (c["bif"]["width"] > 0.01
                     and c["bif"]["fold"] is not None),
          "Candidate model only. The manuscript is explicit that whether "
          "threshold behaviour occurs is an empirical question; this check "
          "records what the nominal parameters imply, not what is measured."),

    Check("V7", "intervention (mechanism, not just burden)",
          "Breaking the FEEDBACK collapses the established state, while "
          "removing CELLS at the same time point does not",
          lambda c: (c["end"]["as_sasp"]["S"] < 0.05
                     and c["end"]["as_seno"]["S"] < c["end"]["as"]["S"]
                     and c["end"]["as_seno"]["S"] > 0.20),
          "PREDICTION, not retrospective agreement: no senolytic or "
          "SASP-neutralisation arm exists for this cell system. Alimirah 2020 "
          "(Cancer Res 80:3606) shows p16-positive ablation reduces skin "
          "tumour progression, which is the closest published analogue."),

    Check("V8", "EMD4 -> KCC7 (DIRECTION SENSITIVE)",
          "Impaired immune clearance raises the senescent burden; effective "
          "immune clearance lowers it. Only the first is a KCC7-positive "
          "observation",
          lambda c: (c["end"]["as_immune_lo"]["S"] > c["end"]["as"]["S"]
                     and c["end"]["as_immune_hi"]["S"] < c["end"]["as"]["S"]),
          "The ontology rule: recruitment of immune cells that promotes "
          "effective clearance of senescent cells must NOT be recorded as a "
          "positive KCC7 observation. Encoding it as a signed check makes the "
          "rule structural rather than editorial."),

    Check("V9", "EMD4 -/-> KCC9 (OPPOSING POLARITY)",
          "Raising the ESCAPE RATE raises cumulative escape X while LOWERING "
          "the senescent burden S. (This is NOT the claim that raising S "
          "cannot raise X -- dX/dt = k_esc*S, so it can and does.)",
          lambda c: (c["end"]["as_escape"]["X"] > c["end"]["as"]["X"]
                     and c["end"]["as_escape"]["S"] < c["end"]["as"]["S"]),
          "The refined KCC framework treats senescence as biologically opposed "
          "to immortalization. Exposure-induced senescence must not be scored "
          "as evidence that an agent causes immortalization; escape is a later, "
          "separately evaluated event. That is an EVIDENTIARY RULE imposed on "
          "the model, not a result derived from it: X is terminal (no outgoing "
          "edge) but is still driven by S, and X counts cumulative escape "
          "events -- it is not a live-cell fraction and not an assay of "
          "immortalisation, which this model does not represent at all."),

    Check("V10", "containment threshold (MEASURABLE, TWO WAYS)",
          "Bistability is a condition on secondary-cell secretion: below a "
          "critical q_sec the loop cannot self-sustain. The mean-field "
          "threshold must fall between the largest q at which EVERY replicate "
          "is contained and the smallest at which the MAJORITY escape",
          # The spatial arm is REQUIRED, not optional: `spatial is None` means
          # the run was launched with --no-spatial, and a check that silently
          # passes when half its evidence was skipped is not a check. This
          # mirrors V12, which fails under --no-identifiability.
          # Both band edges must be MEASURED. They used to fall back to 0.0 and
          # 1.0 when no q value was fully contained or fully escaping, which
          # degraded the test to `0 <= q_crit <= 1` -- a condition every
          # possible q_crit satisfies. They are None now when undetermined, and
          # None here is a failure, not a pass.
          lambda c: (c["bif"]["q_crit"] is not None
                     and c["end"]["as_qsec_lo"]["S"] < 0.05
                     and c["end"]["as"]["S"] > PERSIST
                     and c["spatial"] is not None
                     and c["spatial"]["q_contained"] is not None
                     and c["spatial"]["q_majority_escape"] is not None
                     and (c["spatial"]["q_contained"]
                          <= c["bif"]["q_crit"]
                          <= c["spatial"]["q_majority_escape"])),
          "Martin 2023 (Aging Cell 22:e13892): differences in signalling output "
          "between primary and secondary senescent cells limit spread. HOLDING "
          "THE OTHER ASSUMED PARAMETERS FIXED, that ratio decides whether a "
          "tipping point exists here -- but it is not the only quantity that "
          "sets one. The analytic threshold q* = (K_P d_P / k_sasp) * "
          "[n / (r S*^(n-1))]^(1/n), with r = k_p/(gamma + k_esc), also moves "
          "with receptor sensitivity, SASP clearance, paracrine gain, "
          "cooperativity and senescent-cell removal, so a secretome ratio alone "
          "cannot place a real system on one side of it. Note also that Martin "
          "et al. distinguish paracrine from juxtacrine secondary cells, which "
          "this implementation merges, and that the bistable set is not always "
          "a half-line in q_sec: in 19/250 draws it is a window that closes "
          "again at high secretion. Above that upper fold the LOW branch is "
          "the one destroyed, leaving a monostable high burden with no healthy "
          "state -- the worst case, not a recovery of containment."),

    Check("V11", "NULL COMPARISON (THE HEADLINE)",
          "A no-feedback model matches the forward ENDPOINT dose-response "
          "almost exactly; the time course and the withdrawal arm separate them",
          # Every clause of the statement is now asserted. Previously the
          # time-course half was claimed in prose but never tested, so a null
          # that also matched the time course would still have passed.
          lambda c: (c["null"]["endpoint_r2"] > 0.99
                     and c["null"]["timecourse_r2"] < 0.99
                     and c["null"]["cascade_r2"] < 0.99
                     and c["null"]["worst_dev_simple"] > 0.05
                     and c["null"]["worst_dev_cascade"] > 0.05
                     and c["end"]["as"]["S"] - c["null"]["simple_end"] > 0.5
                     and c["end"]["as"]["S"] - c["null"]["cascade_end"] > 0.5),
          "This is the EMD4 analogue of EMD1's global-m6A scalar. The forward "
          "endpoint dose-response is the experiment almost every published "
          "study runs, and it is the one that cannot decide the question. A "
          "densely sampled time course at a single dose already can, because "
          "no monotone relaxation reproduces an autocatalytic rise -- which is "
          "a cheaper discriminating experiment than the withdrawal arm."),

    Check("V12", "identifiability (MAGNITUDE vs EXISTENCE)",
          "The STRENGTH of the paracrine feedback is not identifiable from "
          "S(t) -- k_p admits a wide ridge -- even though its EXISTENCE is: "
          "k_p = 0 is rejected",
          lambda c: (c["ident"]["kp_ridge_width"] > 0.20
                     and not c["ident"]["null_admissible"]),
          "Reported as a finding, not hidden. The distinction matters for the "
          "ontology: an EMD4 annotation can be supported by data that cannot "
          "quantify the feedback it invokes. A fitted k_p should therefore "
          "never be reported as a measured biological quantity."),

    Check("V13", "EMD4 -> KCC10 (conditional neighbour expansion)",
          "Neighbour-compartment expansion is reported as ΔlnN versus "
          "concurrent control. Control sits at zero; the reference "
          "withdrawal arm expands; the low-dose arm expands less; and SASP "
          "neutralisation truncates expansion relative to the untreated "
          "withdrawal arm",
          lambda c: (abs(c["end"]["control"]["lnN"]) < 1e-9
                     and c["end"]["as"]["lnN"] > 1.0
                     and c["end"]["as"]["lnN"] > c["end"]["as_low"]["lnN"]
                     and c["end"]["as"]["lnN"] > c["end"]["as_sasp"]["lnN"]),
          "KCC10 is conditional on a neighbour response to the secreted SASP. "
          "The raw ODE for lnN carries a basal drain -d_N, so absolute "
          "control drifts as -d_N·t; the reported observable cancels that "
          "drain against a time-matched control, matching how an experiment "
          "reads fold-change. This check asserts the SASP-conditional "
          "pattern on that relative scale."),
]
