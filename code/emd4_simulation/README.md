# EMD4 mechanistic simulation — `KCC5 → EMD4 (KCC6) → KCC7, KCC10; KCC9 opposing`

A mechanistic simulation of the EMD4 chain, with arsenite as the exemplar
exposure. Built to the contract established by `emd1_simulation/`.

**What this is:** a hypothesis-testing model in *relative effect sizes*, whose
job is to decide which measurements can distinguish a tissue tipping point from
a steep dose-response — and which cannot.
**What it is not:** a quantitatively validated carcinogenicity-prediction model.
No parameter here is a physiological rate constant. Time is in nominal days and
dose is dimensionless, with `E = 1` the reference arsenite exposure.

> **Status.** All three tiers built, plus the 250-draw uncertainty layer and the
> publication figure. 13/13 checks pass.
>
> **Revised after the uncertainty pass.** Two claims made at nominal parameters
> did not survive the spread and have been rewritten below: the marker-falls-
> after-withdrawal clause of `V3` (44%) and the senolytic/neutralisation
> contrast of `V7` (56–64%). What survived is stated as surviving; what did not
> is stated as not.
>
> **Revised again after an audit of the reported quantities.** Three numbers
> this README quoted turned out to be artefacts of how they were measured
> rather than results, and one statistic did not compute what its name said:
>
> - the **SA-β-gal over-read** was reported on an absolute scale that includes
>   the assay's unexposed background — 8.0 of the 13.8 points at nominal
>   parameters. Net of the concurrent control the claim is *fragile* (84.4%),
>   not sign-robust;
> - **"escape is certain at `q_sec ≥ 0.50`"** was 5/5 replicates; at 40 it is
>   80% [65, 90], and the "spatial structure makes propagation harder"
>   conclusion is not resolved at this replicate count;
> - the **`k_p` ridge** ran off the top of its own profile grid, so both
>   quoted intervals ended at exactly 2× nominal by construction;
> - **`t50`** halved the *final* value rather than the peak, so on a decaying
>   arm it reported an onset 21 days early.
>
> Every one is fixed in code, re-measured, and carries a regression test. The
> conclusions that changed are marked below; `V12`'s conclusion got stronger,
> `V3`'s got weaker.

## Run

The separate [public-data workflow](public_data/README.md) adds measured RNA
fits, persistent alternatives, donor–recipient interventions and held-out
prediction checks. Run `python3 -m emd4_simulation.public_data complete` from the
parent directory; results go to `emd4_simulation/public_data_results/complete_report.md`.
It reports evidence gaps explicitly and does not alter the manuscript outputs.

```bash
python -m emd4_simulation.run                        # full run
python -m emd4_simulation.run --no-identifiability   # skip the k_p profile (slow)
python -m emd4_simulation.run --n-ensemble 0 --no-spatial --no-figure   # quick
```

Exit status is non-zero if any validation check fails. Output lands in
`figures/output/`: `figure2c_emd4_simulation.{pdf,svg,png,eps,tif}` (exactly
183.00 mm wide) and `emd4_simulation_summary.json`.

> **The skip flags are for development only.** A partial run still writes a
> well-formed `emd4_simulation_summary.json` — one in which `V12` has silently
> failed and the identifiability numbers are `NaN`. Because `build_docx.sh`
> does not rerun the simulation unless given `--full`, such a summary would
> otherwise flow straight into the manuscript. `manuscript_tables.py` audits the
> summary and **refuses to write the manuscript tables** from a partial or
> failed run (override with `--allow-incomplete`, never for a manuscript).

## The load-bearing design decision

**`S` is a latent multiparameter state, never a marker.** Every observable
reaches it through an observation model with its own sensitivity, its own
*exposure-driven* false-positive rate, and its own onset delay:

| marker | sens | base FP | stress FP | lag | reads what? |
|---|---|---|---|---|---|
| p16INK4a | 0.85 | 0.03 | 0.10 | 3 d | the latent state |
| p21 | 0.80 | 0.06 | 0.55 | 1 d | state + acute stress |
| lamin B1 loss | 0.90 | 0.05 | 0.05 | 5 d | the latent state |
| **SA-β-gal** | 0.95 | 0.08 | **0.90** | 2 d | **mostly stress** |
| **DDR foci** | 0.75 | 0.04 | **1.20** | 0.5 d | **mostly acute stress** |
| SASP protein | 0.90 | 0.02 | 0.05 | 8 d | the state, late |

The *stress FP* column is the EMD4 construct-validity argument. The *base FP*
column is the one that has to be reported next to it, and used not to be.

In the low-dose arm at day 14, SA-β-gal reads 0.190 while the latent senescent
fraction is 0.052 — an over-read of 13.8 percentage points. **8.0 of those 13.8
points are the assay's positivity in unexposed tissue**: the concurrent control
reads 0.080 against a latent state of 0.000 at the same instant. The
exposure-attributable over-read is **5.8 points**, not 13.8.

The same correction changes the ensemble verdict. In the sampled ensemble, the
absolute-scale over-read is a median 12.2 points, 5th–95th [8.0, 28.3]. Those
positive sampled values include the unsampled `m_sabg.fp = 0.08` background,
but that background is not a universal positive lower bound: at zero stress and
marker steady state, `y - S = 0.08 - 0.13 S`, which is negative above
`S = 0.615`.
Netting the concurrent control off — the correction already applied to the KCC10
observable, see `apply_control_relative_lnN` — leaves a median **+3.5 points,
[−1.7, +15.9], positive in 84.4% of draws**. By the same ladder used everywhere
else here (≥90% robust, ≥60% fragile) that is **fragile, not sign-robust**.

The other half of the original claim did not survive either. "The marker falls
after withdrawal while `S` persists" holds in only **44%** of draws. So the
defensible statement is weaker than the one this README used to make, and still
worth making: **SA-β-gal reads above the latent state during exposure in most of
the plausible parameter space, and that reading is part assay background and
part stress response in all of it.** A model fitted to SA-β-gal alone is fitting
part stress response, always — and an over-read quoted without a concurrent
control is partly just the assay. `V3` asserts both scales; the netted clause is
the load-bearing one.

## Structure

```
exposure E ─→ R (ROS, KCC5) ─→ S1 (primary senescence) ─→ C1 ─┐
                    ↑                                          ├→ P (SASP) ─┬→ secondary senescence S2
                    │                                   S2 ─→ C2 ─┘ (q_sec)  ├→ ΔlnN vs control  KCC10
                    │                                                        └→ (clearance γ_immune)  KCC7
                    └──────────────── transient; returns to baseline on withdrawal
                                              S ──k_esc──→ X  escape events, terminal (no outgoing edge)   KCC9 (opposing)

KCC10 is reported as **ΔlnN versus concurrent control**, not absolute `lnN`.
The neighbour ODE carries a basal drain `-d_N`, so raw control drifts as
`-d_N·t`; the relative observable cancels that drain (control → 0; reference
withdrawal → ≈ 8.29) and matches how an experiment reads fold-change.
```

Governing equations, in `model.py`:

- `dR/dt = k_E·E + r_basal − k_R·R` — upstream driver, KCC5
- `dS1/dt = k_S·stress(R)·(1−S) − γ·S1 − k_esc·S1` — direct induction
- `dS2/dt = k_p·P^n/(K_P^n+P^n)·(1−S) − γ·S2 − k_esc·S2` — paracrine induction
- `dC_i/dt = (S_i − C_i)/τ_C` — arrest → SASP competence delay
- `dP/dt = k_sasp·(C1 + q_sec·C2) − d_P·P` — secretion, primary vs secondary
- `dX/dt = k_esc·S` — cumulative escape EVENTS (may exceed 1; not a live-cell
  fraction). Terminal = no outgoing edge; it is still driven by `S`

### Four deliberate departures from the sketched formulation

1. **The paracrine loop runs through a secreted pool `P`, not through `S`.**
   This makes SASP neutralisation an intervention rather than a parameter edit,
   forces the arrest→competence delay to be represented, and makes the feedback
   depend on *secretion* rather than on marker positivity. `run.py` verifies the
   reduction by evaluating the **full 14-state `rhs` on the slow manifold** and
   comparing its `d(S1+S2)/dt` against the manuscript form: in the fast-`P`,
   no-delay, `q_sec = 1`, `k_esc = 0` limit they agree to **4.2e-17**. With
   escape left on, the residual is *exactly* the terminal drain `-k_esc*S`
   (4.1e-17), which is the only thing the reformulation adds.

   > **This check used to be a tautology.** Until it was rewritten it rebuilt
   > the reduced right-hand side from the same algebra as `manuscript_rhs` and
   > compared the two, reporting `0.00e+00` in every run — a number guaranteed
   > by construction, because `rhs` was never called. The README quoted it as
   > evidence. `tests/test_regressions.py::test_qss_check_actually_evaluates_the_ode`
   > now perturbs the ODE in a way the reference form cannot see and fails if
   > the residual does not move. Note also that `k_esc -> 0` is part of the
   > limit: the manuscript equation has no escape term, and the old check could
   > not have noticed that either.
2. **Senescent cells are split into primary and secondary,** secreting at
   different rates. This is ref 57's mechanism, and it is load-bearing — the
   bistability condition turns out to be a condition on `q_sec` (below).
3. **Clearance is split into immune and intrinsic,** so that the KCC7
   direction-sensitivity rule can be checked as a signed test (`V8`) rather than
   asserted editorially.
4. **Escape is an explicit TERMINAL flux — which is not the same as inert.**
   `X` is the KCC9 observable and nothing reads *from* it: it has no outgoing
   edge, so no downstream path carries senescence back into the model. But `X`
   is still driven *by* `S`. Since `dX/dt = k_esc·S`, we have
   `X(t) = k_esc·∫S`, and a larger accumulated burden gives a strictly larger
   `X` — the reference arm reaches `X = 0.540` against `0.004` at low dose and
   `0` in control. Earlier text here claimed that "raising `S` can never raise
   the immortalisation readout through any path in the model"; that contradicts
   the equation and has been withdrawn. What `V9` actually tests is the
   different proposition that raising the *escape rate* `k_esc` raises `X`
   while **lowering** `S`.

   `X` is also not a measurement of immortalisation. It counts cumulative
   escape events normalised to the model population, so `X > 1` is possible; it
   is not a live-cell fraction. The model contains no telomere maintenance, no
   indefinite proliferative capacity and no immortalisation assay. KCC9's
   evidentiary rule — that exposure-induced senescence must not be scored as
   evidence of immortalisation — is an ontology commitment imposed on the
   model, not a result derived from it.

## Equilibrium structure

At zero exposure the **nominal** parameter set is bistable (two stable
equilibria separated by an unstable one). Across the ensemble this holds in
68.0% of draws — see [Uncertainty](#uncertainty), and do not quote the
retention count in its place:

| state | S | stability | recovery time |
|---|---|---|---|
| senescence-free | 0.0000 | stable | 37.7 d |
| separatrix | 0.1804 | unstable | — |
| persistent | 0.9240 | stable | 37.7 d |

- **Quasi-static fold** `E* = 0.0702`
- **Hysteresis loop area** 0.0641 — the up-sweep jumps at `E*`, the down-sweep
  never returns. This is an *area* between the two sweeps (senescent fraction ×
  dose), not a width in `E`; the name `width` survives in the code for
  compatibility. The integrand jumps at the fold, so the trapezoid needs a fine
  grid: `n = 16` reported 0.0678, 5% high against the converged 0.0641 and the
  analytic estimate `S_upper × E* = 0.0648`. `run.py` now uses `n = 121`
- **Normal form**: gap² is linear in `(E* − E)` with r² = **0.9888** — the
  textbook saddle-node signature
- **Critical slowing**: recovery time rises from 37.7 d far from the fold to
  1009.8 d at `E* − E = 2.1e-5` — a factor of **26.8** over the probed interval.
  Inside the asymptotic regime (`E* − E < 10⁻²`) the log–log slope is **−0.509**
  against the textbook `(E* − E)^(−1/2)`. Over the *full* probed interval the
  apparent slope is −0.428, because the two farthest points sit outside that
  regime, where relaxation is set by the linear rate rather than by the fold —
  so the ratio across the whole sweep is not itself a √ relationship

### The model containment threshold

**`q_sec* = 0.243` at the nominal settings.** Below that ratio the reference
withdrawal arm resolves and the second attractor is absent. Nominal `q_sec` is
0.50, so this parameter set lies above the lower fold. The threshold also
depends on receptor sensitivity, SASP clearance, paracrine gain, cooperativity
and senescent-cell removal. A measured protein-output ratio alone therefore
cannot locate a biological system relative to the threshold and need not equal
the ratio of functional senescence-inducing activity. Checked as `V10`; the
`as_qsec_lo` arm sets `q_sec = 0.20` and senescence resolves to 0.002.

## The null comparison — the headline

> **Read this first.** Every fitting target in this section is a trajectory
> **generated by this same mechanistic model**. No published time series or
> marker panel is fitted anywhere in this build. Rejecting these nulls
> therefore shows that *these particular* no-feedback models cannot reproduce
> *these synthetic* trajectories — it does not show that feedback is identified
> in biological data. Both nulls are also constructed to relax to zero once
> exposure stops, so their failure to retain senescence rejects reversible
> relaxation specifically, not every mechanism lacking a paracrine loop.
> Cell-intrinsic irreversible arrest and slowly-resolving damage are untested
> alternatives. As a structural counterexample, setting `k_p = 0` with
> senescent-cell loss and escape also zero leaves `S = 0.4954` at day 400 after
> withdrawal in this very ODE: persistence alone does not require the loop.

A no-feedback model, fitted just as hard, is given every chance:

| data given to the null | simple null (4 par) | cascade null (5 par) |
|---|---|---|
| forward **endpoint** dose-response, day 60 | **R² = 0.996** | — |
| full continuous-exposure **time course** (12 doses × 121 points) | R² = 0.884, worst dev 0.54 | R² = 0.902, worst dev 0.53 |
| **withdrawal** at day 14, S at day 400 | **0.000** (mechanistic: **0.924**) | **0.000** |

The cascade null exists so the simple null cannot be dismissed for the wrong
reason: `E → U → S` produces a sigmoidal rise in time with no self-amplification
at all. It still fails.

> **The forward endpoint dose-response — the experiment almost every published
> study runs — cannot distinguish a tissue tipping point from a steep sigmoid.**
> A no-feedback model reproduces it at R² = 0.996 with a Hill coefficient of **17.9**.

### What revised the plan

The build plan predicted that *only* withdrawal and recovery-time data could
separate the two. The data-confrontation pass says otherwise, in the direction
that helps:

| planned claim | what the model actually shows |
|---|---|
| only withdrawal separates the models | a **densely sampled time course at a single dose already separates them** (R² 0.88–0.90, worst deviation 0.53) — no monotone relaxation reproduces an autocatalytic rise. This is a much cheaper experiment than a withdrawal arm. |
| the tipping point is a dose threshold | it is a **dose × duration** condition. At `E = 1.0` exposure must exceed **2.19 d** to persist; at a 14-day exposure the threshold dose is **`E = 0.209`**. Neither equals the quasi-static fold (0.070), which is therefore not the threshold any finite experiment measures. Both are computed by `ensemble.threshold_duration` / `threshold_dose` and land in the summary JSON as `finite_threshold`. |
| persistence follows from the senescent fraction at withdrawal | it does **not**. `S = 0.086` at withdrawal resolves while `S = 0.100` persists, because the basin boundary lives in the full state — including the SASP pool and the competence lag — and not in `S` alone. Reporting `S` at withdrawal is insufficient to predict persistence. |

## Identifiability

Profiling `k_p` against a 2-percentage-point noise floor:

| fit target | admissible `k_p` | ridge | is `k_p = 0` admissible? |
|---|---|---|---|
| continuous exposure, one dose | 0.208 – 0.743 | **3.57-fold** | **No** |
| withdrawal-only design | 0.103 – 14.46 | **141-fold** | **No** |
| continuous + withdrawal, each arm within tolerance | 0.208 – 0.743 | **3.57-fold** | **No** |

> **Both rows used to be grid artefacts.** The profile ran on
> `linspace(0, 2*k_p, 21)`, so the largest `k_p` it could ever report was 0.700
> — and the fit there was still at rmse 0.0175, inside the 0.020 noise floor.
> `admissible.max()` sat on the grid edge in every run, and the published
> interval "0.210 – 0.700 / 140%" had an upper endpoint set by the choice of
> grid rather than by the data. The grid is now log-spaced out to 128× nominal
> and stops being admissible on its own; `profile_kp` returns
> `kp_ridge_censored`, and `manuscript_tables.audit` refuses to write tables
> when it is True.

> **The "withdrawal widens the ridge" claim was wrong, and has been withdrawn.**
> `withdrawal_breaks_degeneracy` never added the withdrawal arm to anything: it
> built a fresh 81-point, 0–400 d target and dropped the 61-point
> continuous-exposure residuals entirely, so the comparison was
> continuous-only *versus* withdrawal-only — two different experiments, not one
> plus more data. It is now named `profile_kp_withdrawal_only`, and
> `profile_kp_joint` does the real test. Fitting both arms in one objective and
> requiring **each** arm to sit inside the noise floor returns exactly the
> continuous-only admissible set (3.57-fold, `k_p` in 0.208–0.743): the
> withdrawal arm neither narrows the ridge nor widens it. The old 140.82-fold
> figure came from pooling into a single **average** RMSE, which is not
> invariant to the number of observations — appending a long easy record drags
> the mean down and can admit a `k_p` whose continuous-exposure fit is plainly
> bad (at `k_p = 5`: withdrawal-only 0.0144, passes; its continuous-arm
> residual 0.0421; pooled joint 0.0267). The qualitative conclusion survives;
> the number and the "dilution" explanation do not.

Two conclusions follow within this restricted synthetic profile:

- **Zero feedback is rejected for the model-generated continuous record.** This
  is conditional on the generating model, fixed parameters, selected nuisance
  refits and the assumed two-percentage-point tolerance. It does not establish
  feedback in biological data or the evidentiary sufficiency of an EMD4
  annotation.
- **Feedback strength remains poorly constrained.** Values of `k_p` spanning
  3.57-fold fit the single-dose continuous record within tolerance. The separate
  withdrawal-only design admits a 141-fold range, but adding that arm to the
  continuous record and requiring each arm to pass leaves the 3.57-fold range
  unchanged. A fitted paracrine rate is not a measured biological quantity.

## Interventions

| arm | S at day 400 | reading |
|---|---|---|
| arsenite, withdrawn d14 | 0.924 | the persistent state |
| + senolytic from d60 (6× clearance) | 0.618 | **removing cells does not collapse it** |
| + SASP neutralisation from d60 (0.25×) | 0.000 | **breaking the feedback does** |
| + impaired immune clearance (**from d0**) | 0.966 | KCC7-positive direction |
| + enhanced immune clearance (**from d0**) | 0.822 | *not* a KCC7-positive observation |
| + poor secondary secretion (`q_sec` 0.20, **from d0**) | 0.002 | below the containment threshold |

**Only the senolytic and SASP-neutralisation arms are timed to day 60.** Those
two are clearance experiments on an established state, which is what the
published ablation designs are. The immune, `q_sec` and escape arms use the
default `t_start = 0`: they are co-exposure or baseline-modification
conditions, a different experiment, and describing them as interventions begun
on day 60 would misstate the protocol.

The senolytic/neutralisation contrast is the model's sharpest prediction at
nominal parameters: a 6-fold increase in clearance leaves two thirds of the
burden standing, because the survivors re-seed the loop, while a 4-fold
reduction in SASP activity collapses it entirely.

**It is a soft prediction, not a robust one.** Across 250 draws, SASP
neutralisation collapses the state in 63.6% and the senolytic leaves >0.20
standing in 56.4%. Restricted to the 147 draws where the comparison is not
degenerate — where the senolytic has not already collapsed the state, making it a
comparison of zero with zero — neutralisation beats the senolytic in 67%. So the
direction is right more often than not, and that is the honest ceiling on it.

**These are predictions awaiting data, not retrospective agreement** — no
senolytic or SASP-neutralisation arm exists for this cell system. Alimirah 2020
(p16-positive ablation in two-stage skin carcinogenesis) is the closest
published analogue.

## Validation

Thirteen checks (`conditions.py`), all passing — including `V13`, which asserts
the KCC10 pattern on **ΔlnN vs concurrent control**. Four conditions are
plotted; the rest are held out of the figure. The honest caveat about "held out" here: EMD1 could
hold out whole *published* experiments, whereas EMD4's senolytic,
SASP-neutralisation and immune-clearance arms have no published counterpart in
this system. They are labelled predictions in the battery and must not be
reported as agreement with measurement. The withdrawal arm (`V2`) is the one
that is genuinely anchored, to Okamura 2024.

These are **internal consistency checks, not a preregistered battery and not
independent experimental validation.** Several were narrowed after seeing the
uncertainty results — `V3` twice, as its own basis text records. That is good
practice for keeping a check honest, but it means the set must not be described
as "predefined validation". Calling them predefined would claim an
independence the procedure does not have.

The Okamura 2024 anchor also supports a narrower claim than the protocol here.
That study used 5 µM arsenite for 72 h and assessed withdrawal over 72 h, 100 h
or seven days depending on the assay; its persistent-SASP evidence is gene
expression, and its MMP3 staining is intracellular. That is short-term
persistence of senescence-associated *features* — not a measured functional
extracellular feedback loop, and not a 400-day state. Day 14 and day 400 are
chosen simulation times, not that study's schedule. The 2022 companion
(conditioned medium from arsenite-exposed LX-2 increasing Huh-7 migration) is a
genuine functional neighbour observation, but it does not measure the
neighbour-proliferation rate, secondary-senescence feedback or `q_sec` this
model uses.

## Tier 3: spatial propagation

A Gillespie simulation on a hex lattice — ligand release, exponential paracrine
kernel plus a juxtacrine term at contact range, a continuous ligand field
converted to an induction hazard through a soft threshold, a delay between
induction and SASP competence, clearance with replacement. Written from the
published description of Martin, Schumacher & Chandra (2023) rather than
vendored, so the assumptions stay visible.

> This README used to describe the binding step as "Poisson-approximated",
> which the code has never done: Martin et al. draw discrete binding counts,
> this implementation carries the mean field and makes only the induction and
> clearance *events* stochastic. `spatial.py`'s docstring was corrected earlier;
> this line was not. Single lesions here are therefore less variable at low
> ligand than the source model would make them.

**Reaching the edge** is membership of the lattice's outer ring, not a radius
threshold. The hex embedding is anisotropic — row spacing is √3/2 of column
spacing — so the previous criterion (`radius > 0.45·min(nx, ny)` = 18.45 on a
41×41 lattice, against a half-height of 17.32) scored a lesion that ran to the
top and bottom edges as contained.

**Escape is "the front reached the boundary at any time", not "the boundary is
occupied at day 300".** `reached_edge` is an end-of-run snapshot and would score
a lesion that touched the edge and was then cleared as contained;
`ever_reached_edge` accumulates the event as it happens and is what the scan now
reports. On the nominal 41×41, 40-replicate scan the two criteria agree at every
`q_sec`, so no count changes — but the definition now matches the claim. The
scan also reports `frac_extinct` separately, because "contained" bundles two
different outcomes: the lesion died out, or it survived as a bounded cluster.

**The lattice does not independently validate the mean-field threshold.** The
ODE uses a Hill response, an explicit secreted pool and a competence lag; the
lattice uses an exponential distance kernel, threshold-linear induction, a
deterministic maturation delay and its own clearance rate. There is no
demonstrated reduction or parameter mapping making the two thresholds
numerically equivalent, and Martin et al. distinguish paracrine from juxtacrine
secondary cells where this implementation merges them. That `q*` lands inside
the spatial transition band is a consistency observation for the chosen
settings, not a replication of that mechanism.

**Zero escapes is not zero risk.** 0/40 at `q_sec ≤ 0.15` gives a Wilson upper
bound of 8.8% over this follow-up, on this lattice size and seed size.

Two implementation notes that mattered. The ligand field is maintained
**incrementally** — it changes only when one cell starts or stops secreting, and
then only by that cell's kernel column — which took a single run from 107 s to
0.4 s. And **cleared cells are replaced by healthy ones**; without replacement
the lattice depletes and every lesion "resolves" for the trivial reason that the
tissue is gone.

**40 replicates per value**, on a 41×41 lattice:

| `q_sec` | escaping | 95% CI (Wilson) | bounded lesions |
|---|---|---|---|
| 0.05–0.15 | 0% | [0%, 9%] | ≤10% |
| 0.20 | 20% | [10%, 35%] | 0% |
| 0.25 | 42% | [29%, 58%] | 8% |
| 0.30 | 57% | [42%, 71%] | 0% |
| 0.35 | 62% | [47%, 76%] | 0% |
| 0.40 | 75% | [60%, 86%] | 0% |
| 0.50 | 80% | [65%, 90%] | 0% |
| 0.65 | 88% | [74%, 95%] | 0% |

The mean-field threshold `q_sec* = 0.243` falls inside the band between
"every replicate contained" (`q_sec ≤ 0.15`) and "the majority escape"
(`q_sec = 0.30`), which is what `V10` checks.

> **A mean-field tipping point is not a sharp threshold in spatially discrete
> tissue — it is a probability.** Between `q_sec` 0.15 and 0.65 the same
> parameters give containment in some replicates and escape in others,
> depending on the fate of the first few induction events.

**Two claims this table used to make, and no longer does.**

> **"Escape is certain at `q_sec ≥ 0.50`" was five coin flips.** The scan ran at
> `n_rep = 5`, where the escape fraction is quantised to steps of 0.2 and 5/5 is
> an unremarkable outcome for a probability of 0.8. At 40 replicates `q_sec =
> 0.50` escapes in 80% [65, 90], and *no* `q_sec` tested reaches certainty. The
> band edge `q_escape` is gone; `q_majority_escape` replaces it, and both edges
> are `None` rather than 0.0/1.0 when undetermined — the old fallbacks silently
> degraded `V10` to `0 ≤ q_crit ≤ 1`, which nothing can fail.
>
> **"Spatial structure makes propagation harder than well-mixed" is not
> resolved by this scan.** The 50%-escape crossing is at `q_sec = 0.275`, above
> the mean-field 0.243 — but by 0.032, against Wilson intervals ~0.15 wide on
> either side. At `n_rep = 5` the crossing read 0.35 and the gap looked like
> 0.107. The direction may well be right, and the ligand-dilution argument for
> it is sound; this measurement does not establish it.

**What this tier does and does not measure.** At this lattice size the outcome
is near-binary: a seeded lesion either dies out (0 cells) or grows to the
mean-field upper branch and fills the lattice (~1450 of 1681 cells, out to the
corner). Bounded lesions — neither extinct nor saturating — appear in at most
10% of replicates, and in 0% at every `q_sec` above 0.25. So *cluster size* and
*radius* are not independent readouts: both equal the saturation value times the
escape probability (radius = 26.458 × escape fraction, r = 1.000). The old
three-column table reported one number three times, and those columns are now
labelled diagnostics in the summary JSON and dropped from the manuscript tables.

**This tier measures extinction probability, not propagation extent.** That is
weaker than reproducing ref 57's finite-propagation result, which requires
observing a lesion that stops while the tissue around it is still healthy — and
this lattice is smaller than the lesion.

There is one size number that does mean something, and it is now reported in
its place: the **occupied fraction of surviving lesions**, which is comparable
against the Tier-1 upper branch and was not previously checked against it.

| `q_sec` | spatial, surviving lesions | mean-field upper branch |
|---|---|---|
| 0.35 | 0.868 | 0.905 |
| 0.50 | 0.887 | 0.924 |

The two agree to within ~4%, with the spatial value below the mean-field one as
it should be: the lattice has an edge, and cells in the induced-but-not-yet-
secreting state are not counted. This is the only quantitative agreement between
the two tiers besides `q_sec*`, and it is worth more than the three collinear
columns it replaces.

This matters for how the saddle-node in Tier 1 should be read. It is a
statement about the average behaviour of a well-mixed compartment, and the
tissue-level question it stands in for has a stochastic answer.

## Uncertainty

Nothing here is measured in the exemplar system, so the free effect sizes are
log-normally uncertain (σ = 0.20–0.50, wider than EMD1's 0.15–0.35 precisely
because EMD1 could anchor its magnitudes on published measurements and this
build cannot). Structural choices are not sampled: the loop exists, secondary
cells secrete less than primary ones, and escape is terminal.

Sampling moves draws *across* the bifurcation, which is the point — it converts
the assertion in `V6` into a measurement:

| | 250 draws | verdict |
|---|---|---|
| high burden retained at day 600 (**retention, not bistability**) | **94.0%** | weakest of the three; robust |
| **two stable equilibria at `E = 0`** (**bistability**) | **68.0%** | ordinary bistability — **not robust** |
| — empty state **and** persistent state (strictest) | **46.0%** | **not robust** |
| **senescence-free state LOST** (`R* > R_ref`) | **48.8%** | basal ROS alone commits cells |
| — of which **still bistable anyway** | **22.0%** | refutes "no empty state ⇒ one attractor" |
| **critical `q_sec*`** | median, see summary | conditional on the **198/250** draws with a detected crossing |
| — of those, a bistable **window** not a half-line | **19/250** | above the upper fold the LOW state is gone: monostable **high** |
| threshold dose, 14-day exposure | median 0.197, [0.000, 0.755] | order-of-magnitude only; 235/250 draws |
| reference dose persists after withdrawal (unpaired) | 92.4% | descriptive, **not attributable** |
| — concurrent control also persistent | 25.6% | not an exposure effect |
| — **persistent and matched control below threshold** | **66.8%** | fragile — the exposure claim |
| SA-β-gal over-read during exposure, **absolute** | +12.2 pts, [8.0, 28.3] | includes assay background |
| — of which unexposed assay background | +8.0 pts, [8.0, 17.5] | not an exposure effect |
| SA-β-gal over-read, **net of concurrent control** | +3.5 pts, [−1.7, +15.9] | **84.4% — fragile** |
| low dose stays below persistence (`S < 0.30`) | 61.2% | fragile |
| — low dose near baseline (`S < 0.05`) | 59.2% | **not robust** |
| SASP neutralisation collapses the state | 63.6% | fragile |
| marker falls while latent state persists | 44.4% | **not robust** |
| senolytic leaves >0.20 standing | 56.4% | **not robust** |

### The containment threshold is a window edge, and no scan over `q_sec` can find it

`q_sec_critical` used to gate on the two endpoints of the search range —
returning "no threshold" unless the draw was monostable at `q_sec = 0.01` and
bistable at `q_sec = 0.99`. Bistability is **not monotone in `q_sec`**: in
**19/250** draws it appears as secondary secretion rises and then disappears
again, so those draws are monostable at *both* ends and the gate dropped them.
The reported denominator was 179/250; it is now **198/250**.

**Scanning cannot fix this, at any affordable density.** The narrowest window in
the ensemble is draw 193's, `q_sec ∈ (0.104977, 0.105546)` — **5.7 × 10⁻⁴**
wide. An 81-point scan steps over it; so does a 401-point scan. An intermediate
version of this code scanned a scalar "distance to fold" built from the local
*maximum* of `F`, concluded draw 193 was monostable everywhere, and wrote that
into the manuscript. It was wrong twice over: the grid could not see the window,
and three equilibria require the local maximum above zero **and** the local
minimum below it, so tracking only the first gives the wrong answer above the
upper fold.

The folds are now read off an **analytic equilibrium curve with numerically
located extrema**. `F` increases monotonically with `q_sec`, so `q*(S)` can be
written in closed form by inverting the Hill function
(`bifurcation.fold_curve`), and the saddle-nodes are exactly its interior local
extrema. This removes the search over `q_sec` entirely and resolves windows far
narrower than any affordable scan. Cross-validated against the independent
eigenvalue test at interval midpoints, just below onset, just above the upper
edge, and a 25-point sweep for every draw reported monostable: **0 failures**.
An independent derivative-based fold calculation agrees on all 198 thresholds
to within **1.3 × 10⁻¹¹**.

It is **not** an unconditional guarantee, and earlier wording here ("computed
analytically", "a window of any width is resolved") overstated it. The extrema
are located on a finite `S` grid and resolution still matters:

| S grid points | draws with a threshold |
|---:|---:|
| 2,001 | 180 |
| 20,001 | 195 |
| 200,001 | **198** |
| 1,000,001 | 198 |

### The two-crossing cutoff is a calibration, not a derivation

`q*(S)` is obtained by dividing through by `S₂ = S − a₀(1−S)/g`, which vanishes
at **`S = a₀/(a₀+g)`** — not `a₀/g`, as an earlier version of this section
said. Near that asymptote `q*` diverges, so on a uniform grid the lowest
equilibrium is often unresolved, sitting between grid points where the curve is
nearly vertical. Draw 23's low state (3.844 × 10⁻⁴) behaves that way.

The claim that the low state is *never* a crossing and exists at every `q_sec`
was also wrong: for `a₀ > 0` it generally **is** a crossing given enough
resolution — draw 193 has three crossings at `q_sec = 0.1052661` and one at
`q_sec = 0.99` — and the exactly empty state at `a₀ = 0` is a separate
equilibrium the positive branch omits by construction.

So requiring **two** crossings is an empirical calibration that compensates for
an unresolved low branch, validated against the independent calculation rather
than derived. Raising the cutoff to three *without* clustering the grid near
the asymptote makes things worse (174–177 of 250), because it removes the
compensation and not the cause. A principled version needs a grid clustered at
`a₀/(a₀+g)` or a reparametrisation; until then, re-validate the cutoff whenever
the sampler or the `S` grid changes.

### Losing bistability is ambiguous — say which branch died

Above the upper fold of a window it is the **low** branch that disappears, so
the system is monostable at a **high** burden with no healthy state left: draw
193 sits at `S = 0.812` at `q_sec = 0.99`. That is the worst case, not a return
to containment. Below the lower fold the surviving state is the low one, which
*is* containment. An earlier version of this README and of the main text said
"above `q_sec*` it tips" is false above the upper edge — exactly backwards.

### Three different quantities, none of them interchangeable

`frac_high_state_retained` (94.0%) counts draws where a high initial burden is
*still high* at the day-600 probe. That is a finite-time **retention** test, not
a count of stable states, and near a fold it can score a slow transient as an
attractor. It was previously called `frac_bistable` and was tabulated in the
manuscript as "Bistable at `E = 0`, robust". It is not a bistability count and
that row has been corrected.

`frac_two_stable_equilibria` (68.0%) is ordinary bistability, obtained by
reducing the zero-exposure system to a scalar map, bracketing **every** sign
change, and classifying each root with the full Jacobian
(`bifurcation.equilibria_at_zero`). Under the 90% convention this build uses for
"robust", 68.0% is not robust.

`frac_empty_plus_persistent` (46.0%) additionally requires the low state to be
exactly empty. It does not
check that the *senescence-free* state still exists — and under this prior it
often does not. `S = 0` is an equilibrium at zero exposure only when the
paracrine term is o(S) at the origin (`n_P > 1`, enforced) **and** basal
oxidative stress is below the commitment threshold, `R* = r_basal/k_R <= R_ref`.
At nominal parameters `R* = 0.30 = R_ref` exactly — the condition holds on its
boundary. The ensemble log-samples `r_basal`, `k_R` and `R_ref` independently,
so it does not hold generally.

**In 48.8% of draws there is no empty state at all.** Requiring both
states leaves **46.0%**.

**Losing the empty state does not leave one attractor.** An earlier version of
this README and of Note S11 said it did. That is false: a small *positive*
low-senescence equilibrium can coexist with a high one, and **22.0% of draws
(55/250) are bistable in exactly that way** — one of them carries a stable low
state at `S = 0.00038`, an unstable separatrix at `S = 0.449` and a stable high
state at `S = 0.731`, with no empty state anywhere. The
gap between 68.0% and 46.0% is a difference of *definition*, not magnitude
uncertainty about the loop. The `[0.000, ...]` lower bound on the threshold dose is the same
phenomenon seen from another angle, not an independent finding.

A prior that sampled the *ratio* `R*/R_ref` rather than its three constituents
would separate the two questions cleanly. That is the obvious next revision; it
is not applied here, because choosing a prior after seeing which result it
produces is how the original problem was created.

The pattern is consistent and worth stating plainly: **the existence claims
survive and the contrast claims do not.** That senescence persists is robust;
*which* dose separates persisting from resolving is not, because the threshold
dose spans [0.000, 0.755] and the chosen low dose (0.15) sits inside that
interval. An experiment aiming to demonstrate the contrast would need to
establish its own threshold first rather than borrow one.

The lower bound of 0.000 on the threshold dose is not a numerical artefact: in
those draws, basal oxidative stress alone commits enough cells to cross the
separatrix, and senescence becomes spontaneous with no exposure at all.

## Limits

- **Tier 3 resolves neither front velocity nor front extent.** At 41×41 a
  lesion either dies out or saturates the lattice, so the layer measures the
  probability of extinction and nothing about how far a surviving lesion
  travels. `V10` checks the mean-field threshold against the band between full
  containment and majority escape, which is a real test; front velocity, front
  extent, and a clean separation of the paracrine and juxtacrine contributions
  all need a lattice larger than the lesion. This is weaker than reproducing
  ref 57's finite-propagation result in full.
- **The spatial scan is a 40-replicate binomial estimate.** Escape
  probabilities carry Wilson intervals ~0.15 wide near the transition, which is
  not enough to resolve the offset between the 50%-escape crossing and the
  mean-field threshold. Distinguishing those needs a few hundred replicates.
- **The bifurcation numbers are nominal-parameter point estimates.** The fold,
  the hysteresis width, the normal-form fit and the critical-slowing ratio are
  quoted without intervals; only the endpoint and threshold quantities went
  through the ensemble.
- **The parameters are not fitted to data.** They are chosen to place the
  system in the regime the published qualitative observations describe
  (persistence after withdrawal, secondary senescence, resolvable low-dose
  response). The magnitudes are illustrative; the *structure* is the hypothesis.
- **Bistability is asserted by the parameter choice, not measured.** The
  manuscript is explicit that whether threshold behaviour occurs is an empirical
  question. What this build contributes is the demonstration of which
  measurements could settle it — and the `q_sec*` number that makes it a
  secretome experiment rather than a modelling argument.
- **The two tiers are not coupled.** Tier 1 and Tier 3 share the `q_sec`
  parameter and agree on two things — where propagation stops, and the occupied
  fraction of an established lesion (0.868 vs 0.905 at `q_sec` 0.35; 0.887 vs
  0.924 at 0.50) — but the spatial model does not feed back into the
  compartmental one, and neither carries the other's marker observation layer.
- **Tier 1 is one compartment, one cell type.** No tissue architecture, no
  immune population dynamics, no proliferating compartment competing with the
  senescent one.
- **The spatial lattice is small.** 61×61 with a dense N×N kernel is the
  practical ceiling for the precomputed-kernel approach (~110 MB); the
  containment scan runs at 41×41. Larger tissues need an FFT convolution.
- The 400-day horizon and the day-14 withdrawal are modelling choices matched to
  the Okamura protocol's shape, not to a measured exposure protocol.
