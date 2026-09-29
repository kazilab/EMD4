> Updated public-data version: [complete report](complete_report.md). The original submitter GSE222400 files remain excluded; a separate NCBI count matrix is now analysed.

# EMD4 public-data analysis

**Conclusion: this run does not establish EMD4, functional SASP feedback, or bistability.** It adds measured RNA checks, persistent alternatives, held-out predictions and testable intervention predictions. The original simulation and manuscript outputs are unchanged.

## Public-data quality and scope

- [GSE144397](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144397): 36 samples, nine predefined genes, analysed separately as RAS, RAF and quiescence trajectories. Submitter-normalised log2 expression is retained. Induction continues throughout observation; there is no withdrawal experiment in these selected arms.
- [GSE210140](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210140): 20 samples, nine genes, log2(TPM+1). Direct DOX is compared with its control; recipient DOX-conditioned medium is compared with control-conditioned medium. Five samples per group; replicate numbers are not assumed to establish pairing.
- [GSE222400](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE222400): **excluded**. All 52 deposited files contain differential-expression tables rather than individual-sample expression, with 25 distinct uncompressed payloads. `baseMean` is never treated as a sample measurement. Reanalysis of raw reads or a corrected deposit is needed for this withdrawal series.
- [Okamura arsenite study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701098/): full article and figures cached with provenance. Huh-7 exposure is 72 hours; withdrawal and reseeding are separate protocol clocks. Figures are retained for review; no numerical values or individual replicates have been invented from their images.

## Held-out RNA prediction

Each culture context is fitted separately. Both final-time replicates are excluded from all kinetic fitting, observation coefficients and noise covariance estimation. The deposited RMA/batch normalisation predates this split and uses the source collection; this is within-study forecasting, not independent external validation. Replicate trajectories can be correlated across time, so no independent-sample hypothesis test is claimed.

| Context | Held-out day | Lowest predictive error | Whitened MSE | Feedback MSE | Last-observation MSE |
|---|---:|---|---:|---:|---:|
| RASOIS | 6 | feedback / lag_memory | 12.645 | 12.645 | 13.883 |
| QUIESCENCE | 4 | slow_damage / matched_clearance | 9.875 | 10.736 | 4.661 |
| RAFOIS | 4 | feedback / matched_clearance | 37.346 | 37.346 | 42.398 |

Smaller error means better prediction of these selected RNA channels. It does not identify a mechanism. Last-observation and linear-time forecasts provide simple training-only benchmarks (both in JSON). Large errors relative to estimated replicate variability and parameter-bound hits are evidence of poor prediction or weak identification, even when a candidate ranks first. Durable arrest and persistent SASP without feedback are **observationally equivalent in this RNA observation model**, because secreted protein is not measured. Rankings between tied candidates are arbitrary. Failed optimisations are excluded from ranking and retained with status in JSON. Parameter bounds, observation rank and all fits are retained.

The feedback profile refits nuisance kinetics and observation coefficients at each gain. It is a finite-grid sensitivity analysis, not a confidence interval. Baseline occupancy (0, 0.02, 0.10), maturation/clearance structure and diagonal versus correlated observation noise are also examined. No RNA-derived estimate of `q_sec_critical` is produced: functional fraction and receptor-response scale are confounded.

![Observed RNA fits](rna_timecourse.png)

## Conditioned-medium evidence

The measured result is a transcriptomic contrast after 72 hours of 50% conditioned medium. Endpoint RNA alone does not show recipient senescence, self-sustaining recipient secretion, SASP protein potency, or exclusion of residual toxicant. Intervals are descriptive Welch 95% intervals, without multiplicity correction; no significance screening or post hoc marker selection is used. A channel with zero observed variance receives no interval: all-zero TPM is not evidence of a precisely zero biological effect.

![Conditioned-medium RNA](conditioned_medium_RNA.png)

## Virtual interventions and structural checks

The donor model uses the public 2-hour exposure, wash and 72-hour conditioning schedule. Only extracellular protein transfers. Exposure carryover is an explicit separate control. Recipient simulations include medium washout, receptor inhibition, partial-block ligand add-back, continuous donor input and donor removal. A one-way paracrine alternative induces recipient arrest while secondary cells cannot secrete active factors; persistence of that arrest does not imply a feedback loop. The latter extensions and 14-day outcomes are predictions, with assumed rates and equal donor cell number/medium volume. Complete receptor inhibition cannot be rescued by more ligand in this model.

Two secretion structures are compared: a phenomenological lag that retains memory after arrest clearance, and an immature/mature model in which secreting cells undergo the same clearance as arrested cells. Baseline and parameter sweeps expose assumption dependence. Protein abundance and functional activity remain separate.

![Virtual interventions](virtual_transfer.png)

## Model-confusion benchmark

25 synthetic trials per generating mechanism, seed 723. Feedback was selected in 36 of 125 nonfeedback trials with a converged winner. The conditional Monte Carlo Wilson interval is 0.216–0.373. This is not an empirical false-positive rate or the probability that EMD4 is true.

Generating conditions vary persistence, induction, baseline and maturation structure. Every candidate fits noisy observables and re-estimates its measurement coefficients. An unseen final time assesses prediction. Misspecified structure and replicate-level offsets are included. Sparse trials and assumed generators limit interpretation; increase `--trials` for precision. No classifier cutoff is chosen using this benchmark.

![Model confusion](model_confusion.png)

## What this supports for the manuscript

- Supported as a modelling capability: persistent nonfeedback mechanisms are compared, public measured RNA is used, protocols are explicit, and feedback is challenged with held-out data and interventions.
- Supported by the imported data: the reported gene-level trajectories and conditioned-medium contrasts, in their own cell systems and protocols.
- Still unestablished: EMD4 in arsenite-exposed Huh-7 with a quantitatively calibrated functional SASP, necessity/sufficiency of feedback, bistability, tumor promotion, and a biological lower bound for `q_sec_critical`.
- Most informative next validation: the same donor/recipient system with proliferation plus multiple senescence assays, secreted protein, toxicant carryover controls, recipient washout, pathway inhibition and an appropriate rescue. The design-time ranking in JSON is conditional on assumed observable loadings, not a power calculation.

## Reproducibility

See `summary.json`, `observations.csv`, `endpoint_contrasts.csv`, `source_manifest.json`, `washout_archive_audit.json` and `runtime.json`. Every downloaded input is checksum-verified; modified or untracked cache files fail closed. Source code hashes, package versions, seeds and fitted sample IDs are recorded. `--offline` reproduces analysis without accessing the network. Original manuscript tables do not consume these new files.
