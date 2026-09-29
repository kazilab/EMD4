# EMD4: completed public-data version

The public-data implementation is complete for the sources and analyses specified below. **It does not establish a self-sustaining feedback loop, bistability or a biological value of `q_sec_critical`.** Direct functional calibration remains an evidence gap, not a parameter inferred from RNA or protein abundance.

## What changed in this version

- Recovered GSE222400 using NCBI-generated sample counts from the public sequencing reads. The original 52 submitter tables remain excluded because they contain differential-expression statistics. No local raw-read pipeline was needed.
- Added quantitative, traceable digitization of matching-system arsenite/Huh-7 figures, with correct assay clocks and reading uncertainty.
- Imported SASP Atlas protein measurements, cell-density normalization records and paired CITE-seq RNA/ADT measurements.
- Added an independent RNA study, with source predictions frozen before comparison and no target kinetic refitting.
- Retained six competing mechanisms, protocol-specific medium replacement, held-out prediction, structural sensitivity and the cross-generator benchmark.

## Recovered post-withdrawal RNA evidence

[GSE222400](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE222400) supplies 52 samples in the separate NCBI count matrix. Fifty enter the time-course analysis; the two replicative-senescence samples have no comparable experimental time origin and are excluded. The three exposure arms contain 16 samples each, plus two shared untreated controls. D0 is the end of a 24-hour induction, so D16 is model day 17. Medium replacements are represented at days 1, 5, 9, 13 and 17.

The [NCBI processing method](https://www.ncbi.nlm.nih.gov/geo/info/rnaseqcounts.html) uses HISAT2/featureCounts and can differ from the authors’ analysis. Exact GeneID mappings come from the pinned GPL17586 annotation. Median-of-ratios normalization learns its reference genes and geometric means from pre-holdout samples only. This measures relative expression, not RNA per cell. No longitudinal untreated controls are available.

| Pulse | Best mechanistic predictor | Whitened MSE | Feedback MSE | Last-observation MSE |
|---|---|---:|---:|---:|
| SDS | feedback / lag_memory | 10.94 | 10.94 | 11.08 |
| KCl | slow_damage / matched_clearance | 3.49 | 72.41 | 32.97 |
| DXR | feedback / matched_clearance | 6.79 | 6.79 | 5.37 |

SDS: feedback improves on the last-observation forecast by less than 5%. KCl: slow damage predicts best among the mechanistic candidates. KCl: the last-observation forecast outperforms feedback. DXR: the last-observation forecast outperforms feedback. These are descriptive comparisons on a small held-out sample, not significance tests.

Ranking alone is not mechanism identification. Large forecast errors, parameter-bound hits, observationally equivalent candidates and absent longitudinal controls limit interpretation. The final D16 pair is never used to fit kinetics, observation coefficients or noise covariance.

![Withdrawal RNA](withdrawal_DXR.png)

## Quantitative arsenite evidence

[Okamura et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701098/) provides aggregate figures rather than downloadable replicate tables. `arsenite_digitized.csv` records all 38 values, image hashes, axis anchors and point/bar coordinates. Audit overlays show every reading. ±2 pixels is a digitization sensitivity range, not an experimental confidence interval.

Approximate SA-β-gal positivity changes from 44.8% to 26.9% in treated cells, while controls change from 27.1% to 8.7%. The control-adjusted excess is approximately 17.7 versus 18.2 percentage points at withdrawal and seven days later. These point estimates illustrate why absolute marker decline should not be called recovery. Five microscopy fields are not treated as five independent cultures.

The withdrawal proliferation assay uses normalized luminescence after harvesting/reseeding. Constant and relaxing inhibition curves are fitted descriptively to the treated/control signal. The apparent relaxation rate is **not senescent-cell clearance**: metabolism, viability, control growth and shared normalization can contribute. Exposure PCR uses 18S, whereas withdrawal PCR uses RPLP1; those scales are not pooled into a decay rate. Near-zero control bars receive no fold-change estimate.

![Arsenite evidence](arsenite_public_evidence.png)

## Secreted proteins and single-cell measurements

The [SASP Atlas source tables](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000599) provide 6,420 reported protein comparison rows across eight contexts, including 24 rows for the predefined marker panel. Source workbook row/cell addresses and author q-values/SD are retained. No SD is converted to biological SEM and no replicate values are reconstructed. Missing proteins are not set to zero. Soluble and vesicle measurements, cell types and induction times remain separate.

The measurements test secretome composition and temporal/context dependence. They do not identify functional fraction, receptor potency or feedback gain. Intracellular-marker proteins found in medium (for example LMNB1) are not equated with their intracellular assays.

![Public secretome](public_secretome.png)

[GSE250041](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250041) pairs RNA and eight surface-antibody measurements in proliferating and irradiated WI-38 libraries. All RNA features contribute to library-size and mitochondrial QC; antibody counts are excluded from RNA QC. RNA and ADT are normalized separately. The main filter requires at least 500 detected RNA features and at most 20% mitochondrial RNA; alternative thresholds are reported.

| Library | Deposited cells | Retained cells | Biological libraries |
|---|---:|---:|---:|
| proliferating | 8664 | 7132 | 1 |
| senescent | 4949 | 3515 | 1 |

Marker detection fractions, within-library distributions and paired RNA/ADT associations describe heterogeneity. They are not senescent-state sensitivity/specificity estimates. Thousands of cells do not replace biological replication; no treatment p-values, biological confidence intervals or trained senescence classifier are reported. No doublet removal or reproduction of author clusters is claimed.

![Paired single-cell evidence](public_single_cell.png)

## Independent-context validation

[GSE235768](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235768) adds 24 samples from BJ and HFL1 fibroblasts, with three samples in each treated/control group. H₂O₂ arms include 72 hours of recovery; bortezomib arms use sustained treatment. Raw 3-prime feature counts are aggregated by gene and normalized against the independent study’s controls.

Each source model’s RNA direction predictions are frozen using GSE144397 training times. Those predictions are compared with the independent-study contrasts and eligible secretome measurements. All source models are shown; no target outcomes select or refit a source model. This tests response-direction consistency across systems, not transferable kinetic rates or feedback necessity. Sign agreement across a few correlated genes has no p-value or biological confidence interval.

| Independent context | Agreeing directions across source models | Compared markers |
|---|---:|---:|
| GSE235768/BJ/Bortezomib | 6–7 | 9 |
| GSE235768/BJ/H2O2 | 6–7 | 9 |
| GSE235768/HFL1/Bortezomib | 5–6 | 9 |
| GSE235768/HFL1/H2O2 | 6–7 | 9 |
| PXD013721/ATV SASP | 2–2 | 2 |
| PXD013721/ATV SASP (Day 9) | 2–2 | 2 |
| PXD013721/IR Epithelial SASP | 2–2 | 3 |
| PXD013721/IR SASP | 3–3 | 3 |
| PXD013721/RAS SASP | 3–4 | 4 |
| PXD013721/RAS SASP (Day 4) | 0–0 | 2 |

These mixed results include disagreements. The comparisons are not evidence that one mechanism explains all cell types, exposures or times.

![Independent RNA contrasts](independent_RNA.png)

## Cross-generator uncertainty check

The benchmark includes 150 synthetic trials (25 per generating mechanism). Feedback was selected in 36 of 125 nonfeedback-generated trials with a converged winner. The conditional Monte Carlo Wilson interval is 0.216–0.373. This exposes model-selection ambiguity under the specified generators; it is not an empirical false-positive rate or a probability that EMD4 is true. Full records are in `summary.json`.

The public [LX-2 arsenite study abstract](https://pubmed.ncbi.nlm.nih.gov/36089002/) reports a conditioned-medium effect on Huh-7 migration. This is relevant functional evidence in a different donor system, but no quantitative raw migration data were identified for this workflow. Migration is not treated as secondary senescence or proof of a feedback loop.

## Claims that this version can and cannot support

| Claim | Public-data status |
|---|---|
| Persistent arsenite-associated marker changes | Quantitative aggregate-figure evidence, with assay and replication limits |
| RNA changes after pulse withdrawal | Measured and fitted in independent WI-38 exposure protocols |
| SASP protein abundance varies by context and time | Measured public proteomics |
| RNA and surface-marker heterogeneity | Described in paired single-cell libraries |
| RNA response directions generalize to other systems | Explicit independent comparisons; inspect agreements and failures |
| Functional SASP potency in arsenite-exposed Huh-7 | Not calibrated by these public inputs |
| Self-sustaining feedback, bistability, tumor promotion | Not established by these analyses |
| Biological `q_sec_critical` and its lower edge | Not identifiable from these assays |

The mathematical threshold calculation in the original simulation is unchanged. New measurements are not pooled across cell systems to manufacture a physiological parameter estimate. Public data support triangulation and falsification; they do not supply every matched perturbation required for a causal feedback claim.

## Reproduce and review

From the parent directory, run `python3 -m emd4_simulation.public_data complete --offline`. `complete_summary.json` contains all results; `complete_source_manifest.json` contains pinned inputs; `complete_runtime.json` records code/input hashes and versions. Tables, digitization overlays and figures are standalone artifacts. See `public_data/README.md` for scope and commands. Original manuscript files remain unchanged; `manuscript_claims.md` provides wording tied to the new evidence.
