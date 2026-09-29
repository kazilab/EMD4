# Public-data evidence workflow

This extension tests EMD4-related hypotheses against public measurements and stronger alternatives. It is separate from the nominal arsenite model and cannot establish EMD4 by reproducing its own simulated truth. RNA observations are not interpreted as a senescent-cell fraction or functional SASP potency.

From the parent `hKCC` directory:

```bash
python3 -m emd4_simulation.public_data fetch-complete
python3 -m emd4_simulation.public_data complete --offline --trials 25
python3 -m pytest emd4_simulation/tests -q
```

The download cache is `emd4_simulation/data/public/`; results are in `emd4_simulation/public_data_results/`. Open `complete_report.md` for the complete version, `manuscript_claims.md` for evidence-linked wording, and `report.md` for the original core analysis. The `run` and `fetch` commands remain available for that smaller core workflow. Change locations with `--data-dir` and `--outdir`. `--trials 100` increases the synthetic benchmark from its default 25 trials per mechanism; `--starts 3` controls multistart fitting. Dependencies are NumPy, SciPy, pandas, matplotlib, Pillow and pytest for tests; exact verified versions are in [requirements-tested.txt](requirements-tested.txt). No account, API key, raw-read aligner or private data is required. The complete cache is approximately 355 MiB; the sparse single-cell importer does not densify the complete gene matrix.

Downloads are atomic and pinned on first retrieval with URL, accession, byte count, UTC retrieval time and SHA-256. Every subsequent run checks the cached bytes. The first download trusts the public source; it is not compared with a publisher-provided cryptographic checksum. A new source snapshot belongs in a new data directory. Offline runs do not access the network. Code hashes and package versions are recorded, but cross-platform floating-point results can differ slightly.

## Data and protocol clocks

| Source | Imported evidence | Treatment of evidence |
|---|---|---|
| GSE144397 / GPL17586 | 12 RAS, 12 quiescence and 12 RAF samples; nine genes | Each context fitted separately; continuous induction, physical hours converted to days. Entire last time point held out. Source RMA/batch correction retained. |
| GSE210140 | 20 IHH samples; TPM and sample metadata | 2-hour donor pulse, wash, 72-hour conditioning; recipients receive 50% CM for 72 hours. Endpoint RNA contrasts only. |
| GSE222400 | NCBI-generated raw counts for 52 samples; 50 used in timed analysis | **Recovered:** the separate NCBI matrix supplies sample measurements. The original 52 submitter DE tables remain excluded. D0 is model day 1; D16 (day 17) is held out. |
| PMC11701098 | 38 aggregate means digitized from Huh-7 figures 2, 5, 6 | Quantitative assay evidence with image hashes, coordinates and ±2-pixel sensitivity. Luminescence after reseeding has its own clock; PCR normalizers differ by phase. |
| PXD013721 | SASP Atlas S1 and S6 | 6,420 protein-comparison rows and 38 cell-culture records; abundance is not functional potency. |
| GSE250041 | Paired RNA and eight ADT markers | Descriptive analysis of 13,613 deposited cells; 10,647 retained by primary QC. One biological library per condition. |
| GSE235768 | 24 independent BJ/HFL1 samples | Three samples per treated/control group; H2O2 includes 72-hour recovery. Frozen source predictions are checked without target model fitting. |

A fixed panel includes CDKN1A, CDKN2A, LMNB1, MKI67, IL6, CXCL8, SERPINE1, MMP3 and GDF15. Probe annotations come from the GPL table bundled with the public deposit. Multiple-symbol probes are excluded; unambiguous probes for a gene are aggregated by median. Historical IL8 maps explicitly to CXCL8. Exact Ensembl gene IDs from that annotation join the IHH matrix. TPM is summed for exact IDs mapping to the same symbol, then transformed by log2(TPM+1). This is a descriptive assay scale, not a count-based differential-expression pipeline. Zero observed variance produces no Welch interval; all-zero TPM must not appear as a precisely zero biological effect. Missing genes and all selected mappings are recorded. Studies are not pooled to estimate universal biological rates.

The complete workflow imports the SASP Atlas and CITE-seq data rather than merely cataloguing them. The LX-2 arsenite conditioned-medium study (PMID36089002) remains published-summary evidence: it reports migration in recipient Huh-7 cells, not a quantitative secondary-senescence feedback assay. No numerical effect is invented from its abstract.

GSE222400 is recovered through NCBI's public HISAT2/featureCounts output, avoiding local raw-read reprocessing. This is a distinct processing source, not a repair of the unsuitable submitter tables. The two replicative-senescence samples are excluded because they lack the acute protocol's time origin. Exact marker GeneIDs come from the already pinned GPL17586 annotation. No unsupported gene-ID inference or baseMean substitution is used.

For the recovered series, median-of-ratios normalization learns positive reference genes and geometric means from the 44 pre-holdout samples. Each held-out sample is scaled against that frozen reference; it never changes the reference or selected genes. The independent study sums raw 3-prime features by gene and uses its controls as the normalization reference. This is a relative-expression analysis, not absolute RNA per cell. The separately cached author-normalized GSE235768 table is retained for source inspection; it is not mixed into the raw-count analysis.

The secretome adapter reads cached source values and their sheet/row/cell addresses, never edits the source workbooks and rejects formulas without cached values. Author SD and peptide-ratio counts are not biological SEM or biological replicate n. All imported rows are retained, with author q-values; missing proteins and ambiguous protein groups are not converted into zero measurements or duplicate gene observations.

The single-cell adapter requires unique column-major sparse coordinates to prevent duplicated entries inflating detected-gene counts, streams the sparse matrix, uses RNA only for per-cell UMI, detected-feature and mitochondrial QC, and normalizes RNA as log1p(counts/total RNA × 10,000). ADTs are transformed as log1p(count) minus the within-cell mean across the eight antibodies. This explicitly defined compositional scale is not background-corrected protein concentration. Primary QC uses at least 500 RNA features and at most 20% mitochondrial RNA; 200-feature and 10% mitochondrial alternatives are counted. Surface RNA/ADT associations and marker detection fractions are descriptive. No cell is treated as an independent culture, no treatment p-value is calculated, and no clustering or doublet-removal result is claimed.

Digitization records preserve phase, assay, normalization, source image and manual coordinates. The normalised 4-hour luminescence value is a reference, not an additional independent datum. PCR measured relative to 18S during exposure is never pooled with RPLP1-normalized withdrawal PCR. Fold changes are withheld when the control bar cannot be resolved above the pixel-reading range. Apparent luminescence relaxation is not senescence clearance. SA-beta-gal fields are not assumed to provide independent biological replication. Pixel sensitivity is separate from experimental uncertainty.

## Mechanisms and observation model

Six candidates are implemented:

1. Reversible arrest with rapidly resolving damage.
2. Slow damage resolution with reversible arrest.
3. Durable intrinsic arrest without secreted protein.
4. Persistent arrest and SASP without paracrine induction.
5. One-way paracrine induction: primary cells secrete and induce secondary arrest, but secondary cells do not secrete active factors.
6. Paracrine feedback: secondary cells can sustain active secretion.

The state vector is damage, primary immature/mature arrest, secondary immature/mature arrest, total secretory memory, extracellular protein, and secondary secretory memory. Arrest occupancy is bounded by one. It is a closed-population approximation; clearance returns occupancy to the available pool and is not a detailed model of death, cell division, immune recruitment or tissue growth. Physical days match the protocol. Dimensionless exposure amplitude and protein units remain model conventions.

Two secretion structures are available: `matched_clearance`, where immature and mature cells have the same arrest-loss rate; and `lag_memory`, where secretory drive relaxes after arrest has disappeared. Protocol integration is split exactly at exposure withdrawal, medium replacement, blocking, rescue and donor removal. Medium replacement removes extracellular protein, not intracellular damage. Conditioned-medium transfer carries neither donor cells nor donor damage. A separate assumed toxicant-carryover control can be simulated.

Measured RNA is fitted through gene-specific intercepts and signed loadings. Arrest-panel genes use arrest occupancy; CDKN1A also permits damage; GDF15 uses damage; secretory-panel genes use maturation/memory. These mappings are **hypotheses**, not validated marker calibrations. They do not turn RNA into measured protein. Coefficients are re-estimated for every competing mechanism. No gene signs or sample labels are used to certify senescence.

## Inference and its limits

Kinetics use bounded multistart least squares. Nuisance observation coefficients are solved by linear GLS at every kinetic evaluation. Within-time replicate residuals estimate marker covariance; default shrinkage is 50% toward the diagonal, with a 0.05 log2-unit SD floor. Baseline occupancy and diagonal covariance are checked separately. These choices are declared assumptions. The RAS sensitivity analysis is not an exhaustive posterior over all model forms. The same six candidates and two structures are fitted to all three recovered withdrawal arms with explicit medium-replacement events.

Induction spans approximately 0.005–20/day, maturation 0.1–10 days, persistence rates 0.002–1/day and paracrine gain 0.05–10/day. Fixed damage repair, protein clearance, response scale and functional fraction are conventions, not public-data estimates. Reversible arrest has fixed loss 1/day; the slow-damage candidate varies repair instead of arrest loss. All bounds, fixed rates, boundary hits, ranks, sample IDs, starts and convergence flags are recoverable from source/results. A boundary hit is a warning about identification, not a precise estimate. Last-observation and linear-time forecasts provide additional training-only predictive benchmarks.

Both final-time replicates are held out together, and all observation coefficients/covariance estimates use earlier times. The published preprocessing predates this split. Repeated replicate trajectories may have temporal dependence, so forecasting error is descriptive; it is not an independent validation p-value. No final all-data refit is substituted for the forecast. Tied candidates are not evidence for one mechanism over another. In particular, durable arrest and nonfunctional persistent SASP are observationally equivalent in the selected RNA map when paracrine induction is zero.

Feedback gain is profiled on a finite grid with other quantities refitted. No confidence interval or biologically calibrated threshold is reported. Functional fraction and receptor-response scale are confounded: multiplying both by the same factor leaves trajectories unchanged. This workflow therefore **cannot tighten the biological interpretation of `q_sec_critical`**, including its lower band edge. It also does not change the existing mathematical threshold calculation.

The synthetic benchmark generates data from every candidate, varies parameters/structure/baseline, adds correlated noise and replicate offsets, and fits every candidate to measured channels with nuisance coefficients. Last-time predictive winners form a confusion table. The reported false-feedback selection fraction is conditional on these generators. Its Monte Carlo Wilson interval is not an empirical specificity estimate. Increase trial count before interpreting its numerical precision.

## Virtual experiments and remaining evidence

The donor–recipient extension compares control CM, treated CM, recipient washout, receptor inhibition, partial-block ligand add-back, continuous donor input, donor removal and toxicant carryover. One-way induction can persist as arrest after the transferred protein disappears; that is not a secretion loop. Complete receptor blockade cannot be rescued by more ligand. Baseline arrest, rates, functional fraction, cell-density normalisation and the extra interventions are uncalibrated assumptions.

A parameter/structure sweep ranks candidate post-withdrawal sampling times by separation in assumed RNA observables. It is a design heuristic, not power or expected information gain, and does not refit nuisance coefficients. Direct arrest/proliferation measurements plus functional recipient perturbations remain necessary. EMD4, feedback necessity/sufficiency and bistability are distinct claims; a favorable RNA fit establishes none of them on its own.

The generated `complete_report.md`, `complete_summary.json`, `complete_source_manifest.json` and `complete_runtime.json` keep evidence and provenance explicit. Independent validation reports response-direction agreement for every frozen source model, not target-fitted kinetic parameters. The original `report.md` and `summary.json` retain the core analyses. Original publication outputs are neither overwritten nor read as empirical validation. No manuscript claim is automatically strengthened.
