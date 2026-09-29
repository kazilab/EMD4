# EMD4: minimal reproducibility deposit

An illustrative mechanistic simulation of EMD4 (persistent cellular senescence with a functionally characterised senescence-associated secretory phenotype, SASP), with arsenite as the exemplar exposure ([Okamura et al., *Environ Health Prev Med* 29:74, 2024](https://doi.org/10.1265/ehpm.24-00139); Huh-7 cells). The model implements the proposed typed links KCC5 → EMD4 (home KCC6) → KCC7, KCC10, with KCC9 as opposing polarity. It has 14 states: a latent senescent state, a secreted SASP pool, primary and secondary senescent cells, immune and intrinsic clearance, and terminal escape. Around it sit a stochastic hex-lattice layer and a 250-draw sensitivity ensemble. A separate public-data comparison fits six candidate mechanisms to public RNA time courses and scores them on held-out samples.

The mechanistic model's parameters are illustrative, not fitted. They place the system in the regime the published qualitative observations describe: persistence after withdrawal, secondary senescence and a resolving low-dose response. Its 13 internal consistency checks were revised during development; they are neither preregistered nor independent experimental validation. The public-data comparison uses measured data but does not calibrate the mechanistic model.

## Reproduce the simulation offline

Use Python 3.11 or later. From this archive's root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-tested.txt
cd code
python -m emd4_simulation.run
python -m emd4_simulation.manuscript_tables
python -m pytest emd4_simulation/tests -q -p no:cacheprovider
```

No downloads are needed. Output goes to `code/figures/output/`: `emd4_simulation_summary.json`, `emd4_manuscript_tables.json` and the figure `figure2c_emd4_simulation.{pdf,svg,png,eps,tif}`. The full run took about 4 minutes on the reviewed machine. It covers the 250-draw ensemble, a 40-replicate spatial scan at each tested `q_sec`, and the `k_p` identifiability profile. It exits non-zero if any check fails. `--n-ensemble 0 --no-spatial --no-identifiability --no-figure` gives a quick diagnostic run, but `manuscript_tables.py` refuses to write tables from a partial summary.

Three of the tests check the package README's quoted numbers and the JSON files against a generated summary in `code/figures/output/`. They skip until `run` has written it, so run the tests last.

`manuscript_tables.py` also reads the bundled public-data results (`code/emd4_simulation/public_data_results/complete_summary.json` and `summary.json`) for the public-data table. The tables therefore reproduce without the public-data download.

The independent threshold audit cited in the supplement is also bundled. From
the archive root, after activating the same environment:

```bash
python verification/threshold/check_thresholds.py
```

It checks the fixed seed-4 ensemble against a separately parameterised
derivative solver and repeats the grid-refinement counts. See
`verification/threshold/README.md` for its scope.

Compare the two generated JSON files with `outputs/`. When regenerated from an isolated copy of this archive, both were **byte-identical** to the deposited copies. Floating-point results may vary slightly on other platforms. Reviewed software: Python 3.14.2 on macOS (arm64), NumPy 2.4.2, SciPy 1.17.1, Matplotlib 3.10.8, pandas 2.3.3, Pillow 12.1.1, pytest 9.0.2. `requirements-tested.txt` pins those versions; the core requirements give minimum versions. The deposited run passed **13/13** internal checks and **59** regression tests, and the isolated copy reproduced both.

## Reproduce the public-data comparison (optional)

This part downloads about 322 MiB from GEO, Europe PMC and PLOS. From the archive root:

```bash
python data/fetch_data.py
cd code
python -m emd4_simulation.public_data complete --offline --outdir ../reproduced_public
```

`fetch_data.py` checks every file against the SHA-256 recorded when the deposited results were made and refuses any that differ. `python data/fetch_data.py --verify` reports what is present without downloading. The complete run took about 3 minutes on the reviewed machine. Compare its output with `code/emd4_simulation/public_data_results/`. Without `--outdir` it overwrites those bundled results in place, and `manuscript_tables.py` reads them from there.

From an isolated copy of this archive with the pinned inputs installed, every numerical result regenerated identically, with these expected metadata or serialisation exceptions:

- `complete_runtime.json` records the absolute data path; the deposited copy also carries a finalisation timestamp and note from the final QA pass. Its source-code hashes match the released complete workflow.
- `runtime.json` is retained as the historical record of the earlier core public-data run. In addition to its absolute data path, its hashes for `__main__.py`, `arsenite.py`, `complete.py`, `extension_sources.py` and `single_cell.py` predate the final extension and digitisation QA changes. The current code reproduces the core result files byte-identically; `complete_runtime.json` is the final-code provenance record.
- `complete_summary.json` is identical in every value, but two top-level text keys appear in the opposite order. The deposited copy was written during that final QA pass.
- `verification.json` is a hand-written record of the original verification, not a code output.

## What the analysis shows

**An endpoint fit does not identify feedback.** A no-feedback model fits the simulated forward dose–response at R² = 0.996 with a Hill coefficient of 17.9. It fits the longitudinal record less well (R² 0.884; 0.902 for a cascade null) and relaxes after withdrawal, while the mechanistic model persists at S = 0.924 on day 400. The targets are generated by the mechanistic model itself, so this shows what a no-feedback model cannot reproduce, not that feedback is present in biological data.

**Whether the nominal model has a tipping point reduces to a containment threshold, `q_sec* = 0.243`.** This is secondary-cell secretion relative to primary-cell secretion. The threshold also moves with receptor sensitivity, SASP clearance and paracrine gain: doubling `K_P` doubles it. A measured protein-output ratio alone therefore cannot calibrate it. Across the ensemble the threshold is 0.254 [0.113–0.489], conditional on the 198/250 draws with a crossing.

**Existence claims survive the ensemble; contrast claims do not.** High burden is retained in 235/250 draws. Two stable equilibria at zero exposure occur in 170/250, and exposure-attributable persistence in 167/250. The low-dose contrast, the SASP-neutralisation-versus-senolytic contrast and marker decline after withdrawal are fragile or not robust. Net of the concurrent control, the SA-β-gal over-read is +3.5 points [−1.7, +15.9], positive in 211/250 draws. The absolute figure (+12.2 points) includes the assay's own background.

**Feedback existence is identifiable in a synthetic record; its strength is not.** `k_p = 0` is rejected by a single-dose continuous time course. Admissible `k_p` still spans 0.208–0.743 (3.57-fold) under a restricted profile with a two-percentage-point tolerance.

**Public RNA data give feedback no consistent predictive advantage.** Feedback had the lowest training error in each withdrawal arm. On held-out samples, a last-observation forecast beat it for DXR (5.37 versus 6.79) and KCl (32.97 versus 72.41), and feedback improved only slightly for SDS (10.94 versus 11.08). In a synthetic benchmark, feedback was selected in 36 of 125 records generated without feedback.

## Contents and provenance

| Path | Purpose |
|---|---|
| `code/emd4_simulation/model.py` | 14-state ODE, exposure schedule, intervention handles, six-marker observation model |
| `code/emd4_simulation/bifurcation.py` | Equilibria, folds, hysteresis, critical slowing, and the analytic fold curve behind `q_sec*` |
| `code/emd4_simulation/nulls.py` | Simple and cascade no-feedback nulls |
| `code/emd4_simulation/identifiability.py` | `k_p` profile: continuous, joint and withdrawal-only designs |
| `code/emd4_simulation/conditions.py` | Checks V1–V13, each with its evidentiary basis |
| `code/emd4_simulation/ensemble.py` | 250-draw log-normal sensitivity analysis and finite exposure thresholds |
| `code/emd4_simulation/spatial.py` | Gillespie hex-lattice layer |
| `code/emd4_simulation/run.py`, `figure.py`, `manuscript_tables.py` | Full run, publication figure, and table extraction with a completeness audit |
| `code/emd4_simulation/README.md` | Detailed technical record, including the audit corrections and the claims they changed |
| `code/emd4_simulation/public_data/` | Public-data workflow; its `README.md` records per-study units, clocks, holdout design and exclusions |
| `code/emd4_simulation/public_data_results/` | Results of the public-data run: summaries, reports, pinned source manifest and tables. Figures are excluded and regenerate. |
| `code/emd4_simulation/tests/` | Regression tests for the simulation and the public-data workflow |
| `verification/threshold/` | Independent derivative-based threshold check, grid refinement and expected results |
| `outputs/` | Reference simulation summary and manuscript-table JSONs |
| `data/` | Source manifest, third-party attribution and the checksum-verified downloader. **No third-party raw file is bundled.** |
| `MANIFEST.sha256` | Checksums for every deposit file except the manifest itself |

The deposit excludes publication images, Word files and their generators, audit and review working folders, the raw public-data cache and caches. The figures regenerate from code. The Word toolchain stays in the working manuscript project and is not needed to reproduce the numerical analysis.

## Archive identity and reviewer access

This directory is the complete EMD4 reproducibility archive, version 1.0.0,
released 27 September 2026. For peer review, provide the whole directory as the
submission's code/data archive or through a reviewer-accessible repository; the
general hKCC GitHub link in `CITATION.cff` does not by itself identify this
exact snapshot. When a DOI-minting repository record is created, add that DOI
to `CITATION.cff` and cite it in the manuscript without changing the archived
version.

Suggested manuscript statement before the DOI is assigned:

> Code, derived results and the exact tested environment for the EMD4 analyses
> are supplied with the submission as the EMD4 reproducibility archive,
> version 1.0.0. The archive includes reference outputs, regression tests,
> checksum-pinned source manifests and an independent threshold verification.
> Public experimental inputs remain in GEO, Europe PMC and PLOS at the
> accessions and URLs listed in `data/SOURCES.tsv`; `data/fetch_data.py`
> retrieves and verifies the analysed snapshots. The archive will be deposited
> in a DOI-minting repository for publication, and the DOI will replace this
> submission-stage statement.

## Evidentiary limits

- **Nothing in the mechanistic model is fitted.** Time is in nominal days and dose is dimensionless, with `E = 1` as the reference exposure. The day-14 withdrawal and day-400 follow-up are simulation choices, not the Huh-7 protocol (72 h exposure, assay-specific follow-up).
- **The null comparison is against model-generated targets.** Both nulls relax by construction once exposure stops, so they reject reversible relaxation specifically. Durable intrinsic arrest and slowly resolving damage are tested only in the separate public-data candidate family.
- **Bistability is a consequence of the parameter choice, not a measurement.** At nominal settings the zero-exposure system is bistable. Across the ensemble this holds in 170/250 draws, below the 90% robustness convention used here.
- **Intervention arms are predictions.** No senolytic, SASP-neutralisation or immune-modulation arm exists for arsenite-exposed Huh-7 cells. Alimirah et al. (2020) is an analogue in a different system.
- **The lattice is not a reduction of the ODE.** It has its own induction, kernel, maturation and clearance rules. At 41×41 it measures extinction probability, not how far a front travels. That the mean-field threshold falls inside its transition band is a consistency observation.
- **Public-data results are exploratory.** RNA is not latent senescent occupancy, and secreted protein abundance is not functional potency. Each withdrawal arm has two held-out samples at one time, and configuration selection uses that same holdout. The imported assays identify neither functional feedback strength, nor bistability, nor a biological containment threshold.
- **`X` counts escape events; it does not measure immortalisation.** It can exceed 1. The KCC9 rule, that exposure-induced senescence is not evidence of immortalisation, is imposed on the model rather than derived from it.
- **Model output is not independent KCC or EMD evidence** and is not a prediction of carcinogenic potency.

Software: MIT. New derived results: CC BY 4.0. Values derived from third-party sources keep their original attribution and terms. See `LICENSE`, `LICENSE-DATA` and `data/THIRD_PARTY.md`.
