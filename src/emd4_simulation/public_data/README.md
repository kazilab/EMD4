# Public-data analysis of senescence

This analysis compares six mechanisms using measured RNA and examines arsenite responses, secreted proteins and single-cell markers. RNA expression is an indirect readout of arrest and secretion.

From the EMD4 directory, prepare the 24 public input files (about 322 MiB), then run the analysis offline.

```bash
python data/fetch_data.py
cd src
python -m emd4_simulation.public_data complete --offline --trials 25 --starts 3
```

The fetcher checks file sizes and SHA-256 hashes and saves inputs under `data/public/`. To use existing downloads, run `python data/fetch_data.py --from /path/to/raw` from the EMD4 directory. Use `--verify` to check a prepared cache. [Input manifest](../../../data/input_sources.json) · [Sources and licenses](../../../data/THIRD_PARTY.md)

## Studies and measurements

| Source | Analysis |
| --- | --- |
| GSE144397 and GPL17586 | Nine-gene panel fitted separately in RAS, RAF and quiescence time courses. |
| GSE222400 | Withdrawal RNA in three treatment arms, fitted from NCBI sample counts. The submitter archive contains differential-expression tables. Earlier samples determine count normalization, kinetic fits, marker loadings and error covariance. Both final-time samples in each arm are held out. |
| GSE210140 | Conditioned-medium and direct-exposure RNA contrasts. |
| GSE235768 | Independent RNA responses in two fibroblast cell types. |
| PMC11701098 | Aggregate marker and luminescence measurements digitized from Huh-7 arsenite figures. |
| SASP Atlas, PXD013721 | Secreted-protein abundance across induction and cell contexts. |
| GSE250041 | Descriptive paired single-cell RNA and surface-antibody measurements in one library per condition. |

Results are written to `outputs/public_data/`. Open `complete_report.md` for the findings and follow its link to the RNA fits. The JSON summaries contain the model estimates, and `complete_runtime.json` and `complete_source_manifest.json` record run settings, software versions and hashes. Use `--data-dir` and `--outdir` for other locations.

After running both the core simulation and this analysis, export the tables with `python -m emd4_simulation.manuscript_tables --public-results ../outputs/public_data` from `src/`.

## Platform sensitivity of the synthetic benchmark

The fitted candidates predict the final time point for each synthetic dataset. Among converged fits, the lowest covariance-weighted mean squared error determines the winner. In the 125 datasets generated without feedback, feedback won 36 times on macOS 26.6.2 and 46 times on macOS 27.2. Both runs used Python 3.14.2, NumPy 2.4.2, SciPy 1.17.1, seed 723, 25 trials per mechanism and three fit starts. All 150 generating parameter sets and initial states matched.

The noise covariance has three equal singular values. Platform differences in the singular vectors used by NumPy to generate multivariate normal noise could change the sampled observations at the same seed. The earlier observations are unavailable for direct comparison. The empirical withdrawal comparisons agreed on all three arms. In two arms the last-observation forecast outperformed the feedback model.
