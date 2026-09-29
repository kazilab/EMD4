# Third-party data attribution

The EMD4 deposit has two parts, and they differ in what they take from third parties.

**The mechanistic simulation uses no third-party data.** Every number in `outputs/emd4_simulation_summary.json` comes from the bundled code and chosen, illustrative parameter values. No published time series or marker panel is fitted. Published studies are cited for structural choices and protocol shape (see below). Those citations are scholarly references, not redistributed material.

**The public-data comparison reads 24 third-party files. None is bundled.** Together they total about 322 MiB and stay with their original repositories. `SOURCES.tsv` lists each file with its accession, URL, byte count and pinned SHA-256. `fetch_data.py` downloads a file only when it matches its pin:

```bash
python data/fetch_data.py            # download and verify (~322 MiB)
python data/fetch_data.py --verify   # check what is present; downloads nothing
```

A source that has changed since the pin is refused, and nothing is installed. NCBI can regenerate GEO `family.soft.gz` files and Europe PMC can re-render article XML, so a mismatch may mean the upstream file changed rather than that the analysis is wrong.

## Derived values that are redistributed

`code/emd4_simulation/public_data_results/` contains the results of the public-data comparison. Its CSV and JSON files include values derived from the sources below: selected-gene expression values, digitised figure means and secretome comparison rows. They are there so reviewers can inspect the comparison and rebuild the manuscript tables without the download. Each keeps its source's attribution and terms:

| Source | Terms | Derived values in this deposit |
|---|---|---|
| Okamura K, Sato M, Suzuki T, Nohara K. *Environ Health Prev Med* **29**:74 (2024), [doi:10.1265/ehpm.24-00139](https://doi.org/10.1265/ehpm.24-00139); PMC11701098 | CC BY 4.0 | `arsenite_digitized.csv`: 38 aggregate means read manually from Figures 2, 5 and 6. Each has image hash, pixel coordinates and a ±2-pixel reading sensitivity. The coordinates are in `public_data/arsenite.py`. The figure images are not redistributed. |
| SASP Atlas: Basisty N *et al.*, *PLoS Biol* **18**:e3000599 (2020), [doi:10.1371/journal.pbio.3000599](https://doi.org/10.1371/journal.pbio.3000599); PRIDE PXD013721 | CC BY 4.0 | `secretome_comparisons.csv`, `secretome_panel.csv`: rows from Tables S1 and S6, each with its source sheet and row. The workbooks are not redistributed. |
| NCBI GEO GSE144397, GSE210140, GSE222400 (NCBI-generated counts), GSE235768, GSE250041 | NCBI places no restrictions on use of GEO data; submitters may hold rights in their submissions | Nine-gene panel values in `observations.csv`, `withdrawal_observations.csv` and `independent_observations.csv`, contrasts, and single-cell summaries. No expression matrix is redistributed. |

What changed from the sources: gene-panel selection, normalisation as described in `code/emd4_simulation/public_data/README.md`, figure digitisation, and conversion to CSV/JSON with source provenance. These are secondary analyses, not new experiments. The EMD4 software authors are not the authors of any of these studies.

## Literature cited for structure, not data

| Reference | Used for |
|---|---|
| Okamura K *et al.*, *Environ Health Prev Med* **29**:74 (2024), [doi:10.1265/ehpm.24-00139](https://doi.org/10.1265/ehpm.24-00139) | Exemplar: arsenite-exposed Huh-7 cells keep senescence-associated phenotypes after withdrawal. Sets the exposure–withdrawal shape; the model's day-14/day-400 schedule is not that study's. |
| Okamura K *et al.*, *Toxicol Appl Pharmacol* **454**:116231 (2022), [doi:10.1016/j.taap.2022.116231](https://doi.org/10.1016/j.taap.2022.116231) | Arsenite-induced senescence and SASP in hepatic stellate cells, a second cell type. |
| Alimirah F *et al.*, *Cancer Res* **80**:3606 (2020), [doi:10.1158/0008-5472.CAN-20-0108](https://doi.org/10.1158/0008-5472.CAN-20-0108) | The closest published analogue for the senolytic arm. A different system; not a matched experiment. |
| Martin L, Schumacher L, Chandra T, *Aging Cell* **22**:e13892 (2023), [doi:10.1111/acel.13892](https://doi.org/10.1111/acel.13892) | Primary versus secondary SASP output; motivates `q_sec` and the spatial lattice. The lattice is written from the published description, not vendored. |
| Ogrodnik M *et al.*, *Cell* **187**:4150 (2024), [doi:10.1016/j.cell.2024.05.059](https://doi.org/10.1016/j.cell.2024.05.059); SenNet recommendations, *Nat Rev Mol Cell Biol* **25**:1001 (2024), [doi:10.1038/s41580-024-00738-8](https://doi.org/10.1038/s41580-024-00738-8) | Marker criteria: SA-β-gal is neither necessary nor sufficient; the observation model follows that. |
| PMID 36089002 | LX-2 arsenite conditioned medium increases Huh-7 migration. Used only as a published summary; no numerical effect is taken from it. |

See `code/emd4_simulation/public_data/README.md` for per-study units, protocol clocks, holdout design and eligibility decisions.
