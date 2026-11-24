# EMD4 data sources and attribution

The core simulation uses assigned parameters and exposure schedules. The public-data analysis uses the following sources. The manifest records the original accession, URL, file size, SHA-256 and retrieval date for each input. [Input manifest](input_sources.json) · [Public-data instructions](../src/emd4_simulation/public_data/README.md) · [Code license](../LICENSE)

- GEO studies GSE144397, GSE210140, GSE222400, GSE235768 and GSE250041 supply expression data, sample annotations and paired RNA and antibody-derived tag measurements. The GSE144397 family file includes GPL17586 probe annotations.
- The code digitizes aggregate values from figures 2, 5 and 6 of Okamura, Sato, Suzuki and Nohara, *Environmental Health and Preventive Medicine* 29, 74 (2024), PMC11701098, under CC BY 4.0. The observed Huh-7 responses motivate the exposure–withdrawal model. The model's day-14 withdrawal and day-400 endpoint are assigned. [Original article](https://doi.org/10.1265/ehpm.24-00139)
- The secretome analysis uses SASP Atlas supplementary tables S1 and S6 from Basisty et al., *PLoS Biology* 18, e3000599 (2020), under CC BY 4.0. The study is indexed under PRIDE PXD013721. [Original article](https://doi.org/10.1371/journal.pbio.3000599)

## Literature informing model assumptions

| Evidence | Source |
| --- | --- |
| Arsenite-induced senescence and SASP in hepatic stellate cells | [Okamura et al. (2022)](https://doi.org/10.1016/j.taap.2022.116231) |
| Senolytic intervention in a different experimental system | [Alimirah et al. (2020)](https://doi.org/10.1158/0008-5472.CAN-20-0108) |
| Primary and secondary SASP production, used to motivate the secondary secretion parameter and spatial model | [Martin et al. (2023)](https://doi.org/10.1111/acel.13892) |
| Multiple markers for senescence, including SA-β-gal | [Ogrodnik et al. (2024)](https://doi.org/10.1016/j.cell.2024.05.059) · [SenNet recommendations (2024)](https://doi.org/10.1038/s41580-024-00738-8) |
| Reported effect of LX-2 conditioned medium on Huh-7 migration, used as qualitative context | [PMID 36089002](https://pubmed.ncbi.nlm.nih.gov/36089002/) |
