# EMD4 senescence after exposure withdrawal

The notebook reproduces Figure 6. It tests whether a senescent-cell population can persist after arsenite withdrawal and whether reversible responses can fit the same observations during exposure. The simulation uses assigned rates, dimensionless exposure and time in model days. [Open the notebook](EMD4_simulation_walkthrough.ipynb).

## Run the notebook

From the EMD4 directory, create an environment with Python 3.14.2 and install the tested scientific packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c requirements-tested.txt
python -m jupyter lab EMD4_simulation_walkthrough.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`. In Jupyter, select the environment’s Python kernel and choose **Run All Cells**. The reference simulation runs offline. It saves Figure 6 and its caption under the filename stem `emd4_figure6_senescence` in `outputs/reproduced/`, alongside the numerical summary, model tables and run log.

## Interpreting the model

The 14-state model tracks two senescent populations. Cells enter senescence directly in response to exposure or in response to factors released by senescent cells (the senescence-associated secretory phenotype, SASP). The two populations together give total senescent burden. Six marker readouts can differ from this burden because they also reflect background signal, stress or delayed responses. The low-dose SA-β-gal comparison adjusts for the marker–burden difference in the unexposed control.

The reference exposure ends on day 14. Reversible alternatives are fitted to synthetic observations during continuous exposure and then used to predict the response after withdrawal. In the 250-draw sensitivity analysis, exposure-attributable persistence requires a senescent fraction above 0.30 on model day 400 in the withdrawal arm and at or below 0.30 in the paired control. A separate calculation tests for two stable equilibria at zero exposure. The spatial analysis uses 40 replicates per secretion ratio to estimate the probability that senescence reaches the lattice boundary by day 300.

## Command line and tests

From the EMD4 directory, run the reference analysis, export its tables and run the regression tests.

```bash
cd src
python -m emd4_simulation.run
python -m emd4_simulation.manuscript_tables
python -m pytest -q emd4_simulation/tests
```

## Public measurements

The optional public-data analysis fits candidate mechanisms to measured RNA and evaluates their predictions on held-out samples. It uses 24 external files (about 322 MiB) and writes to `outputs/public_data/`. The instructions document the inputs and the platform sensitivity of the synthetic benchmark. [Public-data instructions](src/emd4_simulation/public_data/README.md) · [Data sources and attribution](data/THIRD_PARTY.md).
