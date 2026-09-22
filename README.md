# Betaine Catalyst Repository

This repository contains the data, descriptor-generation outputs, modeling
workflows, and DFT/kinetic analysis files used for the work: 
"A Predictive Betaine Organocatalyst Platform for Asymmetric 1,3-Dipolar-Cycloadditions Enabled by Interaction-Aware Modeling" (https://doi.org/10.26434/chemrxiv.15006640/v1).
It is organized as follows:

## Repository Map

| Path | What is inside | Start here |
| --- | --- | --- |
| `dataset.xlsx` | Experimental data workbook. `comb-matrix` contains the main catalyst-substrate matrix; `N-substitution` contains the N-substitution comparison set. | Open the workbook when checking source experimental values. |
| `betaine_env.yml` | Conda environment for the local Python notebooks and helper scripts. | `conda env create -f betaine_env.yml` |
| `descriptor_generation/` | Molecule lists, Gaussian log files, compact descriptor tables, and descriptor-combination scripts. | `descriptor_generation/README.md` |
| `catalyst-substrate-modeling/` | Main LASSO modeling, feature-stability and y-randomization analyses, baseline-model comparisons, prepared feature CSVs, and virtual-screening workflow. | `catalyst-substrate-modeling/README.md` |
| `catalyst-only-modeling/` | Catalyst-only feature workbook and one saved parity plot from the upstream SigmanGroup modeling workflow. | `catalyst-only-modeling/README.md` |
| `DFT_reaction-mechanism/` | Optimized structures, reaction-state energies, transition-state conformer searches, kinetic simulation notebook, and DFT support scripts. | See the folder notes below. |

## Data And Workflow Overview

1. All experimental outcomes are listed in `dataset.xlsx`.
2. Molecular descriptors are prepared under `descriptor_generation/`.
3. Combined descriptor matrices are used by the modeling folders:
   - `catalyst-substrate-modeling/` for the main catalyst-substrate LASSO,
     validation, model-comparison, y-randomization, and virtual-screening
     analyses.
   - `catalyst-only-modeling/` for the catalyst-only model input and exported
     result plot.
4. DFT reaction-mechanism and kinetic-analysis materials are stored under
   `DFT_reaction-mechanism/`.

The folders are meant to be readable independently. Use the local README files in each workflow folder for run-order notes and file-specific details.

## Descriptor Generation

`descriptor_generation/` contains the raw descriptor provenance and compact
tables used to build model-ready feature matrices.

Important files and folders:

- `descriptor_generation/README.md`: detailed guide for the descriptor folder.
- `descriptor_generation/comb_input_template.xlsx`: combined reaction template
  with catalyst, substrate, `ddG`, and descriptor columns.
- `descriptor_generation/fill_paramters.py`: fills descriptor columns in a
  combined workbook by matching catalyst and substrate IDs.
- `descriptor_generation/catalysts/`: catalyst ID workbook, Gaussian logs, and
  `all_cats_descriptors.xlsx`.
- `descriptor_generation/substrates/`: substrate ID workbook, Gaussian logs,
  and `all_substrates_descriptors.xlsx`.
- `descriptor_generation/oos_catalysts/`: out-of-sample catalyst descriptors.
- `descriptor_generation/virtual_screening/`: virtual-screening descriptor
  table for the substrate/analyte set.
- `descriptor_generation/sisso/`: SISSO-style reduced-feature inputs, outputs,
  scripts, and descriptor-name translation table.

To rebuild a combined descriptor table, run from `descriptor_generation/`:

```bash
python fill_paramters.py comb_input_template.xlsx catalysts/all_cats_descriptors.xlsx substrates/all_substrates_descriptors.xlsx
```

The script overwrites the input workbook, so work on a copy when preserving the
template matters.

## Catalyst-Substrate Modeling

`catalyst-substrate-modeling/` contains the main and supporting modeling
workflows:

- `lasso_regression.ipynb`: sequential LASSO modeling, validation,
  fixed-feature y-randomization, nested feature-stability analysis,
  virtual-screening, and PCA analysis notebook.
- `baseline_validation/model_comparison.ipynb`: comparison of categorical
  Ridge, descriptor-space k-NN, linear- and interaction-feature LASSO, and
  Elastic Net under k-fold, LOCO, LOSO, and double-cold-start validation.
- `sisso_y_randomization/`: fold-local SISSO/Boruta feature generation and
  nested LASSO y-randomization workflow.
- `linear_features/`: base linear feature train/test CSVs.
- `linear_plus_steric_interaction_features/`: interaction-feature train/test
  CSVs and code for generating result figures/tables, including the
  feature-stability analysis.
- `sisso_features/`: SISSO feature train/test CSVs.
- `out-of-sample-catalysts/`: out-of-sample catalyst panel.
- `virtual-screening/`: default virtual-screening input set and saved outputs.

The main LASSO notebook is sequential; run cells from top to bottom. The
current detailed run configuration and expected outputs are documented in
`catalyst-substrate-modeling/README.md`.

## Catalyst-Only Modeling

`catalyst-only-modeling/` contains the local inputs/results for the catalyst-only
model:

- `training_plus_val_set.xlsx`: feature table with the `summary_all_cats`
  sheet.
- `model_0_parity_publication.png`: exported parity plot.

The modeling code itself is not vendored here. It uses the
public SigmanGroup `python-modeling` workflow and the README documents how to use the
workbook there.

## DFT Reaction Mechanism And Kinetics

`DFT_reaction-mechanism/` contains DFT structures, reaction-state summaries,
transition-state conformer-search materials, and kinetic-analysis files.

Top-level DFT contents:

- `PES_states_and_energies.xlsx`: potential-energy-surface state and energy
  summary for the N-Me and 3,3'-Ph catalyst cases.
- `N-Me_catalyst/`: optimized structures, TS conformer-search results, selected
  conformer images, MD-run input, and a complete example workflow directory.
- `3_3prime_Ph_catalyst/`: optimized structures, TS conformer summaries,
  selected images, and example Chemshell inputs.
- `conformer_search_scripts/`: reusable CREST/CENSO/Chemshell helper scripts.
  See `DFT_reaction-mechanism/conformer_search_scripts/README.md`.
- `kinetic_simulations/`: kinetic-analysis notebook, concentration-time
  workbook, fitted plots, and SI text. See
  `DFT_reaction-mechanism/kinetic_simulations/README.md`.

## Environment

Create the local environment with:

```bash
conda env create -f betaine_env.yml
conda activate betaine_env
```

The environment includes Python 3.11, Jupyter kernel support, pandas, NumPy,
scikit-learn, SciPy, matplotlib, seaborn, openpyxl, RDKit, DBSTEP, morfeus-ml,
SHAP, Plotly, and notebook-rendering support packages.
The environment also installs the Boruta implementation used by the fold-local
SISSO y-randomization workflow from a fixed source revision.

External tools are still required for some workflows:

- Gaussian/GetProperties for regenerating descriptor tables from `.log` files.
- The SigmanGroup `python-modeling` repository for catalyst-only
  model execution.
- Cluster-specific CREST/CENSO/Chemshell/Turbomole tools for DFT conformer
  searches and optimizations.

## License

Software source code and scripts in this repository are released under the MIT
License. See `LICENSE`.

Data, figures, notebooks, molecular structures, calculation outputs,
descriptor tables, model outputs, and documentation are released under the
Creative Commons Attribution 4.0 International License (CC BY 4.0), unless
otherwise noted. See `DATA_LICENSE.md`.

Third-party tools and external repositories referenced here remain subject to
their own licenses.

## Reproducibility Notes

- Read the folder-level README before rerunning notebooks or scripts.
- Several notebooks are sequential and reuse variables from earlier cells.
- Rerunning notebooks or helper scripts can overwrite plots, workbooks, and
  result summaries in the active output folder.
- Generated modeling `results/` directories are retained locally but ignored
  by Git.
- This repository documents the current on-disk handoff state; it does not
  imply that all scientific computations were freshly rerun end-to-end.
