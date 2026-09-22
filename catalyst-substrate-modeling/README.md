# Catalyst-Substrate Modeling

This folder contains the main LASSO workflow, supporting validation analyses,
and prepared feature tables for catalyst-substrate modeling.

## Main Notebooks And Workflows

- `lasso_regression.ipynb`: main LASSO analysis, fixed-feature
  y-randomization, nested feature-stability analysis, external-test
  predictions, virtual screening, and PCA.
- `baseline_validation/model_comparison.ipynb`: categorical Ridge,
  descriptor-space k-NN, linear- and interaction-feature LASSO, and Elastic
  Net evaluated under identical k-fold, LOCO, LOSO, and double-cold-start
  splits.
- `sisso_y_randomization/`: local fold-specific SISSO/Boruta plus nested LASSO
  y-randomization workflow. See its README for execution details.

The main notebook is organized as a sequential analysis workflow. Run cells
from top to bottom because later sections reuse variables created earlier,
especially `df`, `X`, `y`, `feature_names`, `scaler`, `output_folder`, and
fitted model objects.

## How To Run

The notebook uses paths relative to this directory. Start Jupyter from
`catalyst-substrate-modeling` or confirm the kernel working directory is this
folder before running cells.

```bash
cd catalyst-substrate-modeling
jupyter lab lasso_regression.ipynb
```

For the baseline-model comparison, open
`baseline_validation/model_comparison.ipynb` and run it from top to bottom.

The notebooks were authored with a local `modeling` kernel. In a clean
installation, select the published `betaine_env` Python 3.11 kernel instead.
Core Python packages used by the notebooks include `pandas`, `numpy`,
`matplotlib`, `scikit-learn`, `seaborn`, `plotly`, `boruta`, and `openpyxl`.

## Default Run Configuration

The first data-loading cell currently uses the virtual-screening feature set:

- `input_folder = "virtual-screening/"`
- `input_file = "training_set_base_plus_int.csv"`
- `ext_test_file = "external_test_set_base_plus_int.csv"`
- `output_folder = "virtual-screening/results/"`

The virtual-screening prediction cell then reads:

- `vs_file = "virtual_screen_set.csv"`

With the default settings, generated plots and tables are written to
`virtual-screening/results/`.

## Input Sets

Available prepared input folders:

| Folder | Training file | External-test file | Virtual-screening file |
| --- | --- | --- | --- |
| `virtual-screening/` | `training_set_base_plus_int.csv` | `external_test_set_base_plus_int.csv` | `virtual_screen_set.csv` |
| `linear_plus_steric_interaction_features/` | `training_set_base_plus_int.csv` | `external_test_set_base_plus_int.csv` | not included |
| `linear_features/` | `training_set_base.csv` | `external_test_set_base.csv` | not included |
| `sisso_features/` | `training_set.csv` | `external_test_set.csv` | not included |
| `out-of-sample-catalysts/` | `train_val_set.csv` | `external_test_set.csv` | `virtual_screen_set.csv` |

To switch input sets, edit `input_folder`, `input_file`, `ext_test_file`, and
`output_folder` in the first data-loading cell. For out-of-sample catalyst
screening, also keep `vs_file = "virtual_screen_set.csv"` in the prediction
cell.

## Workflow Sections

The notebook includes:

- imports and matplotlib styling
- data loading, feature filtering, and train/external-test setup
- target-distribution and catalyst-substrate plots
- leave-one-group-out setup and LOGO LASSO validation
- final LOGO test-only parity plot and workbook export
- full-training-set LassoCV and fixed-alpha LASSO fits
- nested fixed-feature y-randomization with fold-local scaling, 1-SE alpha
  selection, OOF evaluation, null distributions, and empirical significance
- repeated nested-CV feature-selection and coefficient-stability analysis
- highlighted catalyst/substrate ranking plots
- external-test prediction plots
- retraining on train + external-test data for virtual screening
- virtual-screening predictions, alpha-summary table, and ddG/ee plot
- PCA, virtual-screening PCA overlays, manual PCA, and manual PCA overlays

## Run Order Notes

Run the LOGO setup cell before the LOGO LASSO cell. Run the full LassoCV cell
before the nested y-randomization and feature-stability sections because they
reuse its alpha grid. Run the fixed-alpha refit before comparing the nested
stability results with the final fitted model.

For virtual screening, run the train + external-test retraining cell before the
prediction cell. The prediction summary plot reads all
`virtual_screen_predictions_alpha_*.xlsx` files in the active `output_folder`,
so remove or move old alpha files if you only want one alpha represented.

For PCA overlays, run the PCA setup cells and the virtual-screening setup cell
first. Manual PCA overlays require the manual PCA cell to be run first so the
fitted steric and electronic scalers are available.

## Outputs

Most generated files are written under the active `output_folder`. 
Existing output filenames are reused by design, so rerunning cells can
overwrite previous files in the selected results folder.

Baseline-comparison outputs are written below `baseline_validation/results/`.
Feature-stability outputs are written below
`linear_plus_steric_interaction_features/results/feature_stability/`.
Generated `results/` directories are ignored by Git and remain local unless
they are deliberately copied elsewhere.

## Reproducibility

For exact reproduction:

- run cells in order
- keep the same Python environment and package versions
- keep input column names unchanged
- keep the intended `input_folder`, `input_file`, `ext_test_file`, `vs_file`,
  and `output_folder` settings for the analysis being reproduced
- clear or archive old result files when summary cells glob over prior outputs
