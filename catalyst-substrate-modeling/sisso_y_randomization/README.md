# Fold-local SISSO/Boruta Y-randomization

This workflow evaluates whether the full response-adaptive SISSO/Boruta plus
LASSO procedure can obtain comparable out-of-fold performance after the
response is shuffled.

For the observed response and every permutation, it:

1. creates a stratified five-fold outer split;
2. generates and selects SISSO/Boruta features using only the outer-training
   response;
3. reconstructs the selected expressions on the untouched outer-test rows;
4. selects the LASSO alpha by repeated inner CV using the 1-SE rule, fitting
   the scaler independently in every inner-training fold;
5. refits the scaler and LASSO on the outer-training data and predicts the
   outer-test rows;
6. reports OOF and mean in-fold R2/MAE values and empirical p-values.

## Files

- `sisso_fold_features.py`: fold-local adapter around the repository's legacy
  SISSO generator and expression calculator.
- `nested_lasso.py`: fold-local scaling and 1-SE LASSO implementation.
- `run_sisso_y_randomization.py`: checkpointed experiment runner and CLI.
- `plot_sisso_y_randomization.py`: plotting and feature-frequency helpers.
- `sisso_y_randomization.ipynb`: configuration, execution, and visualization.

## Run in Jupyter

Use the repository's `betaine_env` environment because it contains `boruta`
and `openpyxl`:

```bash
cd catalyst-substrate-modeling/sisso_y_randomization
jupyter lab sisso_y_randomization.ipynb
```

The notebook defaults to two shuffles as an integration test. Increase
`N_SHUFFLES` after measuring the fold runtime. Completed outer folds are cached
under `results/runs`, so an interrupted calculation can resume.

## Run from the command line

```bash
python run_sisso_y_randomization.py --n-shuffles 2
```

For a longer run:

```bash
python run_sisso_y_randomization.py --n-shuffles 200
```

Increasing the number of shuffles reuses completed results as long as the
remaining configuration and input-file hash are unchanged.

## Generated outputs and debugging

The top-level result tables are:

- `response_summary.csv`: one aggregate row for the observed response and each
  completed permutation;
- `empirical_significance.csv`: observed values, null means/standard deviations,
  and add-one empirical p-values;
- `all_fold_metrics.csv`: feature-generation, alpha-selection, timing, and
  train/test metrics for every completed outer fold;
- `all_predictions.csv`: all observed and randomized OOF predictions;
- `analysis_configuration.json`: input hash and settings used for resumability.

Each response and fold also has its own checkpoint directory:

```text
results/runs/observed/fold_01/
├── fold_record.json
├── selected_augmented_features.txt
├── sisso.log
└── test_predictions.csv
```

`fold_record.json` includes counts at every main SISSO stage, the selected
alphas, number of nonzero LASSO coefficients, in-fold and outer-test metrics,
and elapsed time. Set `verbose=True` in `SISSOConfig` to stream the legacy
feature-generation and Boruta output to the console. With `verbose=False`, the
same legacy messages are captured in each `sisso.log`.

For long runs, the command-line runner is convenient because the notebook stays
available for inspecting completed checkpoints. For example:

```bash
tail -f results/runs/observed/fold_01/sisso.log
```

## Methodological boundary

SISSO/Boruta is fitted independently in every outer-training fold. The selected
representation for that outer fold is then held fixed during its inner LASSO
alpha search. Thus, the outer OOF observations cannot influence feature
generation or selection. This implementation does not rerun SISSO/Boruta a
second time inside every inner alpha-selection fold; the LASSO alpha search is
conditional on the representation selected from the complete outer-training
partition.
