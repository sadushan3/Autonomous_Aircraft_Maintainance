# Supplementary methods and evidence map

## Evaluated bundle

- Version: `four_detector_ensemble_v1`, activated by the existing 18 October 2026 experiment.
- Training-plan ID: `a7640407800ce92ec8735a24ebad802b1e391843e38bf11ae0cb0d2d12d41c96`.
- Source predictions: `Backend/experiments/Anomaly_Health_Monitoring/run_20261018_four_detectors/official_scores.csv`.
- Scores use all four detectors with equal 0.25 weights and cutoff `0.90851712540986`.
- Source protocol: `run_20261018_four_detectors/protocol.json`; fitting uses the shared engine plan, and official test observations are excluded from fitting and calibration.
- Reference regressors and health artifacts were retained by that experiment; their monitoring reference is separate from optional adaptive forecast outputs.

## Engine roles and reference fitting

| Role | Engines | Use |
|---|---:|---|
| Fit | 396 | Learned preprocessing, context, scaling, sensor reference fitting, anomaly reference fitting, health reference, severity bands |
| Calibration | 85 | Healthy-tail fused cutoff calibration |
| Selection | 86 | Reference validation and ensemble weights; development selection in the underlying reference bundle |
| Retrospective confirmation | 142 | Excluded from present fitting/selection; not a fresh blind test |

The partition first separates confirmation engines with seed 42, then fitting engines with seed 20261015, then calibration/selection engines with seed 20261016, using the implementation's engine-level splitting and subset stratification where feasible. The saved manifest gives exact engine membership and source-subset identities.

Reference regressors select `(RUL >= 130 and cycle <= 0.30 * final_training_cycle)` plus the first ten fitting rows per engine. These are offline training-selection rules, not inference-time features. Isolation Forest, LSTM, and residual-rule references use the early-region/RUL intersection without the twin's ten-row inclusion rule. Covariance uses RUL >= 130 after commissioning. The fused cutoff uses RUL >= 125 and cycle > 30 on calibration engines. The LSTM training log records 13,804 full training windows; inference reference percentiles also include the implementation's padded early windows. The neural window-level validation fraction is not a separate engine-disjoint performance experiment.

The four-detector training log records 6,166 healthy calibration-role rows for the fused cutoff. Health reference fitting uses 27,698 healthy fitting-role rows. Severity boundaries use 12,789 fitting-role rows classified into the software's fault category. These are different calibration populations and must not be merged into a single sample count.

## Sensor reference recipe

The underlying rebuild script, `Backend/scripts/Anomaly_Health_Monitoring/retrain_without_leakage.py`, sets six context modes and uses the component's configuration recipe. The four-detector activation reuses this reference rather than training a new one.

| Regressor | Configured recipe |
|---|---|
| Random forest | 200 estimators, maximum depth 14, minimum leaf size 4, seed 42 |
| XGBoost | 250 estimators, maximum depth 6, learning rate 0.08, row/column subsampling 0.9, L2 regularization 1.0, histogram trees, seed 42 |
| LightGBM | 300 estimators, 31 leaves, learning rate 0.06, minimum child samples 20, row/column subsampling 0.9, seed 42 |
| Dense neural reference | ReLU hidden units 128/64/32 with batch normalization, dropout 0.1, linear sensor output, Adam learning rate 0.001, batch size 256, at most 150 epochs, early-stopping patience 20, learning-rate reduction patience 6/factor 0.5/minimum 0.000001, seed 42 |

The design has 11 inputs: three mode-scaled operating settings, absolute cycle, fixed clipped cycle/300, and six mode indicators. All 21 sensor targets are retained in the pooled bundle. Published weights are rounded for readability; exact values remain in `models/twin_ensemble_weights.json`.

## Health and severity

- Health reference deviation: `0.3237016428571428`; residual window 1; equal channel weights; decay 0.25; EWMA alpha 0.2.
- State cutoffs: Healthy >= 80, Degrading >= 60, Warning >= 38, Critical >= 20, otherwise Failure Risk. The state classifies the trend-adjusted health score, whereas the reported health trajectory evaluation uses the health index.
- Relative severity position: `(anomaly_score - threshold) / (1 - threshold)` for fault-category rows. Medium/high/critical starts: `0.4168996368599026`, `0.6757076516511435`, `0.8704193372158282`; other positive fault rows are low.
- Severity and health labels have no independent physical-state validation in this evidence set.

## Results provenance

`build_evidence.py` checks all original `test_FD00*.txt` and `RUL_FD00*.txt` files against the prediction rows, independently reconstructs 104,897 RUL values, checks the four-score average, and recomputes overall/per-subset metrics using the production evaluation definitions. `evidence/paper_evidence.json` includes SHA-256 hashes and detailed bootstrap intervals and engine measures.

| Manuscript item | Existing project source |
|---|---|
| Dataset table | Original `Backend/data/raw` train/test files, independently counted |
| Overall/subset anomaly results | Current four-detector `official_scores.csv`, recomputed |
| Fixed linear baseline | `run_20261015_linear_baseline_verified/comparison.csv` and `protocol.json` |
| Official health table | Recomputed from `run_20261018_four_detectors/official_inputs.joblib` using the current frozen health reference; saved as `evidence/health_recheck.json` |
| Current-batch health figure | `Backend/processed/health_index.csv`, joined to `raw_data.csv` |
| Explanation coverage/fidelity | `Backend/metrics/evaluate_explainability.json` |
| Context/reference settings | Component code, saved model metadata, and leakage-correction rebuild script |

Only the fixed linear baseline rows are copied from its comparison file. Rows for the previous consensus model are deliberately excluded from the current model's result tables.

The older `Backend/metrics/holdout_test_evaluation.json` remains in the evidence map for audit. The manuscript uses freshly recomputed health results, which differ slightly: pooled correlation 0.4601123 versus the older 0.4591615, and monotonicity 0.9477 versus 0.9463. These figures are not mixed across tables or plots.

## Reproducibility limits

This package rebuilds the manuscript's evidence from existing artifacts. It does not package the large trained models, reproduce stochastic training, provide a prospective independent dataset, or assert that historical adaptive benchmark use is removed. It does not modify the application's detectors, serving thresholds, model files, or saved evaluation files.
