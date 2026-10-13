# Autonomous Maintenance Supervisor — Fresh Run Audit (FD002)

**Run analyzed:** `/integration/run-full` executed 2026-09-15 (~11:21 local), input file `test_FD002.txt` (different FD subset from the earlier `test_FD001.txt` run, as requested).
**Repo:** `Anomaly_Health_Monitering_System\Anomaly_Health_Monitering_System\Backend` (the standalone repo confirmed as canonical).
**Scope:** end-to-end pipeline output — Anomaly & Health Monitoring → RUL/Failure-Risk → Supervisor fusion/RL decision → baseline comparison — for this specific run.

---

## 1. Run metadata

| Metric | Value |
|---|---|
| Input | `test_FD002.txt` |
| Rows produced | 33,991 |
| Engines | 259 |
| Columns (merged supervisor input) | 49 |
| Merge validation errors / warnings | 0 / 0 |
| Synthetic fallback used | `false` |
| Inference pipeline duration | 3.45 s |
| Supervisor training pipeline duration | 288.8 s |

The merge step is clean: 0 validation errors, 0 warnings, no synthetic fallback, and — checked directly against the raw CSV — 0 nulls across `predicted_rul`, `risk_10`, `risk_30`, `anomaly_score`, `health_index` over all 33,991 rows. The join/data-plumbing layer is solid for this run.

---

## 2. Critical finding: the PPO policy has collapsed to a single constant action

```
raw_model_decision value_counts:
schedule_maintenance    33991   (100% of rows)
```

Cross-tabulated against `health_state` (Anomaly & Health Monitoring's own independent severity label for the same rows):

| health_state | raw_model_decision |
|---|---|
| Healthy (22,147 rows) | schedule_maintenance |
| Degrading (7,704 rows) | schedule_maintenance |
| Warning (3,996 rows) | schedule_maintenance |
| Critical (137 rows) | schedule_maintenance |
| Failure Risk (7 rows) | schedule_maintenance |

The champion PPO model outputs the identical action for engines the upstream health model calls perfectly healthy and engines it calls in active failure risk. This is not noise — it is a deterministic, universal collapse, and it is consistent with what training already showed:

- `train_ppo_report.json`: 32 evaluation episodes, `std_episode_return: 0.0` — literally zero variance across every episode.
- `compare_baselines_report.json`: Champion Supervisor Agent (PPO) scores accuracy 0.1537 / macro-F1 0.2377, while a trivial Rule-Based Baseline scores 1.0/1.0, and even the single-signal RUL-Only baseline scores 0.9985/0.9959.

**Root cause (confirmed, not speculative):** `rl_environment.py`'s reward function branches on raw physical-unit thresholds — e.g. `predicted_rul <= 1.0`, `<= 15.0`, `> 60.0`, and `risk_10 > 0.85` / `> 0.70`. But the actual features reaching the PPO environment during training/inference have already been through `RobustScaler`. From this run's own `feature_scaling_report.json`:

| Feature | Original range | Scaled mean | Scaled std |
|---|---|---|---|
| `predicted_rul` | 0.0 – 319.96 (mean 103.23) | 0.038 | 0.633 |
| `risk_10` | 0.0 – 0.9625 | 10.25 | 33.24 |

With `predicted_rul` scaled to a mean of 0.038 and a std of 0.633, the condition `predicted_rul > 60.0` is essentially never true and `predicted_rul <= 1.0` is essentially always true regardless of the engine's actual remaining life — so the reward landscape the agent is trained against no longer reflects real degradation state at all. This single scale mismatch is sufficient on its own to explain the zero-variance episodes and the constant output. Notably, `safety_policy.py` elsewhere in the same codebase explicitly detects and unscales standardized RUL/risk values before applying its own thresholds — the pattern for fixing this already exists in the repo, it just isn't applied inside `rl_environment.py`.

**This is the single highest-priority defect in the component.** Every downstream number that depends on `raw_model_decision` — and, transitively, most of the value proposition of an "autonomous" RL supervisor — is not currently trustworthy until this is fixed and the PPO model retrained.

Note also: `final_decision` differs from `raw_model_decision` in 31 of 33,991 rows (all overridden from `schedule_maintenance` to `immediate_maintenance`), so a downstream safety-override layer is doing real work and partially masking the collapse — but 33,960 rows still ship the same untouched `schedule_maintenance` regardless of true health state.

---

## 3. Confidence score and human-review coverage

`confidence_score` takes only two distinct values across all 33,991 rows:

| confidence_score | rows | `model_used` |
|---|---|---|
| 0.70 | 33,699 | 33,110 rows: no model recorded (`NaN`) — placeholder/default confidence, no real scoring; 589 rows: `TCN_Conformal` (genuine) |
| 0.45 | 292 | 289 rows `TCN_Conformal`, 3 rows `TCN_Conformal+RandomForest_Fallback` |

So of 33,991 decisions, only **881 rows (2.6%)** carry a `model_used` value at all, i.e. only 881 rows received genuine per-row RUL/Failure-Risk model scoring; the remaining 33,110 rows (97.4%) are stamped with a flat default confidence of 0.70 with no model attribution. This matches the Anomaly & Health Monitoring log line for this run: *"RUL/Failure-Risk batch complete: 881 point(s) over 209 engine(s)... 50 engine(s) skipped as shorter than the required window."* — i.e. 50 of 259 engines (~19%) received zero genuine scoring for their entire trajectory, relying entirely on forward-filled/placeholder values, by design of the auto-stride budget sampler.

Combined with the PPO collapse, this yields:

```
requires_human_review:
True     33,985   (99.98%)
False        6
```

Effectively every decision the system produces this run is flagged for mandatory human review. That is a reasonable *safety* posture given the PPO collapse, but it means the "autonomous decision-making" objective is not currently being demonstrated end-to-end — the system is, in practice, a review-queue generator right now, not an autonomous actor.

---

## 4. Model ensemble degradation: two deep-learning detectors failed to load

From `logs/anomaly_health_monitoring.log` for this run window:

- `Could not reload twin 'tensorflow': Failed to load model ...tensorflow_twin.keras: ...could not be deserialized properly... Error when deserializing class 'Dense'...`
- `Detector 'lstm_autoencoder' could not be reloaded: Failed to load model ...lstm_autoencoder.keras...`

Both failures are `.keras` deserialization errors (a Keras/TensorFlow version or config-compatibility issue, not a data issue). The system evidently degrades gracefully — the run completed and produced output — but the anomaly ensemble is running with fewer models than intended for this run, which should be treated as a reliability gap to close, not just tolerated.

---

## 5. Severe operating-context (regime) drift flagged

The pipeline's own PSI (Population Stability Index) drift monitor fired repeatedly and severely for this run:

```
Operating-context drift detected: PSI=7.7253 exceeds 0.2000...
Operating-context drift detected: PSI=0.3057 exceeds 0.2000...
```

A PSI above 0.2 is conventionally considered major drift; 7.7 is enormous — roughly 38x the alert threshold. This is expected and informative, not a bug: FD002 has multiple operating conditions and fault modes, unlike FD001, so a calibration/scaler fit on a different regime will legitimately drift this hard. It does, however, mean that whatever RUL/risk/anomaly thresholds were calibrated against FD001 (or a different regime) should not be assumed valid as-is for FD002-style multi-condition data — this is a second, independent reason (beyond the scaling bug above) that decisions on this run should be treated as provisional.

---

## 6. Everything that worked correctly in this run

For balance — this run also confirms several things are working as designed:

- **Join integrity**: 0 nulls across the join keys and all core scored features over 33,991 rows; 0 merge validation errors/warnings.
- **Leakage guard**: `feature_selector.py` and `leakage_guard.py` share the same canonical `FORBIDDEN_FEATURE_COLUMNS`/`DERIVED_OUTPUT_COLUMNS` source — the previously-flagged disconnect is fixed.
- **Champion selection logic behaves exactly as designed**: PPO is selected over RandomForest (0.9996 F1)/XGBoost (0.9992)/LightGBM (0.9984) via the explicit `ppo_mandated_by_directive` rule, specifically to avoid crowning a classifier that's just imitating the rule cascade. That design intent is sound; it's the PPO training itself (the scaling bug above) that's currently broken, not the selection policy.
- **Safety-override layer is active**: 31 rows were escalated from `schedule_maintenance` to `immediate_maintenance` despite PPO's constant raw output, indicating a downstream guard is doing real work rather than the system blindly trusting the RL policy.
- **Graceful degradation**: two failed Keras models did not crash the pipeline; the ensemble fell back and the run completed.
- **Genuine per-row scoring, where it runs, is real**: the 881 rows with a real `model_used` (TCN_Conformal, TCN_Conformal+RandomForest_Fallback) show the actual online conformal/fallback scoring path executing, not a stub.

---

## 7. Prioritized recommendations

1. **Fix the RL reward-threshold scale mismatch in `rl_environment.py` first.** Either (a) unscale `predicted_rul`/`risk_10`/etc. back to physical units before applying the reward thresholds — mirroring the pattern already implemented in `safety_policy.py` — or (b) rewrite the thresholds in scaled-feature units and validate them against the actual `RobustScaler` statistics. Retrain PPO after the fix and confirm episode-return variance is non-zero and `raw_model_decision` varies with `health_state`/`risk_10` before reporting any RL results as valid.
2. **Investigate the two Keras deserialization failures** (`tensorflow_twin.keras`, `lstm_autoencoder.keras`) — likely a save/load format or TF version mismatch — to restore full ensemble coverage.
3. **Re-run the drift check after recalibrating (or documenting) regime-specific thresholds for FD002-style multi-condition data**, since the current PSI values indicate FD001-calibrated thresholds are not directly valid here.
4. **Decide on the auto-stride coverage tradeoff.** 2.6% real scoring coverage with 19% of engines receiving zero genuine RUL/Failure-Risk scoring is a real accuracy/throughput tradeoff worth either tightening (denser stride) or explicitly documenting as a known limitation in the report/paper.
5. Once (1) is fixed, re-run `compare_baselines_report` and confirm the champion PPO agent's macro-F1/cost_penalty move meaningfully off its current near-random values (0.2377 / 146,903) before using this comparison table as evidence of RL competence in the paper.

---

## Bottom line

The data pipeline, join, leakage guard, champion-selection policy, and safety-override layer are all functioning correctly on this fresh FD002 run. The blocking problem is a concrete, root-caused bug: PPO's reward function compares scaled features against raw-unit thresholds, causing the trained policy to collapse to a single constant action regardless of true engine health. This is fixable (the un-scaling pattern already exists elsewhere in the codebase) and should be treated as the top-priority item before any RL-based results from this component are presented as evidence of a working autonomous supervisor.
