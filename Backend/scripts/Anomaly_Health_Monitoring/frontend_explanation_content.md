# Your Anomaly & Health Monitoring Frontend

## Every card. Every number. Every graph.

A practical explanation of the four screens in your AeroTwin OS component, based on the current frontend code, the backend calculations and a read-only capture of the local API on **18 October 2026**.

| Screen | The question it answers |
| --- | --- |
| Health Monitoring | How healthy are the engines, and which ones are deteriorating? |
| Anomaly Detection | Which readings are unusual, how strong is the evidence, and how does the detector perform? |
| Digital Twin | What would a healthy engine be expected to read, and how far do actual readings depart from it? |
| AI Reasoning | Which sensors and subsystems explain the deviation, and what should an analyst inspect? |

The explanations follow the labels visible in the frontend. Each section tells you what the number means, its unit and population, how to read the graph, and what you can reasonably say about the result.

> Worked examples use the captured dashboard: **93 engines, 12,812 cycle rows, mean final-cycle HI 73.6**. These are a dated example, not permanently fixed product values.

This document covers **only Anomaly & Health Monitoring**. The separate RUL Prediction, Failure Risk, Maintenance Supervisor and Confidence & Trust pages are outside this component. The maintenance action queue and feedback panels embedded in your Health Monitoring page are included.

Graphs in this PDF are redrawn from the captured API data for readability. They are explanatory figures, not screenshots of the browser.

---PAGE---
# Reading map

{CONTENTS}

---PAGE---
# 1. Understand the numbers first

## Engine, row and cycle mean different things

An **engine** is one monitored unit. A **row** is one observation of an engine at a particular cycle. A **cycle** is the dataset's operating-time step; it is not automatically one hour or one day. An engine can contribute hundreds of rows and many alerts.

| Value or label | Meaning and direction |
| --- | --- |
| Health index, HI | A derived condition index from 0 to 100. Higher means closer to the frozen healthy reference. It is not model accuracy. |
| Health score | HI adjusted by recent health trend; this is the value used by the backend health-state classifier. |
| Anomaly score | A normalized score from 0 to 1. Higher means stronger anomaly evidence. It is not a calibrated probability of failure. |
| Confidence | This component's 0–100 blend of agreement and data quality. Higher means stronger support for the output, not a healthier engine. |
| Reliability / agreement | Values from 0 to 1. They describe agreement, not measured diagnostic correctness. |
| Residual | Actual sensor reading minus the twin expectation, after the applicable normalization and offset correction. |
| Residual z | Residual standardized against its reference scale. The sign gives direction; absolute size gives unusualness. |
| RUL | Actual remaining cycles, when available for retrospective evaluation. Your component uses RUL condition proxies in training/evaluation; it does not turn HI into a life prediction. |

## How your component connects the evidence

Operating context and sensor data → healthy twin expectation → residuals → health index and anomaly detection → rule-supported cause hypotheses → explanation and confidence → analyst feedback.

The healthy monitoring reference and the separately evaluated adaptive sensor forecast serve different purposes. An actual-behavior forecast may track degradation; an anomaly reference must preserve a healthy comparison against which degradation remains visible.

> **LIVE** is a frontend display badge. The pages read backend outputs, many of which are saved pipeline files. The badge alone does not prove continuous sensor streaming, fresh model training or a recently regenerated report.

---PAGE---
# 2. Health Monitoring — headline cards

## The Basis control changes the population

**Current snapshot** takes the final observed row of each engine: one engine, one vote. **Full lifetime** uses every recorded row: longer histories contribute more observations. The switch changes the mean-HI card and the health-state counts/distribution. It does not change every panel on the page.

| Card | Captured example | What it shows |
| --- | --- | --- |
| Mean HI (final cycle) | 73.6 / 100 | Arithmetic mean of the 93 engines' final-cycle health indices. It does not mean 73.6% prediction accuracy. |
| Lifetime Mean HI | 92.0 / 100 | Mean across all 12,812 recorded health-index rows. Earlier healthy operation explains why it can exceed the snapshot mean. |
| Critical engines / rows | 8 / 34 | Snapshot: eight final rows labelled Critical. Lifetime: 34 historical rows labelled Critical. This excludes the separate lowest band. |
| HI < 20 band engines / rows | 0 / 0 | Counts the backend's lowest health-state label, called Failure Risk internally. It is a health tier, not the Failure Risk agent's output. See the state-label qualification on page 6. |
| Healthy engines / rows | 34 / 10,491 | Counts final-cycle Healthy engines or historical Healthy rows, depending on the basis. |
| Mean confidence | 95.0 | Mean final-cycle component confidence on a 0–100 scale. This card remains a snapshot statistic when Full lifetime is selected. |

## What the headline colors mean

The mean-HI card is green above 70, amber above 45 and red otherwise. These are **presentation thresholds for a fleet average**, not the backend's individual health-state boundaries. Other cards have fixed colors, so a green confidence tile is not evidence that every engine is safe.

For a presentation, say: “The fleet has a final-cycle mean health index of 73.6. Across all recorded operation the mean is 92.0, because the lifetime view includes earlier, healthier cycles.”

---PAGE---
# 3. Health trajectory — lines and axes

![Captured engine E76 trajectory, redrawn from the per-engine timeline endpoint.](figures/health_trajectory.png)

| Graph element | How to read it |
| --- | --- |
| X-axis: cycle | The selected engine's recorded operating cycles, in chronological order. |
| Y-axis: 0–100 | A shared display scale for HI, the RUL-derived proxy and anomaly score × 100. The three curves do not have the same meaning. |
| Cyan health-index area | The component's smoothed residual-based HI. A sustained decline indicates worsening deviation from the healthy reference. The filled area is decorative, not a confidence interval. |
| Dashed purple line | 100 × clip(actual RUL, 0, 130) / 130. The UI calls this “True remaining life”; it is a capped RUL-derived condition proxy on a 0–100 scale, not remaining cycles and not independently measured health. |
| Orange anomaly line | Anomaly score multiplied by 100 to share the axis. A height of 80 represents score 0.80, not 80% failure probability. |
| Dashed horizontal line at 20 | A visual HI reference labelled HI < 20. Backend health states use the trend-adjusted health score, so a label need not switch exactly when this curve crosses 20. |

## The r value in the subtitle

The browser computes **Pearson correlation r** between available HI and target pairs for this engine. It measures linear association, ranging from −1 to +1. It is **not sensor-prediction R²**. At least three valid pairs and nonzero variance are needed; otherwise r is unavailable.

Changing the selected engine changes the trajectory. The default is the lowest-HI engine; the “Fastest Deteriorating” list can select a different one. Missing target values mean the purple line cannot be interpreted as available ground truth.

---PAGE---
# 4. Health states and diagnostic detail

![The same captured fleet counted as engines and as historical rows. The two horizontal scales differ.](figures/health_states.png)

## Health State Distribution

The horizontal axis counts **engines** in snapshot mode or **rows** in lifetime mode; the vertical axis lists the five health states. Snapshot counts are Healthy 34, Degrading 30, Warning 21, Critical 8 and lowest band 0. They sum to 93. Lifetime counts sum to 12,812.

The classifier applies descending thresholds to **health score**, not directly to the displayed HI. Health score = clip(HI + 1.5 × health trend, 0, 100), with trend in HI points per cycle. A negative trend lowers the score.

The local persisted boundary file contains 80 / 60 / 38 / 20. Some captured explanation records were generated with a 40-point Warning/Critical boundary. Use the saved state and its generation-time evidence for that observation; a later threshold file does not rewrite an older label. “HI < 20” is therefore a simplified UI label for the lowest score-based tier.

## Engine E… — Diagnostic Detail

| Field group | What the fields mean |
| --- | --- |
| Health index; state; remaining health | Current HI, score-based state, and 100 × current HI / best HI observed so far for that engine, clipped to 0–100. Remaining health is not RUL. |
| Alert level; score; type; severity | Alert urgency, fused anomaly score, fault/operational/none classification, and score-band severity. They describe different aspects of the observation. |
| Persistence; early warning | Consecutive flagged cycles and a bounded score combining anomaly strength with health decline. Neither is a countdown to failure. |
| Root cause; evolution; top sensors | Suspected cause, temporal pattern and ranked contributing sensors. These are evidence-supported outputs, not confirmed diagnoses. |
| Mode; cycles observed | Operating-regime category and maximum recorded cycle number for the engine. Cycle count assumes the input history is complete. |
| Confidence; reliability; twin spread | Output support on 0–100, detector agreement on 0–1, and mean dispersion of twin predictions in model units. |

---PAGE---
# 5. Health diagnostics — three panels

## Detector Ensemble — E…

This horizontal bar chart shows the selected engine's four legacy diagnostic detector fields plus the fused anomaly score. The X-axis is 0–1; the Y-axis names the detector. Longer bars mean larger scores, not stronger proven causal responsibility.

The small values below the chart are **Detector agreement**, **Model agreement**, **Fusion threshold** and **Uncertainty**. Agreement is higher when outputs are similar; uncertainty is 1 − model agreement. The threshold belongs to the stored observation.

> In the captured configuration, **condition_consensus has weight 1.0 and the other four detectors have weight 0**. The four legacy scores are missing in the current dashboard data. This chart converts missing values to zero, while the fused score can remain high. Do not describe those zero bars as four active detectors finding no anomaly.

## Fleet Root Causes

The horizontal axis is a count of attributed **historical rows**, not unique engines or independent failures. The vertical axis lists the top five stored cause labels. A cause may be repeated across many cycles of the same engine.

In the capture, the leading cause is High-Pressure Turbine degradation, with **386 rows**. Only five causes are included by this dashboard field. The AI Reasoning page can use the full eight-cause distribution from the reasoning report.

## Fastest Deteriorating

This list shows up to eight engines sorted by descending **deterioration rate = max(−health trend, 0)**. Each entry shows engine ID, state, HI and rate in **HI points lost per cycle**. Clicking an entry selects it for the trajectory and diagnostic detail.

Example: engine E76 has HI **33.37**, state **Critical**, and deterioration rate **0.732 points/cycle** at its final observed cycle. The rate is a recent fitted trend, not a promise that exactly that much health will be lost on every future cycle.

The early-warning score uses clip(anomaly score + 0.6 × 30 × decline rate / 100, 0, 1) with the current 30-cycle horizon. At high anomaly scores it can saturate at 1.0. A saturated value is a prioritization signal, not 100% failure probability.

---PAGE---
# 6. Health validation and comparability

## Health Index Validation

These tiles read a **saved retrospective benchmark report**, rather than recomputing quality from the currently displayed 93-engine fleet.

| Tile / table | Meaning and captured availability |
| --- | --- |
| Monotonicity | Fraction of qualifying engines with at most 15% of late-life steps increasing by more than one HI point. Captured value 0.9463: about 94.63% meet this tolerance. It does not require every step to decrease. |
| Within-engine ρ | Intended to show median per-engine Spearman rank correlation between HI and actual RUL. The exact field used by this tile is absent in the captured report, so the tile shows a dash. A differently named trajectory statistic is not automatically displayed here. |
| Fleet ranking ρ | Intended to measure ranking across engines. The expected field is absent, so the captured UI cannot supply this number. |
| RUL correlation | Pooled Spearman correlation between HI and actual RUL: **0.4592** in the saved report. This is an association with a condition proxy, not an R² or diagnostic accuracy. |
| Per-subset table | FD001–FD004 engine count, monotonicity, RUL correlation and mean HI. It reveals whether a pooled result hides weaker subsets. |
| Gate badge | Acceptance status read from evaluate_health, with its reported correlation and monotonicity threshold. The captured report describes a current-batch evaluation, despite the UI's “Training-set acceptance” wording. |

## Cross-engine comparability

When the required bias-quartile data exist, bars show mean HI for Q1–Q4 on the **left axis**, and a line shows median starting RUL on the **right axis**. The two axes have different units. The left HI axis is set to 80–95, so bar differences can look larger than on a full 0–100 scale.

In this capture, `bias_quartiles` is absent; the panel displays its empty state. Do not invent a quartile result or use it to claim that absolute engine rankings have been validated.

> The saved benchmark is labelled **reused_benchmark_exploratory**. Some frontend prose still states a fixed within-engine correlation of 0.98 or near-perfect monotonicity. Those sentences are older copy; use the current report fields and state their population.

---PAGE---
# 7. Historical alert mix and context drift

## Alert level across recorded history

These five tiles count **every scored historical row** in the saved health-alert report: none **10,491**, info **1,688**, warning **564**, high **61**, critical **8**. Each percentage divides its count by the sum of the displayed levels, here 12,812.

They are alert levels, not health-state counts or unique engine counts. A state transition or rapid health decline can escalate a health alert. Selecting Full lifetime does not turn this already-historical panel into a different calculation.

## Operating-Context Drift

![Reference and current operating-mode proportions from the context endpoint.](figures/context_drift.png)

The X-axis lists operating modes. The Y-axis is each mode's percentage of rows. Gray bars describe the persisted training reference; blue bars describe the current context data. A mode number is a category, not a health rank.

| Tile / badge | Captured value and interpretation |
| --- | --- |
| PSI | **7.928**. Population Stability Index compares the two mode distributions; zero means identical proportions. Higher means a larger shift. It is not a percentage. |
| Threshold | **0.20**, the configured drift trigger. |
| Severity / badge | **Major / Drift**. Current context rows concentrate in mode 1, while the reference contains six modes. |

PSI sums (current share − reference share) × ln(current share / reference share), with small-value protection for missing modes. A changed mix warrants checking coverage and calibration. It does **not by itself prove** an engine fault or a failed twin; a single-subset upload can intentionally have a different mix.

---PAGE---
# 8. Maintenance queue and feedback

## Maintenance Action Queue

The queue is a component-level triage table, not the separate Maintenance Supervisor optimization page. It takes the **returned alert feed**, orders alert levels critical → high → warning → low, then orders scores descending, and displays at most **12 rows**. Level buttons filter that returned feed.

| Column / control | Explanation |
| --- | --- |
| Engine; Level | Engine ID and alert urgency for that stored row. Several rows may belong to the same engine. |
| Score | Fused anomaly score, displayed to three decimals. |
| Persist. | Consecutive flagged cycles as recorded by anomaly detection. A larger count suggests a sustained signal. |
| Recommended action | Component-generated inspection guidance for analyst review. |
| “Open alerts” count | Length of the returned feed: 100 in the capture. The backend caps this feed at 100 historical alert rows. This is not a count of unresolved maintenance tickets. |

## Analyst Feedback Loop

| Tile | Captured value and meaning |
| --- | --- |
| Base threshold | **0.5000**, the frozen baseline threshold. |
| Live threshold | **0.5238**, the adapted threshold state returned by the feedback service. |
| Effective scale | **1.0476**, live / base. The service reports this as unclamped in the capture. |
| Verdicts applied | **3**, against three submitted verdicts: one false alarm and two accepted. These are feedback records, not three newly trained models. |

The Health page buttons submit **False alarm** or **Confirm fault** for the selected engine's latest known cycle, with an optional note. The fuller feedback control on Anomaly Detection and AI Reasoning also offers **Correct**, **False Alarm**, **Missed**, and **Add detail** for a corrected cause and notes.

False-alarm feedback generally raises the threshold; missed/accepted evidence can lower it under the backend adaptation policy. The response reports whether thresholds changed. Submitting feedback persists a record and can alter later decisions; it does not establish physical truth by itself.

> Stored alerts still show the threshold used when they were generated: **0.5** in this capture. A live threshold of 0.5238 does not retroactively rescore old alert rows. Compare an alert to its own `threshold_used`.

---PAGE---
# 9. Anomaly Detection — cards and models

| Headline card | Captured value | What the number counts |
| --- | --- | --- |
| Total Anomalies | 1,193 | All historical dashboard rows with anomaly type other than none. It is not 1,193 engines. |
| Genuine Faults | 1,193 | Rows classified by the component as fault anomalies. “Genuine” is a UI label; independently confirmed fault labels are unavailable. |
| Operational (gated) | 0 | Rows classified operational rather than fault. Zero means none were retained in that category for this stored output. |
| Fusion Threshold | 0.5000 | Threshold used for the selected alert. It is not the latest feedback threshold and not an average across alerts. |
| Critical Engines | 3 | Final-cycle engines whose anomaly severity is critical. This differs from the Health page's eight Critical health-state engines. |

## Detector Ensemble explanation cards

| Visible legacy detector | What it is designed to detect |
| --- | --- |
| Isolation Forest | Unusual residual combinations through tree-partition isolation. |
| Mahalanobis | Multivariate distance accounting for covariance between residual channels. |
| LSTM Autoencoder | Unusual sequence behavior through reconstruction error. |
| Residual Rule | Direct exceedance of residual rules or thresholds. |

Each visible detector tile displays its backend weight and algorithm principle. The **captured weights of all four are zero**. The backend additionally reports **condition_consensus, weight 1.0**: ExtraTrees consensus over causal residual histories, trained with RUL condition proxies. The frontend's fixed four-tile layout does not render this fifth detector.

> The page's “four independent detectors” and “healthy p99” descriptions are incomplete for the active configuration. The current consensus detector uses its own frozen healthy-tail scaling and margin. Explain the actual backend configuration, not only the older subtitle.

---PAGE---
# 10. Detection Performance — validation

This panel reads the saved official C-MAPSS retrospective benchmark. It does not evaluate the currently displayed upload on demand. The capture includes 104,897 input rows; **83,788 ready rows from 690 engines** enter the reported evaluation, with warm-up/unready rows excluded.

| Tile | Captured value | Exact interpretation |
| --- | --- | --- |
| AUC | 99.97% | Area under the ROC curve for ranking healthy versus critical RUL-proxy rows. AUC is 0.999713, not 99.97% confirmed diagnosis accuracy. |
| False alarms | 0.16% | Fault flags on healthy-proxy rows / healthy-proxy rows: 80 / 49,232 ≈ 0.1625%. Lower is better. |
| Detection rate | 100.0% | 159 detected engines / 159 engines that reach critical RUL. It is not 159 / all 690 engines and is not row recall. |
| Separation | 20.70 | Cohen's d: standardized separation of healthy and critical score distributions. It has no percent unit. |

## Early-warning lead time

The median **actual RUL at the first alert** is **52 cycles**, with an interquartile range of **43–71**. The marker represents the median; the shaded segment represents the middle 50% of observations. The frontend's visual scale is 0–200 cycles and can clip longer values.

This does not prove that every warning was correct or that true physical fault onset was detected 52 cycles early. The report explicitly notes that first alerts can include false alerts. Sustained-alert lead time is a separate statistic.

![Same saved per-subset results as the UI, split into separate scales here so small false-alarm rates remain visible.](figures/detection_subset.png)

The UI plots FD001–FD004 on X and percentages on Y, with green AUC and red false-alarm bars. Both are percentages but have different denominators and directions. Healthy means **RUL ≥ 125**; critical means **RUL ≤ 30**. Middle-band rows are excluded from binary metrics. These are condition proxies; independent confirmed fault labels are absent.

---PAGE---
# 11. Severity and operational gating

## “Do the severity bands mean anything?”

When available, this chart plots severity none → low → medium → high → critical on X and **mean actual remaining cycles** on Y, using the saved evaluation report. Counts below it show how many evaluated rows contributed to each mean.

Lower mean RUL in more severe bands would support a descriptive association with worsening condition. It would not make a severity label a calibrated RUL estimate. In the current capture, the required severity-mean-RUL fields are absent, so this graph has no populated result. Do not supply older illustrative values as current results.

## Fault vs. operational gating

| Tile | Captured saved-report value | Meaning |
| --- | --- | --- |
| Raw flags | 641 | Threshold-crossing rows counted by the saved evaluate_anomaly report. |
| Kept as faults | 641 | Flagged rows retained as fault under that report's policy. |
| Reclassified operational | 0.0% / 0 | Fraction and count of raw flags treated as operational. |
| Mean RUL at a fault flag | 35.7 cycles | Mean actual remaining life of fault-flagged rows in that report; fleet mean is 141.3 cycles. |

The gate considers operating-context transitions and persistence to avoid treating every mode change as degradation. Inspect the returned numbers: the old explanatory sentence “one flag in five” does not match this capture's zero operational suppression.

The 641-report count differs from the dashboard's **1,193** fault rows. They are different saved stage outputs/policies and cannot be combined as one synchronized confusion matrix. These panels currently lack a shared run identity that would establish equivalence.

## Severity Distribution

X lists anomaly-severity categories; Y counts final-cycle engines. Captured counts are none **52**, low **15**, medium **13**, high **10**, critical **3**; total **93**. The frontend sorts categories by count, so position does not necessarily mean increasing severity.

Severity uses the score's position above its threshold and persisted calibrated bands. It is distinct from health state and from alert level. A Critical health-state engine can have high anomaly severity and a high alert without contradiction.

---PAGE---
# 12. Detector graphs, breakdown and feed

## Mean Detector Score Across Alerts

The horizontal axis is mean score, 0–1; the vertical axis lists the four legacy detectors. Each bar is the arithmetic mean **across the returned alert feed only**, capped at 100 rows. It is not a fleet-wide mean, a weighted contribution or a detector-accuracy comparison.

The red line is the selected alert's fusion threshold. That threshold applies to the **fused** score, not separately to each individual detector. Current missing legacy fields become zero in these bars; the active consensus score is not one of the four bars.

## Engine E… — Detector Breakdown

Engine buttons select from the first **18 alert rows**. The radar chart has one axis per legacy detector and a 0–100 radius. It multiplies each 0–1 score by 100. The radar shape summarizes a score pattern; it is not a failure-probability polygon. Adjacent bars show those values to four decimals.

| Detail tile | Interpretation |
| --- | --- |
| Fused Score; Threshold | Stored score and the threshold used for that same alert. E76 at cycle 198 shows 1.0000 versus 0.5000. |
| Model Agreement; Uncertainty | Similarity among twin predictions and its complement. Example 0.861 and 0.139. |
| Detector Agreement; Reliability | Agreement among available detector scores; reliability equals detector agreement. Example 1.000 and 1.000. With fewer than two available detector columns, agreement defaults to 1. |
| Persistence (cycles) | Consecutive anomaly-flag run length; 42 for the example alert. |
| Early Warning | Current anomaly plus decline pressure, clipped to 0–1; example 1.00. It is not a predicted failure date. |

## Anomaly Feed

The table displays the first **20 returned alert rows**: Engine, Cycle, Severity, Fused score, IF, Mahal., LSTM-AE, Rule, Persist and Agreement. Agreement is detector agreement × 100%; persistence is cycles. A dash means missing data in this table, whereas some charts display the same missing field as zero.

The backend feed uses non-none alert levels when available, orders by severity then cycle, and caps results at 100. It can include multiple historical rows for the same engine and health-derived alerts without a fault anomaly. The feedback control records a verdict for the selected alert's exact engine and cycle.

---PAGE---
# 13. Digital Twin — headline and members

This page presents the **healthy monitoring reference**, using saved training, residual-validation and comparison artifacts. Its cards do not directly display the separate adaptive raw-sensor forecasting R².

| Card | Captured value | How to interpret it |
| --- | --- | --- |
| Ensemble RMSE | 0.4747 | API field weighted_mean_rmse: an inverse-RMSE-weighted average of member RMSEs. It is not a newly computed RMSE of fused predictions and not an R². Units follow the normalized sensor targets. |
| Twin Models Fused | 4 | Random Forest, XGBoost, LightGBM and Keras MLP. These are twin regressors, not the anomaly detectors. |
| Member Disagreement | 0.065 | Mean twin spread across the scored output. Larger dispersion means more disagreement; it is not an error rate or calibrated interval width. |
| Sensors Modelled | 17 / 21 | 21 modeled targets; 17 have nonzero healthy p99 residual z in this response. The picker uses these informative channels. This count is not the forecasting metric's row denominator. |

The subtitle also reports **six regimes, 12,812 rows and 93 scored engines**. These are not the same population as the training/validation rows below.

## Ensemble members

| Model | RMSE / MAE | Weight / fit |
| --- | --- | --- |
| Keras MLP — BEST | 0.4574 / 0.2795 | 25.94%; 110.6 s; 49 epochs |
| Random Forest | 0.4726 / 0.2889 | 25.11%; 19.5 s |
| LightGBM | 0.4741 / 0.2903 | 25.03%; 20.3 s |
| XGBoost | 0.4963 / 0.2947 | 23.91%; 28.1 s |

RMSE penalizes large errors more strongly; MAE is mean absolute error. Lower is better for both on the same targets and rows. The table's bar length scales RMSE against the worst member, so a **shorter** bar is better. BEST means lowest reported member RMSE, not best physical diagnosis.

Weights are proportional to 1 / RMSE, then normalized. The saved training metadata reports **24,892 training rows, 2,921 validation rows, 11 input features and 21 targets**. Fit time describes saved training execution, not request latency or real-time throughput.

---PAGE---
# 14. Actual reading vs. twin's healthy expectation

![Captured trace for engine E81, sensor 15. Both series are in the API's offset-corrected, mode-scaled coordinates.](figures/twin_trace.png)

| Graph element | Explanation |
| --- | --- |
| Engine / sensor selectors | Choose one engine and one informative sensor. Changing engine resets the automatic sensor choice. Clicking a fidelity-table row also selects an engine. |
| X-axis | Recorded operating cycle for that engine. |
| Y-axis | Per-mode standardized sensor units, offset-corrected against the engine's early-life baseline. Values are not raw °C, rpm or pressure unless separately converted. |
| Orange actual line | Reconstructed actual value = displayed expectation + stored residual. |
| Cyan dashed expectation | What the frozen monitoring twin expects under the operating context. It is a reference value, not a remaining-life forecast. |
| Cyan shaded band | Expected value ± the stored member spread. Wider means the twins disagree more. It is not a statistically calibrated 95% prediction interval. |
| Green background window | The API shades through round(last recorded cycle × healthy-region fraction), here cycle 64. It is an approximate early-window annotation, not proof that this individual engine supplied training rows. |
| Red vertical line | First stored anomaly flag, here cycle 150. It marks detector behavior, not independently verified physical fault onset. |

Read the **vertical gap** as the sensor residual in the chart's transformed coordinates. A large or growing gap supports investigating deviation from the reference. Crossing the band alone is not the backend's full anomaly decision rule.

At E81's final cycle 213, expectation is −46.2952, displayed actual −44.3066 and residual **+1.9887**. The absolute level reflects the endpoint's normalization and correction; report the unit and gap rather than interpreting −46 as a physical temperature.

---PAGE---
# 15. Trace statistics and residual gates

## Four statistics beneath the sensor trace

| Tile | Formula and interpretation |
| --- | --- |
| Residual in fitted window | Mean absolute residual among the returned trace points at or before healthy_until_cycle. It is an early-window diagnostic, not the model's held-out RMSE. |
| Residual at last cycle | Absolute residual at the latest available nonmissing trace point. A value of 1.99 means 1.99 transformed sensor units from expectation. |
| Growth factor | Last absolute residual / early-window mean absolute residual. 4× means the final gap is four times that early mean, not four times the failure risk. Missing or zero baseline prevents a meaningful ratio. |
| Warning lead time | Last recorded cycle − first anomaly-flag cycle. E81: 213 − 150 = **63 cycles**. On truncated test histories, this is time until the end of observation, not time until actual failure. |

## Residual acceptance gates and the header badge

The frontend recomputes three displayed gates from the response. The header badge counts how many pass these specific UI checks.

| Gate | Captured result | UI rule |
| --- | --- | --- |
| Residuals centred on zero | Bias 0.02850 — fails | Absolute mean residual bias < 0.000001. A systematic offset would affect downstream residual interpretation. |
| Healthy spread stays bounded | 0.0953% — passes | Exceedance fraction ≤ 2% beyond the ±3.5 z ceiling. This is an aggregate across the evaluated healthy residual values. |
| Residual rises with age | 79 / 93 = 84.95% — fails | At least 95% of checked engines have larger late-life mean residual than healthy-window mean residual. |

The captured page therefore shows **1/3 residual gates passed** under its hardcoded rules, even though the backend response also contains `validation.passed: true`. Those statuses represent different acceptance checks; do not turn either into a blanket “production validated” claim.

The gate panel was checked on **3,798 healthy rows out of 12,812**. “Healthy” in a saved report is the benchmark's selection rule, not an operator-confirmed statement that every included component was fault-free.

---PAGE---
# 16. Sensor spread and deviation patterns

## Residual spread by sensor

![Healthy-reference 99th-percentile absolute z-score for the nonzero-p99 channels.](figures/sensor_spread.png)

The frontend draws this as horizontal bars: sensor name on Y, **99th percentile |z|** on X. The red reference line is the configured ceiling, **3.5**. This PDF rotates the bars for a readable overview; the statistic is the same.

Each value summarizes the large end of that sensor's **healthy-row** residual distribution, not the latest reading. Values above the ceiling suggest checking normalization, reference scale, quantization or model fit. They do not identify a confirmed failure by themselves.

The capture excludes four zero-p99 channels from the picker. Some older UI prose discusses particular problematic sensors; check the current bars rather than assuming those channels still exceed the ceiling.

## Deviation pattern across the fleet

The donut divides all scored cycles into **No deviation**, **Narrow** and **Broad** residual patterns. Slice angle and percentage represent row counts, not confidence or component contribution.

| Pattern | Captured rows | Interpretation |
| --- | --- | --- |
| No deviation | 10,841 — 84.6% | No residual channels meet the comparison-stage deviation rule for that row. |
| Narrow | 1,918 — 15.0% | A small set of channels deviates; a localized pattern for investigation. |
| Broad | 53 — 0.4% | More channels deviate together; a wider pattern for investigation. |

The subtitle's **mean deviating sensors** averages the number of deviating channels across rows. A sensor counts as deviating when **|z| > 2**; a row is Broad when at least **35% of available residual-z channels** deviate, Narrow when some but fewer than 35% deviate, and No deviation when none do. These patterns need not match anomaly severity or the condition-consensus decision.

---PAGE---
# 17. Twin fidelity table and context

## Per-engine twin fidelity

This table ranks saved engine summaries. Search matches the typed digits within the pooled engine ID. Sort options are peak residual, mean residual, sensors deviating and life length; each sorts descending. At most **12 rows** are shown, so the visible table is not the full fleet.

| Column | What it means |
| --- | --- |
| Engine | A pooled engine ID, rendered with its subset when possible. With offset step 10,000, ID 30249 means FD004 engine 249; it does not mean 30,249 observed engines. |
| Life | Last recorded cycle number. A truncated history's last cycle is not the engine's known total lifetime. |
| Mean residual | Average over cycles of each row's mean absolute sensor residual. It summarizes all channels, rather than just the selected trace sensor. |
| Peak residual | Maximum over cycles of the row mean absolute sensor residual. This is not the largest single-sensor residual. |
| Worst cycle | Cycle with maximum mean absolute residual z. This uses standardized residuals and can differ from the peak raw-residual cycle. The shown percentage is worst cycle / last recorded cycle × 100. |
| Dominant sensor | Sensor with largest absolute z at the worst cycle. It is not automatically the root cause. |
| Pattern | Most frequent residual-pattern category across the engine's stored rows. |

Clicking a row selects that engine in the main trace. The default trace is deliberately selected from highly deviating engines whose dominant sensor has a reasonable residual scale, so the opening chart is **not a random or representative engine**.

## “How the twin produces a residual” process cards

**Fit on healthy only** describes commissioning/training selection. **Predict per regime** describes operating-context inputs and sensor targets. **Fuse four members** describes inverse-RMSE fusion and retained spread. **Centre and standardise** describes the fixed reference used to interpret deviations. These are process explanations, not four additional performance measurements.

## Operating context

The four values are number of **currently represented** operating modes, PSI drift, drift severity and scored engine count. Here they are **1**, **7.928**, **major**, and **93**. The six-regime training/model subtitle and this one-mode current-data count describe different populations. This panel reads **/integration/fleet-modes**; the engine count comes from the twin summary. Verify their scopes when changing uploads.

---PAGE---
# 18. AI Reasoning & Explainability

| Headline card | Captured value | Meaning |
| --- | --- | --- |
| Explained Alerts | 100 | Alert-feed rows with nonempty explanation_text. The feed is capped, so this is not the total 1,193 explained fault cycles. |
| Fault Anomalies | 1,193 | Historical rows classified fault in the dashboard, not independently confirmed mechanical failures. |
| Distinct Root Causes | 6 shown from gate report | The tile prefers evaluate_reasoning.n_distinct_causes. The full reasoning distribution currently has eight labels; the reports are not synchronized. |
| Hard Causal Claims | 0 | Count reported by the automated reasoning gate. Zero means no checked record asserted a hard causal claim; it does not establish diagnosis accuracy. |

## Root Cause Attribution — Fleet

![Full reasoning-report distribution: eight attributed causes over 1,193 fault-cycle rows.](figures/root_causes.png)

The graph counts attributed fault-cycle rows on X and cause labels on Y. It prefers the full reasoning report and falls back to the dashboard's top-five field if unavailable. The leading 386 HPT rows represent **32.4%** of 1,193 attributed rows. Repeated cycles may describe the same engine and underlying event.

## Anomaly Breakdown and “How deviations evolve”

The Total / Genuine faults / Operational tiles repeat dashboard counts **1,193 / 1,193 / 0**. The evolution graph counts reasoned rows by gradual, abrupt or stable: **811 / 201 / 181**. Bar height is a cycle-row count; the listed percentages divide by all reasoned rows. Stable means no clear recent score trend, not healthy.

---PAGE---
# 19. Selected alert and reasoning chain

## Explained Alerts — select an engine

The page offers buttons for the first **18 explained feed rows**. Each can refer to a historical alert rather than the engine's final cycle. The focused alert falls back to an available explained row when the requested selection has no explained match.

| Selected-alert field | E76 example at cycle 198 |
| --- | --- |
| Alert Level | High: the stored operational urgency. |
| Severity | Critical: anomaly-score-band severity for this alert. This is distinct from alert level and health state. |
| Cycle | 198: the time of this alert, not the engine's latest cycle 205. |
| Evolution | Stable: the anomaly score is not changing materially over the recent reasoning window. |

## Generated Explanation / Why / Pattern / Severity

The narrative combines the selected observation's score, threshold, residual evidence, suspected cause, timing, severity and health evidence. The separate Why, Pattern and Severity cards expose parts of the same explanation. They are explanatory outputs, not additional independent detectors.

Use wording such as: “This observation is most consistent with HPT degradation based on the listed residual pattern.” Do not describe generated text as a confirmed inspection result.

## Reasoning chain — hypotheses considered

Each hypothesis displays **cause**, **support score**, **evidence**, **rule ID**, **subsystem** and **inspection focus**. Candidates are sorted by support; SELECTED marks the highest score. Bar width = min(support × 100, 100).

For the example, HPT support is **0.5628**, LPT **0.3584** and bleed-system anomaly **0.3438**. Support is a heuristic rule score, not a posterior fault probability; scores need not sum to 1. Alternative hypotheses remain visible because similar sensor patterns can have several explanations.

## Timing tiles

**Onset cycle** is the reasoning window's detected onset, **179** here. **Persisted** is the reasoning duration, **19 cycles**. **Trend slope** is recent anomaly-score change per cycle, approximately **+0.000001**. **Rule support** is the winning heuristic, **0.56**.

Reasoning-window duration 19 differs from anomaly persistence 42 on the same alert. One is temporal analysis within a recent reasoning window; the other counts consecutive anomaly flags. Neither is an independently measured physical fault age.

---PAGE---
# 20. Subsystems, sensors and case memory

## Where the deviation sits — engine anatomy

The engine diagram lays out Fan → LPC → HPC → Combustor → HPT → LPT, with bypass duct and bleed-air sections. Each section's color follows its attributed status: nominal, elevated or degraded. Hover details show contribution share, member sensors and mean residual z.

Contribution is the subsystem's share of the attributed residual evidence, not the probability that the subsystem has failed. The diagram's brightness/opacity is scaled relative to the largest subsystem contribution, so color intensity is not a universal percentage scale.

## Root Cause and Contributing Sensors

The Root Cause card names the selected hypothesis, its subsystem, heuristic support, recommended action and inspection focus. An inspection recommendation is guidance generated from that hypothesis, not a scheduled maintenance task.

The sensor bars show up to **six** ranked sensor residual z magnitudes. X is effectively relative bar length; each bar is scaled to the **largest |z| in this alert**, not to a fixed failure threshold. Numeric labels show |z|, so consult signed evidence when direction matters.

For E76 at cycle 198, sensor 14 has **|z| ≈ 4.96** and sensor 9 **4.48**. A bar at 100% means “largest in this list,” not “100% causal contribution.” If detailed records are absent, sensor-name chips provide a fallback without numeric bars.

## Agreement cards

**Model Agreement** and **Detector Agreement** multiply their 0–1 values by 100 for display on this page. **Reliability** remains a 0–1 number and **Twin Spread** remains model-unit dispersion. At the example alert these are **86.1%**, **100.0%**, **1.0000** and **0.1387**.

Component confidence uses normalized weights: 100 × (0.45 × model agreement + 0.35 × detector agreement + 0.20 × data quality), within configured bounds. With only one detector available, detector agreement can default to 1.0; it does not demonstrate independent corroboration.

## Case memory

When a match exists, the panel shows remembered-case ID, similarity × 100%, number of prior appearances and whether an operator previously rejected the pattern. Similarity measures resemblance, not diagnostic accuracy. “Never rejected” is not equivalent to “confirmed correct.” The example alert has **no close memory match**; its conclusion rests on rule evidence.

---PAGE---
# 21. Local SHAP and health evidence

## Why the monitoring twin expected this reading

![One of the three computed-SHAP alerts in the captured feed. Signed contributions explain the monitoring expectation.](figures/local_shap.png)

The horizontal axis is **signed SHAP contribution in mode-scaled sensor units**; the vertical axis names input-feature groups. Cyan positive bars raise the expected reading. Amber negative bars lower it. The plot explains **one selected sensor's monitoring-reference prediction at one engine/cycle**.

The displayed identity is baseline + signed contributions = expected reading. The baseline includes the fixed commissioning offset. Operating settings and mode are explained together; cycle age is a separate group. The axis is not anomaly probability, sensor damage or causal effect.

SHAP availability is sampled. The capture reports **128 computed rows out of 1,193 explained rows**, about **10.7%**; three of the returned 100 alerts have computed SHAP. “Not sampled” means the explanation was not attempted for that alert, not zero feature influence. Disabled and failed/unavailable states have different messages.

## Health-index evidence

Below the SHAP panel, the health evidence reconstructs how residual deviation became a health index and score. For final-cycle E76, the recorded evidence states:

| Quantity | Captured value and role |
| --- | --- |
| Weighted deviation / healthy reference | 1.7874 / 0.3237. Their ratio above the reference drives decay. |
| Raw HI | 32.2897, from 100 × exp(−0.25 × excess), where excess = max(deviation / reference − 1, 0). |
| Current term + history term | 6.4579 + 26.9081 = **33.3660 HI**, using EWMA alpha 0.20. |
| Trend adjustment / health score | −1.0978 → **32.2682**, classified Critical using its generation-time boundaries. |

Top sensor shares describe the **weighted deviation before nonlinear decay**. They are not additive percentage-point causes of the final HI. The reconstruction is arithmetic fidelity to the implemented calculation; it is not validation of physical health truth.

---PAGE---
# 22. Reasoning validation and graph

## Reasoning & Explainability Validation

| Tile / metric | Meaning and captured interpretation |
| --- | --- |
| Coverage | 100% of the fault rows checked by the reasoning gate received a cause. That gate counts 641 rows; it is not synchronized with the 1,193-row reasoning report. |
| Narrative completeness | 100% of 1,193 explained rows have Why, Which sensors, What pattern and How severe fields. Presence is not a correctness score. |
| Consistency | 100% on the automated sensor-list/record checks. This establishes agreement between stored outputs, not independent fault validation. |
| Memory match rate | The gate shows 100%, while the reasoning rollup reports zero matches. Treat this as mixed saved-report provenance, not proof of perfect case recognition. |
| Local SHAP coverage / failed | 128 computed, zero failed attempts, 10.7% coverage. “Zero failed” does not mean SHAP was computed for every row. |
| Health reconstruction coverage | 100% of the explained rows received numeric health evidence. |
| Hard causal claims / gate badges | Zero reported hard causal claims; badges expose named automated acceptance checks. Passing checks cover output structure and numeric consistency, not mechanically confirmed diagnoses. |

## Subsystem Dependency Graph / Fault Propagation — origin: …

The graph has **30 nodes and 31 edges** in the capture: subsystem nodes, sensor nodes and configured relationships. Node position is a visualization layout, not time, distance, probability or measured severity.

When a suspected root cause matches a subsystem, the graph highlights that origin and reachable downstream sections. The legend distinguishes suspected origin, potentially affected downstream nodes and other nodes. Hover exposes node emphasis. Edges encode configured subsystem couplings or sensor membership; they are not empirically established causal effects for this alert.

The title “Fault Propagation” means a **possible path according to the dependency graph**. It does not prove that damage actually propagated along those edges.

## Recent Explained Alerts

Up to **eight feed rows** show engine, cycle, severity, root cause and a shortened narrative. Clicking selects an engine for inspection. These are recent/severe entries in the capped feed; they are not eight unique engines, eight confirmed failures or the complete history.

---PAGE---
# 23. A practical explanation for your demo

## Start with the fleet and state the population

“This screen contains 93 engines and 12,812 recorded operating-cycle rows. In snapshot mode, each engine contributes only its latest observation. The mean HI is 73.6; 34 engines are Healthy and eight are Critical. In lifetime mode the mean rises to 92.0 because it includes the earlier healthy operation.”

## Select an engine and explain condition versus evidence

“For engine E76 at cycle 205, HI is 33.37 and the backend labels the engine Critical. Recent deterioration is 0.732 health points per cycle. The anomaly score is 1.0 against the stored threshold of 0.5. That is strong anomaly evidence on this scale; it is not a 100% probability of failure.”

## Explain the twin chart before discussing the diagnosis

“The dashed cyan line is the frozen healthy expectation. The orange line is the actual reading expressed in the same corrected sensor units. Their gap is the residual. The band describes member disagreement, and the first-flag marker shows when the detector raised an anomaly. This is an anomaly-reference view.”

If switching to the default E81 sensor-15 trace, name the changed engine explicitly. Its last recorded cycle is 213, its first stored flag is at 150, and the UI calls the 63-cycle difference warning lead time. That is a lead to the end of this recorded history.

## Explain the suspected cause and alternatives

“For the historical E76 alert at cycle 198, the highest-supported hypothesis is HPT degradation, with heuristic support 0.56. Sensor 14 and sensor 9 carry large standardized deviations. Alternative LPT and bleed-system hypotheses remain visible. The recommendation is an HPT inspection, subject to an analyst's review.”

## State what the benchmark establishes

“On the saved reused official benchmark, AUC is 0.9997 for healthy-versus-critical RUL condition proxies. Healthy-row false alarms are about 0.16%, and all 159 engines that reached critical RUL had a qualifying detection. These results are retrospective condition-proxy evidence, not independently confirmed physical fault accuracy.”

## Finish with an analyst action

Review context drift, persistence and residual evidence, then record Correct, False Alarm or Missed only when there is real supporting evidence. Feedback is persisted and can adapt future thresholds. The embedded queue prioritizes inspections; it does not itself execute or optimize maintenance.

---PAGE---
# 24. Questions you may be asked

| Question | Accurate answer |
| --- | --- |
| Is HI 80 equal to 80% accuracy? | No. It is a derived health-index value on a 0–100 scale. |
| Is remaining health the same as remaining cycles? | No. It is current HI relative to that engine's best observed HI. |
| Why is confidence high on an unhealthy engine? | Confidence describes agreement and input quality. Models can agree that an engine is deviating. |
| Does anomaly score 1 mean failure is certain? | No. The score is bounded and can saturate. It is not a calibrated failure probability. |
| Why are the four detector bars zero? | Those legacy fields are unavailable in the captured active configuration; some charts replace missing values with zero. Condition consensus supplies the fused score. |
| Why do eight Critical engines differ from three Critical Engines? | The Health page counts health-state Critical; the Anomaly page counts anomaly-severity critical. |
| Why does the total exceed the alert feed? | The total counts all qualifying historical rows; the backend returns at most 100 feed rows, and tables display smaller slices. |
| Why do 641 and 1,193 both appear? | Different saved evaluation/stage outputs are being displayed together. They are not established as the same synchronized run and policy. |
| Does a low snapshot HI prove poor modeling? | No. It can reflect a degraded end-of-history observation. Check prediction evidence separately, including its population and target definition. |
| Is r or ρ the same as R²? | No. r is linear correlation, ρ is rank correlation, and R² measures predictive error relative to a mean-target baseline. Do not swap their labels. |
| Where is the earlier forecasting R²? | It belongs to the separate raw-sensor forecast evaluation. The Digital Twin page described here reads healthy-reference artifacts and does not currently show that R² as a headline card. |
| Are the AI causes proven? | No. They are rule-supported hypotheses from residual evidence and a configured dependency graph. Inspection or independent labels are needed to validate physical cause. |

---PAGE---
# 25. Current UI qualifications to remember

These observations explain the implementation as captured. Creating this document did not change models, thresholds, scores or frontend behavior.

- **Four-detector text is older than the active configuration.** The fifth backend detector, condition consensus, has full weight. Legacy detector charts do not display it.
- **Thresholds have a time basis.** Stored alert threshold 0.5 and adapted live threshold 0.5238 refer to different decision times. The p99 caption is not the full consensus-calibration rule.
- **Health-state labels use adjusted score.** The “HI < 20” shorthand and graph line are not a literal reclassification of every displayed HI value. Generation-time evidence can also predate current persisted boundaries.
- **Some quality tiles are unavailable.** The current report lacks the exact fields needed by Within-engine ρ, Fleet ranking ρ, the bias-quartile graph and severity mean-RUL graph. Dashes and empty states should remain visible as unavailable information.
- **Some prose contains older fixed claims.** The displayed 0.98 within-engine statement, near-perfect monotonicity language, “one flag in five,” and some run-to-failure wording should not override current data. Official test histories may be truncated.
- **Saved reports are not one atomic snapshot.** The reasoning rollup, reasoning gate, anomaly gate and dashboard disagree on some row counts and memory statistics. Their values must be qualified by report and population.
- **Frontend and backend gate policies differ.** The frontend's residual rules yield 1/3 passes in the capture despite the backend passed flag. The Ensemble RMSE field is a weighted average of member errors, not a measured fused-prediction RMSE.
- **The green trace window is an annotation.** It is computed from a fraction of the recorded history. It is not proof of individual-engine training membership or exact startup readiness.
- **A queue entry is not an unresolved work order.** The feed contains capped historical alert rows, can repeat engines and can include health alerts outside fault-anomaly rows.
- **Confidence is not independent corroboration.** Detector agreement can be 1.0 by fallback when fewer than two detectors are available. Similar model outputs can still share a common error.

> These qualifications make the presentation more accurate: explain what each field actually measures, then state what validation would be needed for a stronger claim.

---PAGE---
# 26. Data sources and reproducibility

## Where each page gets its values

| Page / panel | Backend source |
| --- | --- |
| Health and Anomaly headline cards; alerts; engine details | GET /anomaly-health/dashboard — full server-side fleet aggregate, latest engine summaries, capped alerts and limited bundled timelines. |
| Selected-engine HI trajectory | GET /anomaly-health/engine/{id}/timeline — used when the bundled timeline is unavailable. |
| Validation and saved fleet distributions | GET /anomaly-health/evaluation/metrics — loads the named saved reports. |
| Current detector principles and weights | GET /anomaly-health/detector-config. |
| Health context-drift panel | GET /anomaly-health/context-status. |
| Twin Operating context panel | GET /integration/fleet-modes, plus scored engine count from the twin summary. |
| Twin headline, members, gates, patterns and engine table | GET /anomaly-health/twin/metrics — digital_twin, evaluate_digital_twin and twin_comparison saved artifacts. |
| Sensor trace | GET /anomaly-health/twin/trace — joins stored prediction and residual information for the selected sensor and engine. |
| Detailed reasoning; dependency graph | GET /anomaly-health/reasoning/{id}; GET /anomaly-health/reasoning/sensor-graph. |
| Feedback state and operator action | GET /anomaly-health/feedback/summary; GET /anomaly-health/feedback/thresholds; POST /anomaly-health/feedback when the user submits a verdict. |

## Source files checked

Frontend: HealthMonitoring.tsx, AnomalyDetection.tsx, DigitalTwin.tsx and AIReasoning.tsx; shared AnomalyValidationPanels, FeedbackControl, EngineAnatomy and SensorGraph; API types, hooks and sidebar ownership.

Backend: Anomaly_Health_Monitoring routes and dashboard service; health-index calculator, trend tracker, score engine, classifier and alert engine; confidence/agreement and early-warning calculations; saved reports and persisted threshold metadata.

The accompanying **frontend_evidence.json** records the API capture timestamp, returned values and hashes for selected source/configuration files. **validation.json** records the generated PDF's hash, page map and layout checks. The Markdown copy and build script allow the guide to be refreshed when the frontend changes.

Values were captured through local read-only GET requests on 18 October 2026. Saved artifacts can have earlier generation dates, and requests were sequential rather than transactional. This guide documents that captured implementation; it does not certify a new evaluation run.
