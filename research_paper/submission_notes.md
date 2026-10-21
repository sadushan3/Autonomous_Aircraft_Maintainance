# Before submitting

The manuscript is a complete evidence-based draft in the supplied IEEE conference format. Acceptance depends on the selected conference's scope, originality expectations, experimental standards, and review. Formatting cannot ensure acceptance.

## Information still needed

1. Confirm the author's institution, department, city, and country, or explicitly choose independent-researcher affiliation. These are currently omitted rather than guessed.
2. Name the conference and confirm its page limit, paper size, submission deadline, review anonymity, and any required declarations. The supplied template alone does not establish those rules. The current source uses the template's default US Letter conference layout.
3. Confirm authorship, contributions, funding, conflicts, and data/code availability statements where the venue requires them. No funding or additional author has been assumed.

## Evidence that would most improve the submission

- Reserve genuinely unseen engine trajectories before any further selection and evaluate the final frozen method once. Current results are retrospective because the official C-MAPSS benchmark was inspected during development.
- Obtain independently assessed fault labels and onset times for claims about fault detection, diagnostic accuracy, or advance warning. RUL >= 125 and RUL <= 30 are condition proxies, with the middle band excluded from binary metrics.
- Compare single detectors and leave-one-detector-out ensembles after calibration on the same training roles. The existing precision–recall curves compare stored rankings; they are not a controlled retraining ablation.
- Investigate the current fusion operating point using development engines only. The deployed four-detector F1 is 0.7426 and recall 0.6321; the fixed sensor baseline has F1 0.8976 and recall 0.8908. The paper reports this limitation. Selecting a new cutoff on official test labels would increase benchmark reuse.
- Validate explanations with domain experts and test interrupted telemetry, missing commissioning prefixes, and engines already degraded at commissioning if those are intended deployment conditions.

## Results that must stay distinct

| Evidence | Population | Interpretation |
|---|---|---|
| Four-detector anomaly metrics | 51,806 healthy/critical rows after readiness exclusions | Retrospective condition-proxy discrimination |
| Continuous anomaly/event analysis | 83,788 ready rows, 690 engines | Includes intermediate conditions; no verified onset labels |
| Official-test health evaluation | 104,897 rows, 707 engines | Recomputed health/RUL association and defined monotonicity |
| Health plot | Current batch: 12,812 rows, 93 engines | Implementation illustration, not another independent test |
| Explanation evaluation | 1,193 current-batch explanation records | Automated consistency and numerical fidelity; 128 local SHAP records |

The earlier consensus model's approximately 99.72% condition accuracy belongs to a different, superseded detector bundle. It has not been substituted for the active four-detector results. No results from the Predictive Maintenance or Maintenance Supervisor components are used.

The manuscript's explanation values originate from a saved current-batch evaluation. The paper build independently recomputes anomaly labels, counts, metrics, confidence intervals, score curves, and official-test health using the current frozen reference. It does not rerun training or claim a new prospective experiment.

The older saved health summary has pooled correlation 0.4591615 and monotonicity 0.9463. Recomputing from the four-detector experiment's official residual inputs gives 0.4601123 and 0.9477. The manuscript consistently uses the latter values, with the original summary and differences retained in `paper_evidence.json`. The difference is not presented as a new independent experiment or an improvement claim.
