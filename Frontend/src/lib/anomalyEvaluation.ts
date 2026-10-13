import type { AnomalyEvaluationResponse, AnomalyEvalSplit } from "./types";

/** Accept both legacy reports and the versioned offline report envelope. */
export function readAnomalyHoldout(response?: AnomalyEvaluationResponse) {
  const report = response?.metrics?.holdout_anomaly_evaluation;
  const metrics = report?.metrics ?? report;
  return {
    test: metrics?.test ?? null,
    train: metrics?.train ?? null,
    perSubset: metrics?.test_per_subset ?? null,
  };
}

/** Critical-engine detection must use engines that actually reached critical RUL. */
export function criticalEngineDetection(split: AnomalyEvalSplit): number | undefined {
  return split.condition_proxy_metrics?.engine_events?.detection_rate_of_reaching
    ?? split.detection_rate_of_reaching;
}
