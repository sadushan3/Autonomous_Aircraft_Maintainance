/**
 * Types mirroring the FastAPI backend's actual response shapes.
 *
 * These were derived from live responses, not guessed: the fused decision row
 * carries ~60 columns produced by joining the RUL, Failure-Risk and Anomaly
 * exchange CSVs, so `DecisionRow` is intentionally permissive at the edges
 * (`[key: string]: unknown`) while naming everything the UI actually reads.
 */

/** One fused row from the Maintenance Supervisor's decision output. */
export interface DecisionRow {
  engine_id: number | string;
  cycle: number;
  fd_subset: string;

  // --- RUL Lifecycle component ---
  predicted_rul: number | null;
  lower_bound: number | null;
  upper_bound: number | null;
  uncertainty_width: number | null;
  lifecycle_state: string | null;
  degradation_speed: number | null;
  degradation_acceleration: number | null;
  model_used: string | null;

  // --- Failure Risk component ---
  risk_10: number | null;
  risk_30: number | null;
  risk_50: number | null;
  uncertainty_10: number | null;
  uncertainty_30: number | null;
  uncertainty_50: number | null;
  threshold_10: number | null;
  threshold_30: number | null;
  threshold_50: number | null;
  predicted_fail_in_10: boolean | null;
  predicted_fail_in_30: boolean | null;
  predicted_fail_in_50: boolean | null;
  control_risk: number | null;
  risk_velocity: number | null;
  risk_acceleration: number | null;
  risk_state: string | null;

  // --- Anomaly & Health component ---
  health_index: number | null;
  health_state: string | null;
  anomaly_score: number | null;
  is_anomalous: boolean | null;
  anomaly_severity: string | null;
  anomaly_type: string | null;
  root_cause: string | null;
  uncertainty_score: number | null;
  health_trend: number | null;
  remaining_health_pct: number | null;
  early_warning_score: number | null;

  // --- Maintenance Supervisor fusion ---
  raw_model_decision: string | null;
  final_decision: string | null;
  override_applied: boolean | null;
  override_reason: string | null;
  priority: string | null;
  maintenance_urgency: string | null;
  recommended_action: string | null;
  confidence_score: number | null;
  confidence_level: string | null;
  entropy_score: number | null;
  conflict_score: number | null;
  requires_human_review: boolean | null;
  review_reason: string | null;

  [key: string]: unknown;
}

export interface DashboardResponse {
  total: number;
  returned: number;
  limit: number;
  offset: number;
  truncated: boolean;
  rows: number;
  decisions: DecisionRow[];
}

export interface LeaderboardResponse {
  metric: string;
  rows: number;
  engines: DecisionRow[];
}

export interface MaintenanceQueueResponse {
  rows: number;
  queue: DecisionRow[];
}

export interface FleetModesResponse {
  mode_distribution: Record<string, number>;
  drift: {
    drift_detected: boolean;
    psi: number;
    threshold: number;
    severity: string;
    reference_proportions: Record<string, number>;
    current_proportions: Record<string, number>;
    per_mode_shift: Record<string, number>;
    setting_drift: Record<string, number>;
    reason: string;
  };
}

export interface FeatureRoutingResponse {
  required_upload_columns: string[];
  carried_when_present: string[];
  routes: Record<
    string,
    { id_column: string; features: string[]; scaling: string }
  >;
  not_a_routing_target?: string[];
}

export interface CompareResponse {
  [engineId: string]: DecisionRow[];
}

/** One physical engine section's share of an anomaly's deviation. */
export interface SubsystemAttribution {
  subsystem: string;
  subsystem_label: string;
  contribution: number;
  sensors: string[];
  mean_residual_z: number;
  status: "degraded" | "elevated" | "nominal" | string;
}

/** One engine's rollup from the Anomaly & Health dashboard. */
export interface AnomalyEngine {
  engine_id: number | string;
  latest_cycle: number;
  total_cycles: number;
  health_index: number | null;
  health_state: string | null;
  health_trend: number | null;
  remaining_health_pct: number | null;
  anomaly_score: number | null;
  anomaly_flag: boolean | null;
  anomaly_type: string | null;
  severity: string | null;
  alert_level: string | null;
  confidence_score: number | null;
  root_cause: string | null;
  top_sensors: string[] | string | null;
  operating_mode: number | null;
  explanation_text: string | null;
  twin_spread_mean: number | null;
  deterioration_rate: number | null;
  reliability_score: number | null;
  isolation_forest_score: number | null;
  mahalanobis_score: number | null;
  lstm_ae_score: number | null;
  residual_rule_score: number | null;
  threshold_used: number | null;
  anomaly_persistence_cycles: number | null;
  early_warning_score: number | null;
  subsystem_attribution: SubsystemAttribution[] | null;
  which_sensors: string[] | string | null;
  why: string | null;
  what_pattern: string | null;
  how_severe: string | null;
  shap_top_features: unknown[] | null;
  shap_status?: string | null;
  shap_sensor?: string | null;
  shap_explained_output?: string | null;
  shap_base_value?: number | null;
  shap_prediction?: number | null;
  shap_commissioning_offset?: number | null;
  health_explanation?: string | null;
  health_evidence?: Record<string, unknown>;
  sensor_evidence?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

/** One explained alert, carrying the generated diagnostic narrative. */
export interface AnomalyAlert {
  engine_id: number | string;
  cycle: number;
  alert_level: string | null;
  severity: string | null;
  health_state: string | null;
  root_cause: string | null;
  recommended_action: string | null;
  confidence_score: number | null;
  anomaly_score: number | null;
  explanation_text: string | null;
  twin_spread_mean: number | null;
  deterioration_rate: number | null;
  reliability_score: number | null;
  uncertainty_score: number | null;
  model_agreement: number | null;
  detector_agreement: number | null;
  evolution_pattern: string | null;
  isolation_forest_score: number | null;
  mahalanobis_score: number | null;
  lstm_ae_score: number | null;
  residual_rule_score: number | null;
  threshold_used: number | null;
  anomaly_persistence_cycles: number | null;
  early_warning_score: number | null;
  subsystem_attribution: SubsystemAttribution[] | null;
  which_sensors: string[] | string | null;
  why: string | null;
  what_pattern: string | null;
  how_severe: string | null;
  shap_top_features: unknown[] | null;
  shap_status?: string | null;
  shap_sensor?: string | null;
  shap_explained_output?: string | null;
  shap_base_value?: number | null;
  shap_prediction?: number | null;
  shap_commissioning_offset?: number | null;
  health_explanation?: string | null;
  health_evidence?: Record<string, unknown>;
  sensor_evidence?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

/** Anomaly & Health component's own dashboard payload. */
export interface AnomalyDashboardResponse {
  status?: string;
  fleet: {
    total_engines: number;
    total_rows: number;
    /**
     * Mean health index at each engine's FINAL observed cycle -- a snapshot of
     * "how the fleet stands now", not an average over history. On C-MAPSS
     * train units, which all run to failure, this is necessarily low.
     * Contrast `lifetime_mean_health_index`.
     */
    mean_health_index: number;
    engines_healthy: number;
    engines_degrading: number;
    engines_warning: number;
    engines_critical: number;
    engines_failure_risk?: number;
    total_anomalies: number;
    fault_anomalies: number;
    operational_anomalies?: number;
    mean_confidence?: number;
    top_root_causes?: Record<string, number>;
    /** Mean health index across every row, i.e. over the whole recorded life. */
    lifetime_mean_health_index?: number;
    lifetime_rows_healthy?: number;
    lifetime_rows_degrading?: number;
    lifetime_rows_warning?: number;
    lifetime_rows_critical?: number;
    lifetime_rows_failure_risk?: number;
    [key: string]: unknown;
  };
  engines: AnomalyEngine[];
  alerts: AnomalyAlert[];
  /** Per-engine time series, keyed by engine id. */
  timeline?: Record<string, AnomalyTimelinePoint[]>;
  [key: string]: unknown;
}

/** One cycle of one engine's recorded history. */
export interface AnomalyTimelinePoint {
  cycle: number;
  health_index: number | null;
  /** Ground truth: 100 * clip(RUL, 0, rul_cap) / rul_cap. */
  health_index_target?: number | null;
  health_state: string | null;
  anomaly_score: number | null;
  anomaly_flag: boolean | null;
  confidence_score: number | null;
  uncertainty_score: number | null;
  twin_spread_mean: number | null;
  model_agreement: number | null;
  detector_agreement: number | null;
  reliability_score: number | null;
  deterioration_rate: number | null;
  evolution_pattern: string | null;
  operating_mode: number | string | null;
}

export interface SensorGraphResponse {
  [key: string]: unknown;
}

export interface ContextStatusResponse {
  [key: string]: unknown;
}

export interface RunFullResponse {
  status: string;
  stages: Record<string, { status?: string; [k: string]: unknown }>;
}

/** Derived fleet-level rollup the Overview page renders. */
export interface FleetSummary {
  engines: number;
  rows: number;
  avgHealth: number;
  avgRul: number;
  /** Engines in the worst two bands: "Critical" plus the bottom "Failure Risk". */
  criticalCount: number;
  /** The "Failure Risk" band alone, so the split stays visible where it matters. */
  failureRiskCount: number;
  warningCount: number;
  healthyCount: number;
  anomalyCount: number;
  reviewCount: number;
  decisionCounts: Record<string, number>;
  healthStateCounts: Record<string, number>;
  severityCounts: Record<string, number>;
  avgConfidence: number;
  subsets: string[];
}

/** One horizon's adaptive risk-threshold state. */
export interface HorizonThreshold {
  fd_subset: string;
  horizon: string;
  original_threshold: number;
  current_threshold: number;
  trend: string;
  false_alarm_count: number;
  confirmed_count: number;
  feedback_count: number;
  false_alarm_rate: number | null;
  last_updated: string | null;
}

export interface ThresholdStatusResponse {
  fd_subset: string;
  horizons: Record<string, HorizonThreshold>;
  [key: string]: unknown;
}

/** One row of the champion-vs-baselines benchmark. */
export interface BaselineRow {
  model: string;
  accuracy: number;
  macro_f1: number;
  false_critical_rate: number;
  missed_critical_rate: number;
  cost_penalty: number;
  [key: string]: unknown;
}

export interface BaselineCompareResponse {
  status?: string;
  input_path?: string;
  total_samples?: number;
  comparison_table: BaselineRow[];
  [key: string]: unknown;
}

/** One forecast point from the trajectory service. */
export interface TrajectoryPoint {
  cycle: number;
  predicted_rul: number;
  lower_bound: number;
  upper_bound: number;
  uncertainty_width: number;
  degradation_speed: number;
  degradation_acceleration: number;
  lifecycle_state: string;
  risk_10: number;
  risk_30: number;
  risk_50: number;
  risk_state: string;
  [key: string]: unknown;
}

export interface TrajectoryResponse {
  fd_subset: string;
  engine_id: string;
  stride: number;
  window_start_cycle: number;
  points: TrajectoryPoint[];
}

// --------------------------------------------------------------------------
// Predictive Maintenance -- RUL agent
//
// These mirror `app/schemas/Predictive_Maintenance/rul_schema.py` and the
// memory/evaluation router payloads field-for-field. Anything the backend can
// return as absent is optional here rather than defaulted, so the UI shows a
// dash instead of inventing a number.
// --------------------------------------------------------------------------

/** One raw C-MAPSS row, as `POST /rul/predict` expects it. */
export interface EngineSensorRow {
  cycle: number;
  unit?: number;
  fd_subset?: string;
  op_setting_1: number;
  op_setting_2: number;
  op_setting_3: number;
  [sensor: string]: number | string | undefined;
}

/** Response of `POST /predictive-maintenance/rul/predict`. */
export interface RULPredictionResponse {
  fd_subset: string;
  predicted_rul: number;
  lower_bound: number;
  upper_bound: number;
  uncertainty_width: number;
  lifecycle_state: string;
  degradation_speed: number;
  degradation_acceleration: number;
  confidence_collapse_flag: boolean;
  sudden_health_drop_flag: boolean;
  instability_flag: boolean;
  model_used: string;
  fallback_triggered: boolean;
  fallback_allowed: boolean;
  correction_applied: boolean;
  fallback_model_used: string | null;
  primary_predicted_rul: number | null;
  debug?: Record<string, unknown> | null;
}

/** Response of `GET /predictive-maintenance/memory/status`. */
export interface MemoryStatusResponse {
  total_memories: number;
  distinct_engines: number;
  memories_by_fd_subset: Record<string, number>;
  last_updated: string | null;
  /** 0..1; the backend defines this as min(1, total/200). */
  memory_confidence: number;
}

/** One neighbour from `GET /predictive-maintenance/memory/similar`. */
export interface SimilarMemory {
  engine_id?: string;
  fd_subset?: string;
  distance?: number;
  [key: string]: unknown;
}

/**
 * Response of `GET /predictive-maintenance/evaluation/metrics`.
 *
 * `available` is false until someone drops `evaluation_metrics.json` into the
 * metrics folder -- it is exported from the training notebook and was not in
 * the shipped model archive, so PICP/MPIW render as "--" until then.
 */
export interface EvaluationMetricsResponse {
  available: boolean;
  source: string | null;
  data: {
    models?: Array<{
      name?: string;
      fd_subset?: string;
      /** Held-out engines the figures were computed over. */
      engines_scored?: number;
      engines_skipped_short?: number;
      window_size?: number;
      mae?: number;
      rmse?: number;
      r2?: number;
      nasa_score?: number;
      picp?: number;
      mpiw?: number;
      mean_predicted_rul?: number;
      mean_true_rul?: number;
    }>;
    [key: string]: unknown;
  } | null;
  candidate_paths?: string[];
}

/** Payload shape returned by `GET /predictive-maintenance/demo/sample-request`. */
export interface DemoSampleRequest {
  fd_subset: string;
  engine_id?: string;
  sequence: EngineSensorRow[];
  [key: string]: unknown;
}

/** One point from `POST /predictive-maintenance/predict/trajectory`. */
export interface PmTrajectoryPoint {
  cycle: number;
  predicted_rul: number;
  lower_bound: number;
  upper_bound: number;
  uncertainty_width: number;
  degradation_speed: number;
  degradation_acceleration: number;
  lifecycle_state: string;
  confidence_collapse_flag: boolean;
  correction_applied: boolean;
  /** Failure-Risk-owned; null on a RUL-only replay (`include_risk: false`). */
  risk_10: number | null;
  risk_30: number | null;
  risk_50: number | null;
  risk_state: string | null;
}

export interface PmTrajectoryResponse {
  fd_subset: string;
  engine_id: string | null;
  stride: number;
  window_start_cycle: number;
  points: PmTrajectoryPoint[];
}

/** Response of `GET /integration/fleet-summary` -- one row per engine, reduced server-side. */
export interface FleetSummaryResponse {
  totals: { engines: number; rows_total: number; basis: string };
  aggregates: {
    mean_health_index: number | null;
    mean_predicted_rul: number | null;
    mean_confidence_score: number | null;
    mean_anomaly_score: number | null;
    health_state_counts: Record<string, number>;
    final_decision_counts: Record<string, number>;
    anomaly_severity_counts: Record<string, number>;
    lifecycle_state_counts: Record<string, number>;
    risk_state_counts: Record<string, number>;
    anomalous_engines: number | null;
    review_required_engines: number | null;
    subsets: string[];
  };
  engines: DecisionRow[];
}

// --------------------------------------------------------------------------
// Anomaly & Health -- validation, drift and feedback
// --------------------------------------------------------------------------

/** `GET /anomaly-health/evaluation/metrics` -- offline held-out results. */
export interface AnomalyEvaluationResponse {
  available: boolean;
  searched_paths?: string[];
  metrics: {
    holdout_anomaly_evaluation?: AnomalyHoldoutMetrics & {
      schema_version?: string;
      metrics?: AnomalyHoldoutMetrics;
      error?: string;
    };
    /** Reasoning-stage rollups; carries all causes, not the dashboard's top 5. */
    reasoning?: {
      rows_input?: number;
      rows_reasoned?: number;
      engines?: number;
      cause_distribution?: Record<string, number>;
      subsystem_distribution?: Record<string, number>;
      evolution_distribution?: Record<string, number>;
      memory_matches?: number;
    };
    /** Reasoning-stage acceptance gates. */
    evaluate_reasoning?: {
      rows_reasoned?: number;
      total_fault_rows?: number;
      coverage?: number;
      n_distinct_causes?: number;
      n_distinct_subsystems?: number;
      /** Always 0: the reasoner states likelihoods, never demonstrated cause. */
      hard_causal_claims?: number;
      memory_match_rate?: number;
      acceptance?: Record<string, boolean>;
    };
    /** Narrative completeness and consistency over every explained row. */
    evaluate_explainability?: {
      rows_explained?: number;
      completeness?: {
        completeness_fraction?: number;
        missing_by_field?: Record<string, number>;
      };
      consistency?: { consistency_fraction?: number; rows_checked?: number };
      shap_computed?: boolean;
      fidelity?: {
        shap_rows_computed?: number;
        shap_rows_failed?: number;
        shap_coverage_fraction?: number;
        health_coverage_fraction?: number;
        shap_valid_fraction?: number | null;
        health_valid_fraction?: number | null;
      };
      acceptance?: Record<string, boolean>;
    };
    /** The health stage's acceptance gate, on training rows. */
    evaluate_health?: {
      rows?: number;
      engines?: number;
      rul_correlation?: number;
      monotonicity_score?: number;
      monotonicity_threshold?: number;
      mean_health_index?: number;
      passed?: boolean;
    };
    /** Alert-level mix over every scored row, not just the latest cycle. */
    health_alerts?: {
      rows?: number;
      engines?: number;
      alert_distribution?: Record<string, number>;
    };
    /** The anomaly stage's own acceptance report. */
    evaluate_anomaly?: {
      rul_alignment?: {
        available?: boolean;
        fleet_mean_rul?: number;
        fault_mean_rul?: number;
        faults_concentrate_at_low_rul?: boolean;
      };
      gate_behaviour?: {
        available?: boolean;
        total_flagged?: number;
        fault_anomalies?: number;
        operational_anomalies?: number;
        /** Share of raw flags the operating-context gate reclassifies. */
        operational_suppression_rate?: number;
      };
      acceptance?: Record<string, boolean>;
    };
    holdout_test_evaluation?: {
      rows?: number;
      engines?: number;
      fleet_ranking_spearman?: number;
      monotonicity_score?: number;
      within_engine_spearman_median?: number;
      within_engine_spearman_mean?: number;
      "within_engine_frac_above_0.9"?: number;
      rul_correlation?: number;
      health_index_mean?: number;
      health_index_std?: number;
      /** Negative: engines observed longer read lower, independent of health. */
      mean_hi_vs_observed_length?: number;
      baseline_bias_spearman?: number;
      engines_starting_degraded?: number;
      bias_quartiles?: Record<
        string,
        { n?: number; median_start_rul?: number; mean_hi?: number }
      >;
      per_subset?: Record<string, Record<string, number>>;
    };
  };
}

export interface AnomalyHoldoutMetrics {
  train?: AnomalyEvalSplit;
  test?: AnomalyEvalSplit;
  test_per_subset?: Record<string, { auc: number; false_alarm: number; detection: number }>;
}

export interface AnomalyEvalSplit {
  rows?: number;
  engines?: number;
  auc_healthy_vs_critical?: number;
  cohens_d?: number;
  false_alarm_rate_healthy?: number;
  detection_rate?: number;
  detection_rate_of_reaching?: number;
  label_basis?: string;
  condition_proxy_metrics?: {
    available?: boolean;
    label_basis?: string;
    precision?: number;
    recall?: number;
    engine_events?: {
      detection_rate_of_reaching?: number;
      first_alert_rul_median?: number;
      lead_time_note?: string;
    };
  };
  flag_rate_critical?: number;
  overall_flag_rate?: number;
  lead_time_median?: number;
  lead_time_p25?: number;
  lead_time_p75?: number;
  threshold_used?: number;
  severity_mean_rul?: Record<string, number>;
  severity_row_counts?: Record<string, number>;
  severity_summary_note?: string;
}

/** `GET /anomaly-health/context-status` -- operating-mode drift. */
export interface ContextDrift {
  drift_detected: boolean;
  psi: number;
  threshold: number;
  severity: string;
  reason?: string;
  reference_proportions?: Record<string, number>;
  current_proportions?: Record<string, number>;
  per_mode_shift?: Record<string, number>;
  setting_drift?: Record<string, number>;
}

/** `GET /anomaly-health/feedback/summary`. */
export interface FeedbackSummary {
  feedback?: { total?: number; by_verdict?: Record<string, number> };
  memory?: {
    total_memories?: number;
    false_alarm_patterns?: number;
    confirmed_fault_patterns?: number;
  };
}

/** `GET /anomaly-health/feedback/thresholds` -- adaptive fusion threshold. */
export interface FeedbackThresholds {
  scale?: number;
  fusion_threshold?: number;
  base_threshold?: number;
  effective_scale?: number;
  clamped?: boolean;
  reachable_ceiling?: number;
  feedback_applied?: number;
  last_updated?: string | null;
}

/**
 * Response of `POST /predictive-maintenance/failure-risk/predict`.
 *
 * Mirrors `app/schemas/Predictive_Maintenance/failure_risk_schema.py`. These
 * fields belong to the Failure Risk agent and are presented only on its own
 * page -- the RUL Lifecycle dashboard deliberately shows none of them.
 */
export interface FailureRiskPredictionResponse {
  fd_subset: string;
  risk_10: number;
  risk_30: number;
  risk_50: number;
  /** Monte-Carlo dropout standard deviation per horizon; 0 when not requested. */
  uncertainty_10: number;
  uncertainty_30: number;
  uncertainty_50: number;
  threshold_10: number;
  threshold_30: number;
  threshold_50: number;
  predicted_fail_in_10: boolean;
  predicted_fail_in_30: boolean;
  predicted_fail_in_50: boolean;
  control_risk: number;
  risk_velocity: number;
  risk_acceleration: number;
  risk_state: string;
  action_hint_for_supervisor: string;
  model_used: string;
  debug?: Record<string, unknown> | null;
}

/** One member of the digital twin ensemble, from `GET /anomaly-health/twin/metrics`. */
export interface TwinModel {
  name: string;
  rmse: number | null;
  mae: number | null;
  /** Inverse-RMSE fusion weight; sums to 1 across members. */
  weight: number | null;
  fit_seconds: number | null;
  epochs_trained?: number | null;
  train_rows: number | null;
  val_rows: number | null;
  n_features: number | null;
  n_targets: number | null;
  is_best: boolean;
}

/** Per-engine twin fidelity row, from `metrics/twin_comparison.json`. */
export interface TwinEngineSummary {
  unit: number;
  last_cycle: number;
  mean_abs_residual: number;
  max_abs_residual: number;
  mean_deviating_sensors: number;
  worst_cycle: number;
  dominant_sensor_at_worst: string;
  dominant_pattern: string;
}

/**
 * `GET /anomaly-health/twin/metrics` -- the twin's own evidence.
 *
 * Static offline output: training error, the residual acceptance gates and the
 * per-engine fidelity table, none of which the API recomputes.
 */
export interface TwinMetricsResponse {
  available: boolean;
  message?: string;
  summary: {
    rows: number | null;
    engines: number | null;
    n_twins: number;
    sensors_modelled: number | null;
    informative_sensors: number | null;
    operating_modes: number | null;
    n_features: number | null;
    train_rows: number | null;
    val_rows: number | null;
    weighted_mean_rmse: number | null;
    mean_twin_spread: number | null;
    best_member: string | null;
    best_member_rmse: number | null;
    /** Subset names in pooling order, e.g. ["FD001", ... "FD004"]. */
    subsets: string[];
    /** Unit ids are offset by `subset index * this` when subsets are pooled. */
    unit_id_offset_step: number;
  };
  models: TwinModel[];
  validation: {
    passed: boolean | null;
    healthy_rows_checked: number | null;
    total_rows: number | null;
    centring: {
      mean_absolute_bias: number | null;
      per_sensor_mean: Record<string, number>;
    };
    spread: {
      exceedance_fraction: number | null;
      ceiling: number | null;
      per_sensor_p99_abs_z: Record<string, number>;
    };
    growth: {
      engines_checked: number | null;
      engines_with_rising_residual: number | null;
      fraction_rising: number | null;
    };
    coverage: { expected_sensors?: number; missing_sensors?: string[] };
    acceptance: Record<string, unknown>;
  };
  patterns: Record<string, number>;
  mean_deviating_sensors: number | null;
  engines: TwinEngineSummary[];
  /** Sensors the twin can actually model; constant channels are excluded. */
  available_sensors: string[];
}

/** One cycle of an engine's actual-vs-twin trace. */
export interface TwinTracePoint {
  cycle: number;
  /** What the twin says a healthy engine would read here. */
  expected: number;
  /** The real reading, offset-corrected to this engine's own baseline. */
  actual: number | null;
  band_low: number;
  band_high: number;
  /** Disagreement across the four members -- the twin's own uncertainty. */
  spread: number;
  residual: number | null;
  anomaly: number | null;
  anomaly_score: number | null;
}

/** `GET /anomaly-health/twin/trace`. */
export interface TwinTraceResponse {
  available: boolean;
  engine_id: number;
  sensor: string;
  last_cycle: number;
  /** End of the window the twin was fitted on. */
  healthy_until_cycle: number;
  onset_cycle: number | null;
  points_returned: number;
  cycles_total: number;
  stride: number;
  units: string;
  points: TwinTracePoint[];
}

/** `GET /anomaly-health/detector-config` -- the fusion contract, from config. */
export interface DetectorConfigResponse {
  available: boolean;
  detectors: Array<{
    key: string;
    weight: number;
    weight_share: number;
    principle: string | null;
  }>;
  fusion: {
    threshold_percentile: number;
    fallback_threshold: number;
    normalisation: string;
  };
}

/** `GET /anomaly-health/engine/{id}/timeline` -- any engine's trajectory. */
export interface EngineTimelineResponse {
  available: boolean;
  engine_id: number;
  cycles_total: number;
  points_returned: number;
  stride: number;
  points: AnomalyTimelinePoint[];
}

/** One hypothesised cause with the evidence that fired it. */
export interface CandidateCause {
  cause: string;
  likelihood: number;
  subsystem: string;
  evidence: string[];
  rule_id: string;
  inspection_focus: string;
}

/** How much one sensor contributed to an anomaly. */
export interface SensorContribution {
  sensor: string;
  residual?: number;
  residual_z?: number;
  contribution?: number;
  subsystem?: string;
}

/** One cycle of reasoning output, from `GET /anomaly-health/reasoning/{id}`. */
export interface RootCauseRecord {
  engine_id: number;
  cycle: number;
  anomaly_type: string;
  severity: string;
  top_sensors: string[];
  sensor_contributions: SensorContribution[];
  subsystem: string;
  root_cause: string;
  cause_likelihood: number;
  candidate_causes: CandidateCause[];
  evolution_pattern: string;
  onset_cycle: number | null;
  duration_cycles: number;
  trend_slope: number;
  similar_past_anomaly_id: string | null;
  similarity_score: number;
  times_seen_before: number;
  previously_false_alarm: boolean;
  recommended_action: string;
  inspection_focus: string;
  /** Always false: physical couplings are not demonstrated causation. */
  hard_causal_claim: boolean;
}

export interface EngineReasoningResponse {
  engine_id: number;
  records: RootCauseRecord[];
}


// ---------------------------------------------------------------------------
// Autonomous Maintenance Supervisor Agent — operator view
//
// Mirrors `/integration/supervisor/*`. The payload is deliberately one bundle
// per engine: every card on the Supervisor screen is a facet of the same fused
// decision record, so splitting it across requests would let them render
// mutually inconsistent states while they settled.
// ---------------------------------------------------------------------------

/** One engine in the unit picker, at its latest scored cycle. */
export interface SupervisorEngineSummary {
  engine_id: number | string;
  label: string;
  fd_subset: string | null;
  latest_cycle: number;
  first_cycle: number;
  cycles_observed: number;
  final_decision: string;
  /** 0 (continue operation) … 4 (immediate maintenance). */
  severity: number;
  priority: string | null;
  predicted_rul: number | null;
  risk_30: number | null;
  health_index: number | null;
  anomaly_score: number | null;
  requires_human_review: boolean | null;
}

export interface SupervisorEngineCatalog {
  total: number;
  returned: number;
  engines: SupervisorEngineSummary[];
  decisions_written_at: string | null;
}

/** Whether one upstream agent's columns actually carry values on this record. */
export interface UpstreamAgentStatus {
  key: string;
  name: string;
  status: "active" | "no_signal";
  fields_present: string[];
  model_used: string | null;
}

/** The decision head as it actually is — not as the header slot would flatter it. */
export interface SupervisorPolicyState {
  champion_model: string;
  champion_label: string;
  training_regime: string;
  /** Always false: no candidate head is trained by reinforcement learning. */
  is_reinforcement_learning: boolean;
  target_note: string;
  val_macro_f1: number | null;
  candidates: string[];
  decision_classes: string[];
  feedback_entries: number;
  continuous_learning_active: boolean;
  safety_thresholds: Record<string, number>;
}

/** One failure-risk horizon. The unit is cycles, not hours. */
export interface RiskHorizon {
  key: string;
  label: string;
  cycles: number;
  value: number | null;
  uncertainty: number | null;
  threshold: number | null;
  predicted_failure: boolean | null;
}

export interface SupervisorRiskBlock {
  headline: RiskHorizon;
  horizons: RiskHorizon[];
  risk_state: string | null;
  control_risk: number | null;
  risk_velocity: number | null;
  risk_acceleration: number | null;
  horizon_unit: string;
}

export interface SupervisorRulBlock {
  predicted_rul: number | null;
  interval: {
    lower: number | null;
    upper: number | null;
    margin: number | null;
    /** `inductive_conformal` carries a coverage guarantee; the others do not. */
    source: string;
    target_coverage?: number | null;
    label: string;
  };
  lifecycle_state: string | null;
  degradation_speed: number | null;
  degradation_acceleration: number | null;
  model_used: string | null;
  fallback_triggered: boolean | null;
}

export interface FocusComponent {
  id: string;
  short_label: string;
  name: string;
  mean_abs_z: number | null;
  health_pct: number | null;
  top_sensor: string | null;
  top_sensor_label: string | null;
  is_root_cause: boolean | null;
  inspection_focus: string | null;
  /** Which evidence localised it: twin residual, or the reasoning root cause. */
  basis: string;
}

export interface SupervisorAnomalyBlock {
  anomaly_score: number | null;
  is_anomalous: boolean | null;
  severity: string;
  anomaly_type: string | null;
  root_cause: string | null;
  health_index: number | null;
  health_state: string | null;
  health_trend: number | null;
  remaining_health_pct: number | null;
  early_warning_score: number | null;
  uncertainty_score: number | null;
  focus_component: FocusComponent | null;
}

export interface SupervisorRecommendation {
  final_decision: string;
  final_decision_label: string | null;
  /** Index into the five decision classes — the "action code" on the card. */
  action_code: number;
  severity: number;
  recommended_action: string | null;
  priority: string | null;
  urgency: string | null;
  urgency_label: string | null;
  confidence: number | null;
  confidence_level: string | null;
  entropy_score?: number | null;
  conflict_score?: number | null;
  raw_model_decision?: string | null;
  raw_model_decision_label?: string | null;
  override_applied?: boolean | null;
  override_reason?: string | null;
  requires_human_review: boolean | null;
  review_reason?: string | null;
  cost?: {
    decision: string;
    action_cost: number;
    unplanned_failure_cost: number;
    estimated_cost_avoided: number;
  } | null;
  /** Present only on projected states, where the model is not re-run. */
  basis?: string;
  reasons?: string[];
}

/** One physical sensor reading on a component card. */
export interface ComponentReading {
  sensor: string;
  symbol: string;
  description: string;
  unit: string;
  value: number;
  residual_z: number | null;
  /** False for the constant/quantised sensors excluded from health. */
  informative: boolean;
}

export interface SupervisorComponent {
  id: string;
  short_label: string;
  name: string;
  subsystem_label: string;
  /** Null when the section has no informative twin residual — never invented. */
  health_pct: number | null;
  status: "nominal" | "elevated" | "degraded" | "no_signal";
  mean_abs_z: number | null;
  contribution_pct: number | null;
  sensors: string[];
  informative_sensors: string[];
  top_sensor: string | null;
  top_sensor_label: string | null;
  readings: ComponentReading[];
  is_root_cause: boolean;
  root_cause_text: string | null;
  inspection_focus: string | null;
  suggested_action: string | null;
  note: string | null;
}

export interface ComponentSource {
  available: boolean;
  aligned: boolean;
  unavailable_reason: string | null;
  resolved_unit: number | null;
  root_cause_subsystem: string | null;
  health_scale: {
    metric: string;
    full_scale_z: number;
    elevated_z: number;
    degraded_z: number;
    excluded_sensors: string[];
    excluded_reason: string;
  };
}

/** One safety-policy rule, its live value, and whether it fired. */
export interface PolicyDriver {
  rule: string;
  expression: string;
  value: number | string | null;
  threshold: number | string;
  fired: boolean;
  forces: string;
}

export interface SupervisorCounterfactual {
  feasible: boolean;
  current_decision: string;
  target_decision: string | null;
  original_rul: number | null;
  original_risk_30: number | null;
  binding_constraints: Array<{
    name: string;
    variable: string;
    expression: string;
    current_value: number | null;
    threshold: number;
    required_value: number;
    shortfall: number;
    pressure_pct_required: number;
  }>;
  required_rul_gain: number;
  required_risk_30_drop: number;
  hpc_pressure_reduction_pct: number | null;
  cruise_thrust_reduction_pct: number | null;
  projected_rul: number | null;
  projected_risk_30: number | null;
  focus_component: string | null;
  actionable_statement: string;
  /** Always true: the derate lever rests on assumed, not fitted, sensitivities. */
  lever_is_assumption: boolean;
  assumptions: Record<string, number>;
  blocked_reason: string | null;
}

export interface SupervisorTrajectoryPoint {
  cycle: number;
  risk_10: number | null;
  risk_30: number | null;
  risk_50: number | null;
  threshold_30: number | null;
  predicted_rul: number | null;
  rul_lower: number | null;
  rul_upper: number | null;
  health_index: number | null;
  anomaly_score: number | null;
  confidence_score: number | null;
  final_decision: string | null;
  severity: number;
}

export interface SupervisorAgentView {
  engine_id: number | string;
  label: string;
  cycle: number;
  requested_cycle: number | null;
  fd_subset: string | null;
  cycle_range: { min: number; max: number; observed: number };
  generated_at: string;
  source: {
    decisions_csv: string;
    decisions_written_at: string | null;
    basis: string;
  };
  agents: {
    rul: UpstreamAgentStatus;
    failure_risk: UpstreamAgentStatus;
    anomaly: UpstreamAgentStatus;
  };
  policy: SupervisorPolicyState;
  failure_risk: SupervisorRiskBlock;
  rul: SupervisorRulBlock;
  anomaly: SupervisorAnomalyBlock;
  recommendation: SupervisorRecommendation;
  components: SupervisorComponent[];
  component_source: ComponentSource;
  xai: {
    narrative: string;
    upstream_contributions: { rul: number; failure_risk: number; anomaly: number };
    policy_drivers: PolicyDriver[];
  };
  counterfactual: SupervisorCounterfactual;
  trajectory: SupervisorTrajectoryPoint[];
  /** Cycles until the first recorded critical decision, if the run saw one. */
  critical_offset: number | null;
}

export interface SupervisorWhatIf {
  engine_id: number | string;
  base_cycle: number;
  target_cycle: number;
  cycles_ahead: number;
  /** `recorded` = actually observed and scored; `extrapolated` = projected. */
  basis: "recorded" | "extrapolated";
  basis_note: string;
  failure_risk: SupervisorRiskBlock;
  rul: SupervisorRulBlock;
  anomaly: SupervisorAnomalyBlock | null;
  recommendation: SupervisorRecommendation;
  components: SupervisorComponent[];
}

export interface SupervisorFusionRun {
  status: string;
  engine_id: number | string;
  cycle: number;
  rows_fused: number;
  latency_ms: number;
  live: SupervisorRecommendation;
  stored: SupervisorRecommendation;
  matches_stored: boolean;
  decision_distribution: Record<string, number>;
  note: string;
}

export interface SupervisorFusionSummary {
  engines: number;
  rows_total: number;
  decision_counts: Record<string, number>;
  priority_counts: Record<string, number>;
  risk_state_counts: Record<string, number>;
  health_state_counts: Record<string, number>;
  model_final_agreement: number;
  override_count: number;
  override_reasons: Record<string, number>;
  mean_confidence: number | null;
  decisions_written_at: string | null;
}

export interface SupervisorFeedbackRecord {
  sample_id: string;
  original_decision: string;
  engineer_decision: string;
  engineer_id: string;
  comments: string;
  timestamp: number;
  timestamp_iso?: string | null;
  agreed?: boolean;
}

export interface SupervisorFeedbackHistory {
  total: number;
  returned: number;
  override_count: number;
  agreement_pct: number | null;
  records: SupervisorFeedbackRecord[];
}

export interface SupervisorFeedbackResult {
  status: string;
  entry: SupervisorFeedbackRecord;
  agreed_with_agent: boolean;
  agent_decision: string;
  agent_decision_label: string | null;
  engineer_decision: string;
  engineer_decision_label: string | null;
  severity_delta: number;
  policy: Record<string, unknown> | null;
  total_feedback_entries: number;
  note: string;
}
