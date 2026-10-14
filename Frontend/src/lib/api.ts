/**
 * Typed client for the AeroTwin FastAPI backend.
 *
 * All paths are relative to `/api`, which Vite proxies to the backend in dev
 * (see vite.config.ts). Keeping them relative means the same bundle works
 * behind a reverse proxy in production without a rebuild.
 */
import type {
  AnomalyDashboardResponse,
  FailureRiskPredictionResponse,
  AnomalyEvaluationResponse,
  CompareResponse,
  ContextStatusResponse,
  DashboardResponse,
  FeatureRoutingResponse,
  FleetModesResponse,
  LeaderboardResponse,
  MaintenanceQueueResponse,
  RunFullResponse,
  SensorGraphResponse,
  ThresholdStatusResponse,
  BaselineCompareResponse,
  TrajectoryResponse,
  DemoSampleRequest,
  EngineSensorRow,
  EvaluationMetricsResponse,
  MemoryStatusResponse,
  RULPredictionResponse,
  SimilarMemory,
  PmTrajectoryResponse,
  FleetSummaryResponse,
  TwinMetricsResponse,
  TwinTraceResponse,
  DetectorConfigResponse,
  EngineTimelineResponse,
  EngineReasoningResponse,
  SupervisorAgentView,
  SupervisorEngineCatalog,
  SupervisorFeedbackHistory,
  SupervisorFeedbackResult,
  SupervisorFusionRun,
  SupervisorFusionSummary,
  SupervisorWhatIf,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    // FastAPI reports failures as {"detail": ...}; surface that rather than a
    // bare status code, since the backend's messages are actionable
    // ("Provide 'file' or 'stored_path'", "run /integration/run-full first").
    let detail: unknown;
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? body;
      if (typeof detail === "string") message = detail;
      else if (body?.message) message = String(body.message);
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new ApiError(message, res.status, detail);
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | boolean | undefined>) {
  const out = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") out.set(k, String(v));
  }
  const s = out.toString();
  return s ? `?${s}` : "";
}

export const api = {
  health: () => request<{ status?: string; service?: string }>("/health"),

  // ---- Integration (fused, cross-component) ----
  dashboard: (p: { limit?: number; offset?: number; engine_id?: number } = {}) =>
    request<DashboardResponse>(`/integration/dashboard${qs(p)}`),

  /**
   * Fleet state at each engine's latest cycle, reduced server-side.
   *
   * Prefer this over paging `/integration/dashboard` to compute fleet figures:
   * that endpoint caps at a page of a ~34k-row table, so a client-side
   * "latest per engine" over one page silently covered a fraction of the fleet
   * at mid-life cycles.
   */
  fleetSummary: () => request<FleetSummaryResponse>("/integration/fleet-summary"),

  leaderboard: (p: { metric?: string; limit?: number } = {}) =>
    request<LeaderboardResponse>(`/integration/leaderboard${qs(p)}`),

  maintenanceQueue: () =>
    request<MaintenanceQueueResponse>("/integration/maintenance-queue"),

  fleetModes: () => request<FleetModesResponse>("/integration/fleet-modes"),

  featureRouting: () =>
    request<FeatureRoutingResponse>("/integration/feature-routing"),

  /**
   * Forecast one engine's RUL/risk trajectory.
   *
   * Server-side: the trajectory model needs a full window of raw sensor rows,
   * which the browser never receives, so the backend assembles the sequence
   * from the last preprocessed upload.
   */
  engineTrajectory: (engineId: number | string, maxPoints = 40) =>
    request<TrajectoryResponse>(
      `/integration/engine/${engineId}/trajectory${qs({ max_points: maxPoints })}`,
    ),

  compare: (engineIds: Array<number | string>) =>
    request<CompareResponse>(
      `/integration/compare${qs({ engine_ids: engineIds.join(",") })}`,
    ),

  /**
   * Run the whole pipeline from a single uploaded file.
   *
   * Uploads directly rather than the two-step upload-then-stored_path flow:
   * that round trip requires pasting a Windows path into a query string, where
   * backslashes get double-escaped and 404 on a file that exists.
   */
  runFull: (
    file: File,
    opts: { stride?: number; include_uncertainty?: boolean } = {},
    signal?: AbortSignal,
  ) => {
    const form = new FormData();
    form.append("file", file);
    return request<RunFullResponse>(
      `/integration/run-full${qs({
        stride: opts.stride ?? 20,
        include_uncertainty: opts.include_uncertainty ?? true,
      })}`,
      { method: "POST", body: form, signal },
    );
  },

  uploadOnly: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Record<string, unknown>>("/integration/upload", {
      method: "POST",
      body: form,
    });
  },

  // ---- Anomaly & Health component ----
  anomalyDashboard: (engineId?: number) =>
    request<AnomalyDashboardResponse>(
      `/anomaly-health/dashboard${qs({ engine_id: engineId })}`,
    ),

  sensorGraph: () =>
    request<SensorGraphResponse>("/anomaly-health/reasoning/sensor-graph"),

  /**
   * Full reasoning history for one engine: candidate causes with their
   * evidence and firing rule, per-sensor contributions, onset/duration and any
   * matching remembered case.
   */
  engineReasoning: (engineId: number | string) =>
    request<EngineReasoningResponse>(`/anomaly-health/reasoning/${engineId}`),

  contextStatus: () =>
    request<ContextStatusResponse>("/anomaly-health/context-status"),

  feedbackSummary: () =>
    request<Record<string, unknown>>("/anomaly-health/feedback/summary"),

  feedbackThresholds: () =>
    request<Record<string, unknown>>("/anomaly-health/feedback/thresholds"),

  /** Record an operator verdict on a raised alert. */
  submitFeedback: (payload: {
    engine_id: number;
    cycle: number;
    verdict: "accepted" | "false_alarm" | "missed" | "unknown";
    operator?: string;
    comment?: string;
    confirmed_root_cause?: string | null;
  }) =>
    request<Record<string, unknown>>("/anomaly-health/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  schema: () => request<Record<string, unknown>>("/anomaly-health/schema"),

  /** Offline held-out validation results for the Anomaly & Health component. */
  anomalyEvaluation: () =>
    request<AnomalyEvaluationResponse>("/anomaly-health/evaluation/metrics"),

  /**
   * The detector fusion contract as configured, not as remembered.
   *
   * Weights live in the backend config; reading them keeps the dashboard from
   * describing an ensemble that has since been retuned.
   */
  detectorConfig: () =>
    request<DetectorConfigResponse>("/anomaly-health/detector-config"),

  /**
   * One engine's health trajectory, for any engine in the fleet.
   *
   * `/dashboard` bundles timelines for only the first 50 unit ids, so the
   * health screen could not chart the engines it ranks worst. This serves any
   * of them and reads only trajectory columns, so it stays fast.
   */
  engineHealthTimeline: (engineId: number | string, maxPoints = 600) =>
    request<EngineTimelineResponse>(
      `/anomaly-health/engine/${engineId}/timeline${qs({ max_points: maxPoints })}`,
    ),

  /**
   * The digital twin's own evidence: training error per ensemble member, the
   * residual acceptance gates, and the per-engine fidelity table.
   *
   * Offline artefacts merged server-side -- the API recomputes none of it.
   */
  twinMetrics: () => request<TwinMetricsResponse>("/anomaly-health/twin/metrics"),

  /**
   * One engine's actual-vs-expected trace for a single sensor.
   *
   * Server-side: the join spans a 66 MB prediction table and a 99 MB residual
   * table, so the browser gets a strided series instead of the raw cycles.
   */
  twinTrace: (
    engineId: number | string,
    sensor: string,
    maxPoints = 400,
  ) =>
    request<TwinTraceResponse>(
      `/anomaly-health/twin/trace${qs({
        engine_id: engineId,
        sensor,
        max_points: maxPoints,
      })}`,
    ),

  // ---- Predictive Maintenance ----
  artifactStatus: () =>
    request<Record<string, unknown>>("/predictive-maintenance/artifacts/status"),

  // ---- RUL agent (single-engine, live inference) ----

  /**
   * Score one engine's trailing window.
   *
   * Unlike `/integration/dashboard`, which serves pre-scored batch rows, this
   * runs the TCN in-process on the sequence supplied here, so the caller must
   * hold the raw rows. `include_debug` surfaces the artifact paths, q_hat and
   * the zero-filled-feature list the preprocessing step records.
   */
  /**
   * Score one engine's trailing window through the Failure Risk agent.
   *
   * `include_uncertainty` runs the Monte-Carlo dropout passes that produce the
   * per-horizon standard deviations; without it the service returns zeros.
   */
  failureRiskPredict: (body: {
    fd_subset: string;
    sequence: EngineSensorRow[];
    previous_risk_scores?: number[];
    include_uncertainty?: boolean;
    include_debug?: boolean;
  }) =>
    request<FailureRiskPredictionResponse>(
      "/predictive-maintenance/failure-risk/predict",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    ),

  rulPredict: (body: {
    fd_subset: string;
    sequence: EngineSensorRow[];
    previous_rul_predictions?: number[];
    include_debug?: boolean;
  }) =>
    request<RULPredictionResponse>("/predictive-maintenance/rul/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  /** A ready-made 70-cycle FD001 sequence, for the "Load demo sequence" button. */
  demoSampleRequest: () =>
    request<DemoSampleRequest>("/predictive-maintenance/demo/sample-request"),

  memoryStatus: () =>
    request<MemoryStatusResponse>("/predictive-maintenance/memory/status"),

  memorySimilar: (p: {
    fd_subset: string;
    degradation_speed?: number;
    degradation_acceleration?: number;
    limit?: number;
  }) =>
    request<{ matches?: SimilarMemory[]; [k: string]: unknown }>(
      `/predictive-maintenance/memory/similar${qs(p)}`,
    ),

  /**
   * Offline PICP/MPIW/NASA-score metrics, if the training team has exported them.
   * Returns `{available:false}` rather than erroring when the file is absent.
   */
  evaluationMetrics: () =>
    request<EvaluationMetricsResponse>("/predictive-maintenance/evaluation/metrics"),

  /**
   * Replay a sequence through both agents at growing window sizes.
   *
   * This is what backs the RUL/uncertainty charts and the risk-state donut:
   * the single-shot `/rul/predict` only ever scores the latest window, so a
   * cycle-over-cycle view has to come from here.
   */
  pmTrajectory: (body: {
    fd_subset: string;
    engine_id?: string;
    sequence: EngineSensorRow[];
    max_points?: number;
    /** Leave false to keep the RUL module from invoking the Failure Risk agent. */
    include_risk?: boolean;
  }) =>
    request<PmTrajectoryResponse>("/predictive-maintenance/predict/trajectory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  /** Per-horizon adaptive risk thresholds for one subset. */
  thresholdStatus: (fdSubset: string) =>
    request<ThresholdStatusResponse>(
      `/predictive-maintenance/threshold/status${qs({ fd_subset: fdSubset })}`,
    ),

  /**
   * Record whether a risk alert was a false alarm.
   *
   * Drives the Predictive-Maintenance side's own threshold adaptation, which is
   * separate from the anomaly component's feedback loop and had never received
   * a verdict.
   */
  submitThresholdFeedback: (payload: {
    fd_subset: string;
    horizon: string;
    was_false_alarm: boolean;
  }) =>
    request<Record<string, unknown>>("/predictive-maintenance/threshold/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  // ---- Maintenance Supervisor ----
  baselinesCompare: () =>
    request<BaselineCompareResponse>("/api/v1/supervisor/baselines/compare"),

  supervisorHealth: () =>
    request<Record<string, unknown>>("/api/v1/supervisor/health"),

  // ---- Maintenance Supervisor: agent operator view ----

  /** Engines with fused decisions, most urgent first, for the unit picker. */
  supervisorEngines: (limit?: number) =>
    request<SupervisorEngineCatalog>(`/integration/supervisor/engines${qs({ limit })}`),

  /**
   * The whole Supervisor Agent screen for one engine at one cycle.
   *
   * One request on purpose: the cards are all facets of the same fused decision
   * record, so fetching them separately would let them render mutually
   * inconsistent states while they settled.
   */
  supervisorAgentView: (engineId: number | string, cycle?: number) =>
    request<SupervisorAgentView>(
      `/integration/supervisor/agent-view${qs({ engine_id: engineId, cycle })}`,
    ),

  /**
   * Project one engine forward for the what-if simulator.
   *
   * Inside the observed history the response is the cycle as actually recorded
   * and scored (`basis: "recorded"`); past it, the state is extrapolated along
   * each signal's own slope and says so.
   */
  supervisorWhatIf: (
    engineId: number | string,
    cyclesAhead: number,
    cycle?: number,
  ) =>
    request<SupervisorWhatIf>(
      `/integration/supervisor/what-if${qs({
        engine_id: engineId,
        cycles_ahead: cyclesAhead,
        cycle,
      })}`,
    ),

  /**
   * Execute the decision-fusion engine live for one engine.
   *
   * Runs the real `DecisionFusionEngine` over that engine's stored upstream
   * features and reports the result beside the persisted one. Nothing on disk
   * is modified and no model is retrained.
   */
  supervisorRunFusion: (engineId: number | string, cycle?: number) =>
    request<SupervisorFusionRun>(
      `/integration/supervisor/run-fusion${qs({ engine_id: engineId, cycle })}`,
      { method: "POST" },
    ),

  /** A self-contained, auditable record of one decision. */
  supervisorCertificate: (engineId: number | string, cycle?: number) =>
    request<Record<string, unknown>>(
      `/integration/supervisor/certificate${qs({ engine_id: engineId, cycle })}`,
    ),

  /** Fleet-level record of what the fusion step decided on the last run. */
  supervisorFusionSummary: () =>
    request<SupervisorFusionSummary>("/integration/supervisor/fusion-summary"),

  /**
   * Log an engineer's accept/override verdict on a fused decision.
   *
   * This is the entry point to the Supervisor's continuous-learning loop —
   * the feedback store and adaptive-policy pass existed but had never had a
   * caller. It logs the verdict and re-derives the policy; it retrains nothing.
   */
  supervisorSubmitFeedback: (payload: {
    engine_id: number;
    cycle: number;
    engineer_decision: string;
    engineer_id?: string;
    comments?: string;
    adapt_policy?: boolean;
  }) =>
    request<SupervisorFeedbackResult>("/integration/supervisor/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  supervisorFeedbackHistory: (engineId?: number | string, limit = 20) =>
    request<SupervisorFeedbackHistory>(
      `/integration/supervisor/feedback${qs({ engine_id: engineId, limit })}`,
    ),

  /** Explain one fused decision from its upstream feature values. */
  supervisorExplain: (payload: {
    engine_id: number | string;
    cycle: number;
    fd_subset?: string;
    predicted_rul: number;
    risk_10?: number;
    risk_30?: number;
    risk_50?: number;
    anomaly_score?: number;
    anomaly_severity?: string;
  }) =>
    request<Record<string, unknown>>("/api/v1/supervisor/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};
