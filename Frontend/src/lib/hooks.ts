/**
 * React Query hooks over the backend API.
 *
 * The fused decision set is ~34k rows, so the dashboard hook pulls a bounded
 * page (the endpoint is paged for exactly this reason) and every derived fleet
 * statistic is computed from that page. `total` is always reported alongside so
 * the UI can say honestly how much of the fleet a figure covers.
 */
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { refetchIntervalMs } from "./preferences";
import type {
  DecisionRow,
  EngineSensorRow,
  FleetSummary,
  SimilarMemory,
} from "./types";

/**
 * Default poll interval for "live" panels, in ms.
 *
 * Kept as the documented default; the effective interval is the operator's
 * preference, read per tick via `refetchIntervalMs()` so changing it in
 * Settings takes effect without re-creating every query. Setting it to 0 in
 * Settings stops polling altogether.
 */
export const LIVE_MS = 15_000;

const num = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

const mean = (xs: number[]) =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

function tally(rows: DecisionRow[], key: keyof DecisionRow) {
  const out: Record<string, number> = {};
  for (const r of rows) {
    const v = r[key];
    if (v === null || v === undefined || v === "") continue;
    const k = String(v);
    out[k] = (out[k] ?? 0) + 1;
  }
  return out;
}

/** Latest row per engine — the operationally meaningful snapshot. */
export function latestPerEngine(rows: DecisionRow[]): DecisionRow[] {
  const best = new Map<string, DecisionRow>();
  for (const r of rows) {
    const id = String(r.engine_id);
    const prev = best.get(id);
    if (!prev || (num(r.cycle) ?? 0) > (num(prev.cycle) ?? 0)) best.set(id, r);
  }
  return [...best.values()].sort(
    (a, b) => Number(a.engine_id) - Number(b.engine_id),
  );
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: () => refetchIntervalMs(),
    retry: false,
  });
}

export function useDashboard(limit = 4000, engineId?: number) {
  return useQuery({
    queryKey: ["dashboard", limit, engineId],
    queryFn: () => api.dashboard({ limit, engine_id: engineId }),
    refetchInterval: () => refetchIntervalMs(),
  });
}

export function useLeaderboard(metric = "predicted_rul", limit = 10) {
  return useQuery({
    queryKey: ["leaderboard", metric, limit],
    queryFn: () => api.leaderboard({ metric, limit }),
    refetchInterval: () => refetchIntervalMs(),
  });
}

export function useMaintenanceQueue() {
  return useQuery({
    queryKey: ["maintenance-queue"],
    queryFn: api.maintenanceQueue,
    refetchInterval: () => refetchIntervalMs(),
  });
}

export function useFleetModes() {
  return useQuery({
    queryKey: ["fleet-modes"],
    queryFn: api.fleetModes,
    refetchInterval: () => refetchIntervalMs(),
  });
}

export function useFeatureRouting() {
  return useQuery({ queryKey: ["feature-routing"], queryFn: api.featureRouting });
}

/**
 * The Anomaly & Health component's own dashboard.
 *
 * This is the only endpoint that carries the explainability layer -- generated
 * narrative, ranked contributing sensors, subsystem attribution and detector
 * agreement. The fused /integration/dashboard keeps root_cause but drops the
 * rest, so pages that show "why" must read from here.
 */
export function useAnomalyDashboard(engineId?: number) {
  return useQuery({
    queryKey: ["anomaly-dashboard", engineId],
    queryFn: () => api.anomalyDashboard(engineId),
    // Not LIVE_MS: this payload is a pipeline artefact that only changes when
    // the pipeline re-runs, and building it reduces a 216 MB table. Polling it
    // every 15s spent that repeatedly and, before the endpoint was made
    // read-only, let one poll collide with the previous one mid-write.
    refetchInterval: 60_000,
  });
}

export function useSensorGraph() {
  return useQuery({ queryKey: ["sensor-graph"], queryFn: api.sensorGraph });
}

export function useContextStatus() {
  return useQuery({
    queryKey: ["context-status"],
    queryFn: api.contextStatus,
    refetchInterval: () => refetchIntervalMs(),
  });
}

export function useEngineReasoning(engineId?: number | string) {
  return useQuery({
    queryKey: ["reasoning", engineId],
    queryFn: () => api.engineReasoning(engineId!),
    enabled: engineId !== undefined && engineId !== null && engineId !== "",
  });
}

export function useFeedbackSummary() {
  return useQuery({
    queryKey: ["feedback-summary"],
    queryFn: api.feedbackSummary,
    refetchInterval: () => refetchIntervalMs(),
  });
}

export function useFeedbackThresholds() {
  return useQuery({
    queryKey: ["feedback-thresholds"],
    queryFn: api.feedbackThresholds,
  });
}

/**
 * Submit an operator verdict.
 *
 * This is the entry point to the component's learning loop: the verdict is
 * stored, matched back to the anomaly in alert memory, and the adaptive
 * threshold is recomputed from the full feedback history. Nothing in the UI
 * reached this before, so the loop had never run on real input.
 */
export function useSubmitFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.submitFeedback,
    onSuccess: () => {
      // The verdict can move the threshold, which changes what counts as an
      // anomaly, so the dependent views are refetched rather than left stale.
      qc.invalidateQueries({ queryKey: ["feedback-summary"] });
      qc.invalidateQueries({ queryKey: ["feedback-thresholds"] });
      qc.invalidateQueries({ queryKey: ["anomaly-dashboard"] });
    },
  });
}

/** Per-horizon adaptive risk thresholds for one subset. */
export function useThresholdStatus(fdSubset?: string) {
  return useQuery({
    queryKey: ["threshold-status", fdSubset],
    queryFn: () => api.thresholdStatus(fdSubset!),
    enabled: !!fdSubset,
    refetchInterval: () => refetchIntervalMs(),
  });
}

/** Record a false-alarm / confirmed verdict against a risk horizon. */
export function useThresholdFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.submitThresholdFeedback,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["threshold-status"] }),
  });
}

/** Champion vs baseline benchmark. */
/** Forward RUL/risk forecast for one engine. */
export function useTrajectory(engineId?: number | string, maxPoints = 40) {
  return useQuery({
    queryKey: ["trajectory", engineId, maxPoints],
    queryFn: () => api.engineTrajectory(engineId!, maxPoints),
    enabled: engineId !== undefined && engineId !== null && engineId !== "",
    // Replays the sequence through both models at growing windows, so it is
    // far heavier than a dashboard read -- fetch on demand, never on a poll.
    refetchInterval: false,
    staleTime: 5 * 60_000,
  });
}

export function useBaselines() {
  return useQuery({ queryKey: ["baselines"], queryFn: api.baselinesCompare });
}

export function useArtifactStatus() {
  return useQuery({ queryKey: ["artifact-status"], queryFn: api.artifactStatus });
}

export function useCompare(engineIds: Array<number | string>) {
  return useQuery({
    queryKey: ["compare", engineIds.join(",")],
    queryFn: () => api.compare(engineIds),
    enabled: engineIds.length > 0,
  });
}

/** Fleet rollup derived from the fetched page of decisions. */
/**
 * Every recorded cycle for ONE engine, for per-engine trajectory charts.
 *
 * `/integration/dashboard` filters by `engine_id` server-side, so this pulls
 * only that engine's rows (a few hundred) rather than a page of the whole
 * 34k-row table and filtering in the browser -- which is what the pages did
 * before, and why a chart could silently render nothing when the engine
 * happened to fall outside the fetched page.
 */
export function useEngineRows(engineId?: number | string) {
  const enabled = engineId !== undefined && engineId !== null && engineId !== "";
  const q = useQuery({
    queryKey: ["integration", "engine-rows", String(engineId)],
    queryFn: () => api.dashboard({ engine_id: Number(engineId), limit: 100000 }),
    enabled,
  });
  const rows = useMemo(
    () => [...(q.data?.decisions ?? [])].sort((a, b) => a.cycle - b.cycle),
    [q.data],
  );
  return { ...q, rows };
}

/**
 * Fleet-wide summary at each engine's latest cycle.
 *
 * Backed by `/integration/fleet-summary`, which does the reduction server-side.
 * It previously paged `/integration/dashboard?limit=4000` and reduced in the
 * browser, which was wrong twice over: 4,000 of 33,991 rows is 11.8% of the
 * table and only 33 of the 259 engines, and "latest per engine" computed
 * inside that page gave each engine its last *paged* cycle -- mid-life, not
 * current. The headline read 75.0 health with 33 engines; the true
 * latest-cycle figure is 73.6 across all 259, and the state counts were badly
 * off (Healthy 112 / Degrading 72 / Warning 48 / Critical 27).
 *
 * The `limit` parameter is accepted and ignored so existing call sites keep
 * compiling; there is nothing left to page.
 */
export function useFleetSummary(_limit = 4000) {
  const q = useQuery({
    queryKey: ["integration", "fleet-summary"],
    queryFn: () => api.fleetSummary(),
    refetchInterval: () => refetchIntervalMs(),
  });

  const summary = useMemo<FleetSummary | null>(() => {
    const d = q.data;
    if (!d) return null;
    const a = d.aggregates;
    const lower = (o: Record<string, number>, k: string) =>
      Object.entries(o ?? {}).find(([n]) => n.toLowerCase() === k)?.[1] ?? 0;
    const states = a.health_state_counts ?? {};

    return {
      engines: d.totals.engines,
      rows: d.totals.rows_total,
      avgHealth: a.mean_health_index ?? 0,
      avgRul: a.mean_predicted_rul ?? 0,
      // `health_state` has FIVE bands, and "Failure Risk" is the bottom one --
      // below Critical, not between Warning and Critical (measured on FD003:
      // Failure Risk spans health index 7.5-17.8 against Critical's 21.5-28.0).
      // Counting only "critical" here left those engines in none of the three
      // totals: 31 healthy + 55 warning + 3 critical accounted for 89 of 100
      // engines, and the Overview KPI, the sidebar badge and the topbar alert
      // count all under-reported the fleet's worst condition by 11 engines.
      criticalCount: lower(states, "critical") + lower(states, "failure risk"),
      failureRiskCount: lower(states, "failure risk"),
      warningCount: lower(states, "warning") + lower(states, "degrading"),
      healthyCount: lower(states, "healthy"),
      anomalyCount: a.anomalous_engines ?? 0,
      reviewCount: a.review_required_engines ?? 0,
      decisionCounts: a.final_decision_counts ?? {},
      healthStateCounts: states,
      severityCounts: a.anomaly_severity_counts ?? {},
      avgConfidence: a.mean_confidence_score ?? 0,
      subsets: a.subsets ?? [],
    };
  }, [q.data]);

  return { ...q, summary, latest: q.data?.engines ?? [] };
}

// ---------------------------------------------------------------------------
// RUL agent (single-engine, live inference)
// ---------------------------------------------------------------------------

/**
 * Runs `/rul/predict` on demand and times it.
 *
 * Kept as a mutation rather than a query because the screen is explicitly
 * button-driven ("Run prediction") and each call costs a real TCN forward
 * pass -- polling it would be both wrong and expensive. `inferenceMs` is
 * measured around the request so the header can report it honestly instead of
 * echoing a server-side guess.
 */
export function useRulPredict() {
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [predictedAt, setPredictedAt] = useState<Date | null>(null);

  const mutation = useMutation({
    mutationFn: async (body: {
      fd_subset: string;
      sequence: EngineSensorRow[];
      previous_rul_predictions?: number[];
    }) => {
      const t0 = performance.now();
      const res = await api.rulPredict({ ...body, include_debug: true });
      setInferenceMs(Math.round(performance.now() - t0));
      setPredictedAt(new Date());
      return res;
    },
  });

  return { ...mutation, inferenceMs, predictedAt };
}

/** The demo sequence behind the "Load demo sequence" button. */
export function useDemoSequence() {
  return useMutation({ mutationFn: () => api.demoSampleRequest() });
}

/** Agent memory panel. Polls, since other sessions write to the same store. */
export function useMemoryStatus() {
  const q = useQuery({
    queryKey: ["pm", "memory", "status"],
    queryFn: () => api.memoryStatus(),
    refetchInterval: () => refetchIntervalMs(),
  });
  // Spread first: react-query exposes its own `status` ("pending"/"error"/...),
  // which would otherwise shadow the payload below.
  return { ...q, status: q.data ?? null };
}

/** Nearest past cases for the current degradation signature. */
export function useSimilarMemories(
  fdSubset: string | null,
  speed: number | null,
  accel: number | null,
) {
  const enabled = Boolean(fdSubset);
  const q = useQuery({
    queryKey: ["pm", "memory", "similar", fdSubset, speed, accel],
    queryFn: () =>
      api.memorySimilar({
        fd_subset: fdSubset as string,
        degradation_speed: speed ?? 0,
        degradation_acceleration: accel ?? 0,
        limit: 5,
      }),
    enabled,
  });
  const matches = (q.data?.matches ?? []) as SimilarMemory[];
  return { matches, ...q };
}

/**
 * PICP/MPIW coverage metrics.
 *
 * Never throws on absence: the endpoint answers `{available:false}` when
 * `evaluation_metrics.json` has not been exported, and the UI renders "--".
 */
export function useEvaluationMetrics() {
  const q = useQuery({
    queryKey: ["pm", "evaluation", "metrics"],
    queryFn: () => api.evaluationMetrics(),
    staleTime: 5 * 60_000,
  });
  const first = q.data?.available ? q.data.data?.models?.[0] : undefined;
  return {
    available: Boolean(q.data?.available),
    picp: typeof first?.picp === "number" ? first.picp : null,
    mpiw: typeof first?.mpiw === "number" ? first.mpiw : null,
    /**
     * The whole first entry (the all-subsets roll-up), so callers can show the
     * point-accuracy figures -- MAE/RMSE/R2/NASA score -- alongside the
     * interval ones. Previously only picp/mpiw were surfaced, which left the
     * dashboard unable to state how accurate the RUL model actually is.
     */
    model: first ?? null,
    ...q,
  };
}

/** Backend reachability + artifact readiness, for the header status pills. */
export function useAgentStatus() {
  const health = useQuery({
    queryKey: ["pm", "health"],
    queryFn: () => api.health(),
    refetchInterval: () => refetchIntervalMs(),
  });
  const artifacts = useQuery({
    queryKey: ["pm", "artifacts", "status"],
    queryFn: () => api.artifactStatus(),
    refetchInterval: 60_000,
  });

  // "Model active" means the RUL artifacts for at least one subset actually
  // resolved on disk -- not merely that the process answered a health ping.
  const rows = (artifacts.data?.rul_lifecycle ?? []) as Array<{
    missing_model?: boolean;
  }>;
  const modelActive = rows.length > 0 && rows.some((r) => !r.missing_model);

  return {
    apiConnected: !health.isError && Boolean(health.data),
    modelActive,
    isLoading: health.isLoading || artifacts.isLoading,
  };
}

/**
 * "Analyze trajectory" -- replays the loaded sequence through both agents.
 *
 * Costs one RUL + one failure-risk inference per point, so it stays a manual
 * mutation like `useRulPredict` rather than anything that polls.
 */
export function usePmTrajectory() {
  return useMutation({
    mutationFn: (body: {
      fd_subset: string;
      engine_id?: string;
      sequence: EngineSensorRow[];
      max_points?: number;
      include_risk?: boolean;
    }) => api.pmTrajectory(body),
  });
}

/**
 * Held-out validation results for the Anomaly & Health detector.
 *
 * Static offline output -- computed once against ground-truth labels the API
 * never sees -- so it is cached hard rather than polled.
 */
export function useAnomalyEvaluation() {
  const q = useQuery({
    queryKey: ["anomaly-evaluation"],
    queryFn: () => api.anomalyEvaluation(),
    staleTime: Infinity,
  });
  const test = q.data?.metrics?.holdout_anomaly_evaluation?.test ?? null;
  const train = q.data?.metrics?.holdout_anomaly_evaluation?.train ?? null;
  const perSubset = q.data?.metrics?.holdout_anomaly_evaluation?.test_per_subset ?? null;
  const health = q.data?.metrics?.holdout_test_evaluation ?? null;
  const acceptance = q.data?.metrics?.evaluate_anomaly ?? null;
  const healthAcceptance = q.data?.metrics?.evaluate_health ?? null;
  const alertMix = q.data?.metrics?.health_alerts?.alert_distribution ?? null;
  const reasoning = q.data?.metrics?.reasoning ?? null;
  const reasoningGate = q.data?.metrics?.evaluate_reasoning ?? null;
  const explainGate = q.data?.metrics?.evaluate_explainability ?? null;
  return {
    ...q,
    available: q.data?.available ?? false,
    test,
    train,
    perSubset,
    health,
    acceptance,
    healthAcceptance,
    alertMix,
    reasoning,
    reasoningGate,
    explainGate,
  };
}

/**
 * Runs `/failure-risk/predict` on demand and times it.
 *
 * A mutation, not a query: the screen is button-driven and each call costs a
 * BiLSTM forward pass plus (when uncertainty is requested)
 * FAILURE_RISK_MC_PASSES additional stochastic passes. Polling it would be
 * both wrong and expensive.
 */
export function useFailureRiskPredict() {
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [predictedAt, setPredictedAt] = useState<Date | null>(null);
  const m = useMutation({
    mutationFn: async (body: {
      fd_subset: string;
      sequence: EngineSensorRow[];
      previous_risk_scores?: number[];
      include_uncertainty?: boolean;
      include_debug?: boolean;
    }) => {
      const started = performance.now();
      const result = await api.failureRiskPredict(body);
      setInferenceMs(Math.round(performance.now() - started));
      setPredictedAt(new Date());
      return result;
    },
  });
  return { ...m, inferenceMs, predictedAt };
}

/**
 * The digital twin's training and validation evidence.
 *
 * Offline artefacts written once by the pipeline, so this is cached hard
 * rather than polled -- nothing about it changes between renders.
 */
export function useTwinMetrics() {
  const q = useQuery({
    queryKey: ["twin-metrics"],
    queryFn: () => api.twinMetrics(),
    staleTime: Infinity,
  });
  return { ...q, available: q.data?.available ?? false };
}

/**
 * One engine's actual-vs-twin trace for one sensor.
 *
 * Keyed on engine and sensor so React Query keeps every pair the user has
 * already looked at: the backend read is ~3.5 s cold and instant warm, and
 * re-fetching on every dropdown change would make the picker feel broken.
 */
export function useTwinTrace(
  engineId?: number | string,
  sensor?: string,
  maxPoints = 400,
) {
  return useQuery({
    queryKey: ["twin-trace", engineId, sensor, maxPoints],
    queryFn: () => api.twinTrace(engineId!, sensor!, maxPoints),
    enabled: engineId !== undefined && !!sensor,
    staleTime: Infinity,
  });
}

/**
 * Detector weights and the fusion calibration rule, read from backend config.
 *
 * Static configuration, so it is cached rather than polled.
 */
export function useDetectorConfig() {
  return useQuery({
    queryKey: ["detector-config"],
    queryFn: () => api.detectorConfig(),
    staleTime: Infinity,
  });
}

/**
 * One engine's health trajectory, fetched on demand.
 *
 * The bundled `/dashboard` timelines cover only the first 50 unit ids, which
 * excludes almost every engine this page ranks as worst. Keyed per engine so
 * React Query keeps each one the user has already opened; the backend caches
 * the trajectory columns, so every engine after the first returns in ms.
 */
export function useEngineHealthTimeline(engineId?: number | string) {
  return useQuery({
    queryKey: ["engine-health-timeline", String(engineId ?? "")],
    queryFn: () => api.engineHealthTimeline(engineId!),
    enabled: engineId !== undefined && engineId !== null && String(engineId) !== "",
    staleTime: 5 * 60_000,
    retry: false,
  });
}


// ---------------------------------------------------------------------------
// Autonomous Maintenance Supervisor Agent — operator view
// ---------------------------------------------------------------------------

/** Engines with fused decisions, most urgent first, for the unit picker. */
export function useSupervisorEngines(limit?: number) {
  return useQuery({
    queryKey: ["supervisor", "engines", limit ?? "all"],
    queryFn: () => api.supervisorEngines(limit),
    // Rebuilt only when the pipeline re-runs, and the picker is opened rarely.
    staleTime: 60_000,
  });
}

/**
 * Everything the Supervisor Agent screen shows for one engine at one cycle.
 *
 * Kept as a single query rather than one per card: the backend bundles the
 * whole decision record for exactly this reason, and splitting it here would
 * reintroduce the inconsistent-while-settling problem the bundle avoids.
 * Previous data is held across an engine change so the layout does not collapse
 * to a spinner every time the operator picks a different unit.
 */
export function useSupervisorAgentView(
  engineId?: number | string,
  cycle?: number,
) {
  const enabled = engineId !== undefined && engineId !== null && engineId !== "";
  return useQuery({
    queryKey: ["supervisor", "agent-view", String(engineId ?? ""), cycle ?? "latest"],
    queryFn: () => api.supervisorAgentView(engineId!, cycle),
    enabled,
    placeholderData: (previous) => previous,
    refetchInterval: () => refetchIntervalMs(),
  });
}

/**
 * The what-if simulator's projection.
 *
 * Disabled at zero offset — that is just the current cycle, which the agent
 * view already holds — and cached per offset so dragging back over a position
 * the operator already visited is instant.
 */
export function useSupervisorWhatIf(
  engineId?: number | string,
  cyclesAhead = 0,
  cycle?: number,
) {
  const enabled =
    engineId !== undefined && engineId !== null && engineId !== "" && cyclesAhead > 0;
  return useQuery({
    queryKey: [
      "supervisor",
      "what-if",
      String(engineId ?? ""),
      cycle ?? "latest",
      cyclesAhead,
    ],
    queryFn: () => api.supervisorWhatIf(engineId!, cyclesAhead, cycle),
    enabled,
    placeholderData: (previous) => previous,
    staleTime: 5 * 60_000,
  });
}

/** Fleet-level record of what the fusion step decided on the last run. */
export function useSupervisorFusionSummary() {
  return useQuery({
    queryKey: ["supervisor", "fusion-summary"],
    queryFn: () => api.supervisorFusionSummary(),
    refetchInterval: () => refetchIntervalMs(),
  });
}

/**
 * Runs the decision-fusion engine live and times it.
 *
 * A mutation, not a query: the button is explicit and each press costs a real
 * champion-model forward pass over the engine's whole history. Polling it would
 * be both wrong and expensive.
 */
export function useSupervisorRunFusion() {
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [ranAt, setRanAt] = useState<Date | null>(null);
  const mutation = useMutation({
    mutationFn: async (vars: { engineId: number | string; cycle?: number }) => {
      const started = performance.now();
      const result = await api.supervisorRunFusion(vars.engineId, vars.cycle);
      setElapsedMs(Math.round(performance.now() - started));
      setRanAt(new Date());
      return result;
    },
  });
  return { ...mutation, elapsedMs, ranAt };
}

/** Engineer verdicts logged against the supervisor's decisions. */
export function useSupervisorFeedbackHistory(
  engineId?: number | string,
  limit = 20,
) {
  return useQuery({
    queryKey: ["supervisor", "feedback", String(engineId ?? "all"), limit],
    queryFn: () => api.supervisorFeedbackHistory(engineId, limit),
  });
}

/**
 * Submit an engineer's accept/override verdict.
 *
 * On success the agent view is invalidated as well as the history: the verdict
 * re-runs the adaptive-policy pass, which changes the policy state the header
 * reports.
 */
export function useSupervisorSubmitFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.supervisorSubmitFeedback,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["supervisor", "feedback"] });
      qc.invalidateQueries({ queryKey: ["supervisor", "agent-view"] });
    },
  });
}
