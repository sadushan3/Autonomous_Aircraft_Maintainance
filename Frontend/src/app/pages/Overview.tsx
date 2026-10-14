/**
 * Executive Overview -- the fleet's single command screen.
 *
 * Everything here is derived from ONE server-side reduction
 * (`/integration/fleet-summary`, latest cycle per engine) plus four small side
 * reads: the maintenance queue, the operating-regime drift check and the two
 * offline held-out evaluation artefacts. Nothing on this page recomputes a
 * model output in the browser -- the numbers are the pipeline's own, and every
 * panel states the population it covers so a headline figure is never
 * ambiguous about what it is an average of.
 *
 * Colour is assigned by the job it does, not by taste: the four health states
 * and the five supervisor decisions each use an ordinal status ramp validated
 * for colour-vision separation against this app's slate surface (worst
 * adjacent pair: normal-vision dE 15.6, CVD dE 13.3). The previous
 * amber/orange pairing measured dE 9.6 -- effectively one colour on screen.
 * Magnitude-only charts (RUL distribution, root causes) use a single hue,
 * because their bars encode one measure and hue would carry no information.
 */
import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BadgeCheck,
  Brain,
  ClipboardList,
  Cpu,
  Database,
  Gauge,
  Layers,
  ShieldAlert,
  Signal,
  Timer,
  TrendingUp,
  Wrench,
} from "lucide-react";
import { motion } from "motion/react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  LabelList,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { StatusBadge } from "../components/StatusBadge";
import { GlassCard } from "../components/GlassCard";
import { KPICard } from "../components/KPICard";
import { MetricCard } from "../components/MetricCard";
import {
  AnimatedNumber,
  Empty,
  ErrorState,
  LiveBadge,
  Loading,
  PageHeader,
  Reveal,
} from "../components/Live";
import {
  useAgentStatus,
  useAnomalyEvaluation,
  useEngineRows,
  useEvaluationMetrics,
  useFleetModes,
  useFleetSummary,
  useMaintenanceQueue,
} from "../../lib/hooks";
import { usePreferences } from "../../lib/preferences";
import type { DecisionRow } from "../../lib/types";

// ---------------------------------------------------------------------------
// Palette -- validated as a set against the #0f172a chart surface
// ---------------------------------------------------------------------------

const HEALTHY = "#10b981";
const DEGRADING = "#facc15";
const WARNING = "#f97316";
const CRITICAL = "#e11d48";
const INFO = "#22d3ee";
const NEUTRAL = "#64748b";

/** Single hue for magnitude-only bars. */
const MAGNITUDE = "#22d3ee";

const GRID = "#1e293b";
const AXIS = "#64748b";

/**
 * All five bands the backend emits, worst last. "Failure Risk" is the BOTTOM
 * band, not a mid-range one: measured on FD003 it spans health index 7.5-17.8
 * against Critical's 21.5-28.0. Leaving it out of this list painted those
 * engines with the neutral "unknown" grey.
 */
const STATE_ORDER = ["Healthy", "Degrading", "Warning", "Critical", "Failure Risk"];

/**
 * `FAILURE_RISK` is fuchsia rather than the deeper red the Health Monitoring
 * page uses for the same band. A second red cannot be told apart from Critical
 * on this surface -- #b91c1c measures dE 9.3 against #e11d48 for normal vision,
 * where the floor is 15 -- and going dark enough to separate them drops below
 * the 3:1 contrast the dark background needs. This step clears every gate, and
 * the RUL page already uses it for "Unstable", its own beyond-critical band.
 */
const FAILURE_RISK = "#d946ef";

const STATE_COLOR: Record<string, string> = {
  Healthy: HEALTHY,
  Degrading: DEGRADING,
  Warning: WARNING,
  Critical: CRITICAL,
  "Failure Risk": FAILURE_RISK,
};

/** Ordered least → most urgent, so the ramp reads as an escalation ladder. */
const DECISION_ORDER = [
  "continue_operation",
  "monitor_closely",
  "schedule_inspection",
  "schedule_maintenance",
  "immediate_maintenance",
];

const DECISION_COLOR: Record<string, string> = {
  continue_operation: HEALTHY,
  monitor_closely: INFO,
  schedule_inspection: DEGRADING,
  schedule_maintenance: WARNING,
  immediate_maintenance: CRITICAL,
};

const PRIORITY_COLOR: Record<string, string> = {
  Critical: CRITICAL,
  High: WARNING,
  Medium: DEGRADING,
  Low: HEALTHY,
};

/** Ordinal ramp for the small composition bars (lifecycle / risk state). */
const ORDINAL = [HEALTHY, INFO, DEGRADING, WARNING, CRITICAL];

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

const pretty = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

const fmt = (n: number | null | undefined, d = 1) =>
  n === null || n === undefined || !Number.isFinite(Number(n))
    ? "—"
    : Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

const pct = (n: number | null | undefined, d = 1) =>
  n === null || n === undefined || !Number.isFinite(Number(n)) ? "—" : `${(Number(n) * 100).toFixed(d)}%`;

/**
 * Map a health state onto the three badge tones.
 *
 * "Failure Risk" is matched explicitly. It is the bottom band, and falling
 * through to the default painted the fleet's ELEVEN sickest engines -- health
 * index 7.5 to 17.8 -- with a green "Healthy" badge, while the Health
 * Monitoring page showed the same engines in deep red.
 */
function statusFor(state: string | null): "healthy" | "warning" | "critical" {
  const s = (state ?? "").toLowerCase();
  if (s === "critical" || s === "failure risk") return "critical";
  if (s === "warning" || s === "degrading") return "warning";
  return "healthy";
}

function tallyBy(rows: DecisionRow[], key: keyof DecisionRow) {
  const out: Record<string, number> = {};
  for (const r of rows) {
    const v = r[key];
    if (v === null || v === undefined || v === "") continue;
    out[String(v)] = (out[String(v)] ?? 0) + 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Presentation primitives local to this page
// ---------------------------------------------------------------------------

/** One tooltip style for every chart, so the page reads as one instrument. */
function ChartTip({
  active,
  payload,
  label,
  unit = "",
}: {
  active?: boolean;
  payload?: Array<Record<string, any>>;
  label?: string | number;
  unit?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-700/70 bg-slate-950/95 px-3 py-2 shadow-xl backdrop-blur-sm">
      {label !== undefined && label !== "" && (
        <div className="mb-1.5 text-xs font-semibold text-slate-200">{label}</div>
      )}
      <div className="space-y-1">
        {payload.map((p, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ background: p.payload?.color ?? p.color ?? p.fill }}
            />
            <span className="text-slate-400">{p.name}</span>
            <span className="ml-3 font-semibold tabular-nums text-slate-100">
              {typeof p.value === "number"
                ? fmt(p.value, Number.isInteger(p.value) ? 0 : 1)
                : String(p.value)}
              {unit}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Section rule -- gives a long dashboard a scannable hierarchy. */
function Section({ eyebrow, title, hint }: { eyebrow: string; title: string; hint?: string }) {
  return (
    <div className="flex items-end justify-between gap-4 border-b border-slate-800/60 pb-2 pt-2">
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400/70">{eyebrow}</div>
        <h2 className="text-base font-semibold text-slate-200">{title}</h2>
      </div>
      {hint && <p className="hidden text-xs text-slate-500 md:block">{hint}</p>}
    </div>
  );
}

/** One figure with the meaning stated underneath it. */
function Metric({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "good" | "warn" | "bad" | "neutral";
}) {
  const toneClass = {
    good: "text-emerald-400",
    warn: "text-amber-400",
    bad: "text-rose-400",
    neutral: "text-slate-200",
  }[tone];
  return (
    <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-3">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
      {hint && <div className="mt-1 text-[11px] leading-snug text-slate-500">{hint}</div>}
    </div>
  );
}

/**
 * Stacked share bar. Segments carry a 2px surface gap so adjacent fills never
 * touch, and every segment is direct-labelled below -- identity never rests on
 * colour alone.
 */
function CompositionBar({
  segments,
  total,
}: {
  segments: Array<{ name: string; value: number; color: string }>;
  total: number;
}) {
  if (!total) return null;
  return (
    <div>
      <div className="flex h-3 w-full gap-[2px] overflow-hidden rounded-full bg-slate-800/60">
        {segments.map((s) => (
          <motion.div
            key={s.name}
            initial={{ width: 0 }}
            animate={{ width: `${(s.value / total) * 100}%` }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            style={{ background: s.color }}
            title={`${s.name}: ${s.value}`}
          />
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
        {segments.map((s) => (
          <div key={s.name} className="flex items-center gap-2 text-xs">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} />
            <span className="text-slate-400">{s.name}</span>
            <span className="font-semibold tabular-nums text-slate-200">{s.value}</span>
            <span className="text-slate-600">({((s.value / total) * 100).toFixed(0)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Coloured pill for a categorical cell, so the table scans like a status board. */
function Chip({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium"
      style={{ color, borderColor: `${color}55`, background: `${color}14` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

/** Inline bar behind a numeric cell -- magnitude at a glance without a chart. */
function CellBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-10 text-right text-sm font-semibold tabular-nums text-slate-200">
        {value.toFixed(1)}
      </span>
      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-800">
        <span
          className="block h-full rounded-full"
          style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }}
        />
      </span>
    </div>
  );
}

const TABS = [
  { key: "action", label: "Immediate action" },
  { key: "critical", label: "Critical health" },
  { key: "anomalous", label: "Anomalous" },
  { key: "worst", label: "Lowest health" },
] as const;
type TabKey = (typeof TABS)[number]["key"];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function Overview() {
  const { summary, latest, isLoading, error, dataUpdatedAt } = useFleetSummary();
  const modes = useFleetModes();
  const queue = useMaintenanceQueue();
  const agent = useAgentStatus();
  const anomalyEval = useAnomalyEvaluation();
  const rulEval = useEvaluationMetrics();
  const { refreshSeconds } = usePreferences();

  const [tab, setTab] = useState<TabKey>("action");
  const [picked, setPicked] = useState<string | number | null>(null);

  // Every derivation sits above the early returns so the hook order is stable
  // across the loading, error and ready renders (rules of hooks).
  const ranked = useMemo(
    () =>
      [...latest]
        .filter((r) => r.health_index !== null)
        .sort((a, b) => (a.health_index ?? 0) - (b.health_index ?? 0)),
    [latest],
  );

  const focusId = picked ?? ranked[0]?.engine_id;
  const { rows: engineRows } = useEngineRows(focusId);

  /** Engines the supervisor says to act on now, soonest failure first. */
  const actionable = useMemo(
    () =>
      latest
        .filter((r) => r.final_decision === "immediate_maintenance")
        .sort((a, b) => (a.predicted_rul ?? 1e9) - (b.predicted_rul ?? 1e9)),
    [latest],
  );

  /** Threshold-crossing counts per horizon -- the fleet's forward exposure. */
  const riskExposure = useMemo(
    () => [
      { name: "≤ 10 cycles", engines: latest.filter((r) => r.predicted_fail_in_10).length, color: CRITICAL },
      { name: "≤ 30 cycles", engines: latest.filter((r) => r.predicted_fail_in_30).length, color: WARNING },
      { name: "≤ 50 cycles", engines: latest.filter((r) => r.predicted_fail_in_50).length, color: DEGRADING },
    ],
    [latest],
  );

  /** Health vs RUL, grouped by state so the legend carries identity. */
  const matrix = useMemo(() => {
    const groups: Record<string, Array<Record<string, unknown>>> = {};
    for (const r of latest) {
      if (r.health_index === null || r.predicted_rul === null) continue;
      const state = r.health_state ?? "Unknown";
      (groups[state] ??= []).push({
        id: r.engine_id,
        rul: Number((r.predicted_rul ?? 0).toFixed(1)),
        health: Number((r.health_index ?? 0).toFixed(1)),
        risk: Number(((r.risk_30 ?? 0) * 100).toFixed(1)),
        decision: pretty(String(r.final_decision ?? "—")),
      });
    }
    return groups;
  }, [latest]);

  /** RUL histogram in 20-cycle bins -- the shape of the fleet's remaining life. */
  const rulBins = useMemo(() => {
    const width = 20;
    const bins = new Map<number, number>();
    let max = 0;
    for (const r of latest) {
      if (r.predicted_rul === null) continue;
      const b = Math.floor(Math.max(0, r.predicted_rul) / width) * width;
      bins.set(b, (bins.get(b) ?? 0) + 1);
      max = Math.max(max, b);
    }
    const out: Array<{ name: string; engines: number; start: number }> = [];
    for (let b = 0; b <= max; b += width) {
      out.push({ name: `${b}–${b + width}`, engines: bins.get(b) ?? 0, start: b });
    }
    return out;
  }, [latest]);

  /** Attributed root causes at the latest cycle (anomalous engines only). */
  const rootCauses = useMemo(() => {
    const t = tallyBy(latest, "root_cause");
    return Object.entries(t)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
  }, [latest]);

  const attributed = useMemo(() => latest.filter((r) => r.root_cause).length, [latest]);
  const lifecycleMix = useMemo(() => tallyBy(latest, "lifecycle_state"), [latest]);
  const riskStateMix = useMemo(() => tallyBy(latest, "risk_state"), [latest]);

  /** Governance: how often fusion overrode the model, and the review load. */
  const governance = useMemo(() => {
    const n = latest.length || 1;
    const overridden = latest.filter((r) => r.override_applied).length;
    const review = latest.filter((r) => r.requires_human_review).length;
    const conflict = latest.reduce((a, r) => a + (r.conflict_score ?? 0), 0) / n;
    const entropy = latest.reduce((a, r) => a + (r.entropy_score ?? 0), 0) / n;
    const overrideReasons = tallyBy(latest.filter((r) => r.override_applied), "override_reason");
    return { overridden, review, conflict, entropy, overrideReasons, n: latest.length };
  }, [latest]);

  /**
   * The queue endpoint returns flagged CYCLES, not engines -- 669 rows across
   * 77 units. Both are reported, because a headline of "669" reads as a fleet
   * ten times larger than the one being maintained.
   */
  const queueStats = useMemo(() => {
    const rows = queue.data?.queue ?? [];
    return {
      rows: queue.data?.rows ?? 0,
      engines: new Set(rows.map((r) => String(r.engine_id))).size,
    };
  }, [queue.data]);

  const trajectory = useMemo(
    () =>
      engineRows.slice(-80).map((r) => ({
        cycle: r.cycle,
        health: Number((r.health_index ?? 0).toFixed(1)),
        risk30: Number(((r.risk_30 ?? 0) * 100).toFixed(1)),
      })),
    [engineRows],
  );

  const tableRows = useMemo(() => {
    const pick =
      tab === "action"
        ? actionable
        : tab === "critical"
          ? ranked.filter((r) => (r.health_state ?? "").toLowerCase() === "critical")
          : tab === "anomalous"
            ? [...latest]
                .filter((r) => r.is_anomalous)
                .sort((a, b) => (b.anomaly_score ?? 0) - (a.anomaly_score ?? 0))
            : ranked;
    return pick.slice(0, 12);
  }, [tab, actionable, ranked, latest]);

  if (isLoading) return <Loading label="Loading fleet intelligence…" />;
  if (error) return <ErrorState error={error} />;
  if (!summary) return <Empty label="No decisions available yet." />;

  const focus = latest.find((r) => String(r.engine_id) === String(focusId)) ?? ranked[0];

  // Known states first, then anything else the backend emitted -- an unexpected
  // state must still appear, or the shares would silently not sum to the fleet.
  const stateKeys = [
    ...STATE_ORDER.filter((s) => summary.healthStateCounts[s] !== undefined),
    ...Object.keys(summary.healthStateCounts).filter((s) => !STATE_ORDER.includes(s)),
  ];

  const statusChart = stateKeys.map((name) => ({
    name,
    value: summary.healthStateCounts[name],
    color: STATE_COLOR[name] ?? NEUTRAL,
  }));

  const decisionChart = DECISION_ORDER.filter((k) => summary.decisionCounts[k] !== undefined).map((k) => ({
    key: k,
    name: pretty(k),
    value: summary.decisionCounts[k],
    color: DECISION_COLOR[k] ?? NEUTRAL,
  }));

  const anom = anomalyEval.test;
  const healthEval = anomalyEval.health;
  const rulModel = rulEval.model;
  const healthyShare = summary.engines ? summary.healthyCount / summary.engines : 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Executive Overview"
        subtitle="Fleet condition, forward risk and decision quality — one screen, one source of truth."
        right={
          <div className="flex flex-col items-end gap-1">
            <LiveBadge />
            <span className="text-[11px] text-slate-500">
              Updated {dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : "—"}
              {refreshSeconds > 0 ? ` · every ${refreshSeconds}s` : " · auto-refresh off"}
            </span>
          </div>
        }
      />

      {/* Provenance strip -- states exactly what every figure below is computed on. */}
      <Reveal>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-slate-800/60 bg-slate-900/40 px-4 py-3 text-xs">
          <span className="flex items-center gap-2 text-slate-400">
            <Database className="h-3.5 w-3.5 text-cyan-400" />
            <span className="font-semibold text-slate-200">{summary.engines}</span> engines ·{" "}
            <span className="font-semibold text-slate-200">{summary.rows.toLocaleString()}</span> scored cycles
          </span>
          <span className="flex items-center gap-2 text-slate-400">
            <Layers className="h-3.5 w-3.5 text-cyan-400" />
            Subset {summary.subsets.length ? summary.subsets.join(", ") : "—"}
          </span>
          <span className="flex items-center gap-2 text-slate-400">
            <Timer className="h-3.5 w-3.5 text-cyan-400" />
            Basis: latest cycle per engine
          </span>
          <span className="flex items-center gap-2 text-slate-400">
            <Cpu className="h-3.5 w-3.5 text-cyan-400" />
            Models {agent.modelActive ? "loaded" : "unverified"}
          </span>
          <span className="ml-auto flex items-center gap-2">
            <Signal className={`h-3.5 w-3.5 ${agent.apiConnected ? "text-emerald-400" : "text-rose-400"}`} />
            <span className={agent.apiConnected ? "text-emerald-400" : "text-rose-400"}>
              API {agent.apiConnected ? "connected" : "unreachable"}
            </span>
          </span>
        </div>
      </Reveal>

      {/* ------------------------------ KPI band ------------------------------ */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {[
          {
            title: "Fleet Health Score",
            node: <AnimatedNumber value={summary.avgHealth} decimals={1} suffix="%" />,
            icon: Gauge,
            status: summary.avgHealth > 70 ? "success" : summary.avgHealth > 45 ? "warning" : "danger",
            subtitle: `${(healthyShare * 100).toFixed(0)}% of the fleet is healthy`,
          },
          {
            // Counts BOTH worst bands. Naming only "Critical" while the bottom
            // "Failure Risk" band exists is what let 11 engines go unreported.
            title: "Critical / Failure Risk",
            node: <AnimatedNumber value={summary.criticalCount} />,
            icon: ShieldAlert,
            status: "danger",
            subtitle: summary.failureRiskCount
              ? `${summary.criticalCount - summary.failureRiskCount} critical · ${summary.failureRiskCount} failure risk`
              : `${((summary.criticalCount / (summary.engines || 1)) * 100).toFixed(1)}% of fleet`,
          },
          {
            title: "Degrading / Warning",
            node: <AnimatedNumber value={summary.warningCount} />,
            icon: Activity,
            status: "warning",
            subtitle: `${((summary.warningCount / (summary.engines || 1)) * 100).toFixed(1)}% of fleet`,
          },
          {
            title: "Anomalous Engines",
            node: <AnimatedNumber value={summary.anomalyCount} />,
            icon: TrendingUp,
            status: "info",
            subtitle: `${rootCauses.length} distinct root causes attributed`,
          },
          {
            title: "Mean Predicted RUL",
            node: <AnimatedNumber value={summary.avgRul} decimals={0} />,
            icon: Timer,
            status: "neutral",
            subtitle: "cycles remaining at latest window",
          },
          {
            title: "Maintenance Queue",
            node: <AnimatedNumber value={queueStats.engines} />,
            icon: Wrench,
            status: "warning",
            subtitle: `${queueStats.rows.toLocaleString()} flagged cycles queued`,
          },
        ].map((k, i) => (
          <Reveal key={k.title} index={i}>
            <KPICard
              title={k.title}
              value={k.node}
              icon={k.icon}
              status={k.status as never}
              subtitle={k.subtitle}
            />
          </Reveal>
        ))}
      </div>

      {/* ---------------------- Situation + act-now list ---------------------- */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Reveal index={1} className="xl:col-span-2">
          <GlassCard variant="primary" className="h-full">
            <div className="flex items-start gap-4">
              <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 p-3">
                <Brain className="h-6 w-6 text-cyan-400" />
              </div>
              <div className="flex-1">
                <h3 className="mb-2 text-lg font-semibold text-slate-200">Fleet Situation</h3>
                <p className="text-sm leading-relaxed text-slate-300">
                  Mean health index is{" "}
                  <span className="font-semibold text-cyan-400">{summary.avgHealth.toFixed(1)}</span> across{" "}
                  <span className="font-semibold text-slate-100">{summary.engines}</span> engines at their latest
                  cycle: <span className="font-semibold text-rose-400">{summary.criticalCount}</span>{" "}
                  critical or at failure risk,{" "}
                  <span className="font-semibold text-amber-400">{summary.warningCount}</span> degrading or in
                  warning, <span className="font-semibold text-emerald-400">{summary.healthyCount}</span> healthy.{" "}
                  <span className="font-semibold text-rose-400">{actionable.length}</span>{" "}
                  {actionable.length === 1 ? "engine is" : "engines are"} on an immediate-maintenance decision, and{" "}
                  <span className="font-semibold text-amber-400">{riskExposure[1].engines}</span> cross the
                  30-cycle failure threshold.
                  {focus && (
                    <>
                      {" "}
                      The most degraded unit is{" "}
                      <span className="font-semibold text-slate-100">E{String(focus.engine_id)}</span> at health{" "}
                      <span className="font-semibold text-rose-400">{fmt(focus.health_index)}</span> with{" "}
                      <span className="font-semibold text-slate-100">{fmt(focus.predicted_rul, 0)}</span> cycles
                      predicted
                      {focus.root_cause ? (
                        <>
                          , attributed to <span className="text-purple-300">{focus.root_cause}</span>
                        </>
                      ) : null}
                      .
                    </>
                  )}
                </p>
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Metric
                    label="Mean confidence"
                    value={pct(summary.avgConfidence)}
                    hint="Supervisor fusion"
                    tone={summary.avgConfidence > 0.7 ? "good" : summary.avgConfidence > 0.5 ? "warn" : "bad"}
                  />
                  <Metric
                    label="Human review"
                    value={`${governance.review}/${governance.n}`}
                    hint="Escalated by the confidence gate"
                    tone={governance.review > governance.n / 2 ? "warn" : "good"}
                  />
                  <Metric
                    label="Safety overrides"
                    value={`${governance.overridden}`}
                    hint="Fusion overruled the model"
                  />
                  <Metric
                    label="Mean conflict"
                    value={fmt(governance.conflict, 2)}
                    hint="Cross-agent disagreement"
                    tone={governance.conflict > 0.4 ? "warn" : "good"}
                  />
                </div>
              </div>
            </div>
          </GlassCard>
        </Reveal>

        <Reveal index={2}>
          <MetricCard
            title="Act now"
            subtitle={`${actionable.length} engines on an immediate-maintenance decision`}
            className="h-full"
          >
            {actionable.length ? (
              <div className="space-y-2">
                {actionable.slice(0, 5).map((r) => (
                  <button
                    key={String(r.engine_id)}
                    onClick={() => setPicked(r.engine_id)}
                    className="flex w-full items-center gap-3 rounded-lg border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-left transition-colors hover:border-rose-500/50 hover:bg-rose-500/10"
                  >
                    <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-semibold text-slate-100">Engine E{String(r.engine_id)}</div>
                      <div className="truncate text-[11px] text-slate-500">
                        {r.root_cause ?? r.override_reason ?? "Supervisor escalation"}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-semibold tabular-nums text-rose-400">
                        {fmt(r.predicted_rul, 0)}
                      </div>
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">cycles</div>
                    </div>
                  </button>
                ))}
                {actionable.length > 5 && (
                  <p className="pt-1 text-center text-[11px] text-slate-500">
                    +{actionable.length - 5} more in the priority board below
                  </p>
                )}
              </div>
            ) : (
              <Empty label="No engine currently requires immediate maintenance." />
            )}
          </MetricCard>
        </Reveal>
      </div>

      {/* --------------------------- Fleet condition --------------------------- */}
      <Section
        eyebrow="Condition"
        title="Where the fleet stands"
        hint="Every figure at each engine's latest recorded cycle"
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Reveal index={1} className="lg:col-span-2">
          <MetricCard
            title="Fleet composition"
            subtitle={`${summary.engines} engines by health state, lifecycle stage and failure-risk state`}
          >
            <div className="mb-6">
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Health state</div>
              <CompositionBar segments={statusChart} total={summary.engines} />
            </div>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              <div>
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                  Lifecycle stage
                </div>
                <CompositionBar
                  segments={Object.entries(lifecycleMix)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value], i) => ({
                      name: pretty(name),
                      value,
                      color: ORDINAL[i] ?? NEUTRAL,
                    }))}
                  total={Object.values(lifecycleMix).reduce((a, b) => a + b, 0)}
                />
              </div>
              <div>
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                  Failure-risk state
                </div>
                <CompositionBar
                  segments={Object.entries(riskStateMix)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value], i) => ({
                      name: pretty(name),
                      value,
                      color: ORDINAL[i] ?? NEUTRAL,
                    }))}
                  total={Object.values(riskStateMix).reduce((a, b) => a + b, 0)}
                />
              </div>
            </div>
          </MetricCard>
        </Reveal>

        <Reveal index={2}>
          <MetricCard
            title="Forward risk exposure"
            subtitle="Engines above the adaptive per-horizon failure threshold"
          >
            <ResponsiveContainer width="100%" height={228}>
              <BarChart data={riskExposure} margin={{ top: 18, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                <XAxis dataKey="name" stroke={AXIS} tickLine={false} axisLine={{ stroke: GRID }} fontSize={11} />
                <YAxis stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} allowDecimals={false} />
                <Tooltip cursor={{ fill: "#ffffff08" }} content={<ChartTip />} />
                <Bar dataKey="engines" name="Engines" radius={[4, 4, 0, 0]} maxBarSize={54} animationDuration={800}>
                  {riskExposure.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                  <LabelList dataKey="engines" position="top" fill="#e2e8f0" fontSize={12} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-2 text-[11px] leading-snug text-slate-500">
              A horizon counts an engine only when its predicted failure probability exceeds that horizon's own
              calibrated threshold, so the three bars are not nested subsets of one another.
            </p>
          </MetricCard>
        </Reveal>
      </div>

      {/* ------------------------------ Risk matrix ------------------------------ */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Reveal index={1} className="lg:col-span-2">
          <MetricCard
            title="Fleet risk matrix — health against remaining life"
            subtitle="One dot per engine · dot size = 30-cycle failure probability · the lower-left quadrant is the action zone"
          >
            <ResponsiveContainer width="100%" height={330}>
              <ScatterChart margin={{ top: 10, right: 18, bottom: 16, left: -8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                <XAxis
                  type="number"
                  dataKey="rul"
                  name="Predicted RUL"
                  stroke={AXIS}
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: GRID }}
                  label={{
                    value: "Predicted RUL (cycles)",
                    position: "insideBottom",
                    offset: -8,
                    fill: AXIS,
                    fontSize: 11,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="health"
                  name="Health index"
                  domain={[0, 100]}
                  stroke={AXIS}
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  label={{
                    value: "Health index",
                    angle: -90,
                    position: "insideLeft",
                    offset: 22,
                    fill: AXIS,
                    fontSize: 11,
                  }}
                />
                <ZAxis type="number" dataKey="risk" range={[36, 300]} name="Risk @30" />
                <ReferenceLine
                  y={50}
                  stroke={WARNING}
                  strokeDasharray="4 4"
                  label={{ value: "health 50", fill: WARNING, fontSize: 10, position: "insideTopRight" }}
                />
                <ReferenceLine
                  x={30}
                  stroke={CRITICAL}
                  strokeDasharray="4 4"
                  label={{ value: "30 cycles", fill: CRITICAL, fontSize: 10, position: "insideTopLeft" }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3", stroke: AXIS }}
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const p = payload[0].payload;
                    return (
                      <div className="rounded-lg border border-slate-700/70 bg-slate-950/95 px-3 py-2 text-xs shadow-xl">
                        <div className="mb-1 font-semibold text-slate-100">Engine E{String(p.id)}</div>
                        <div className="text-slate-400">
                          Health <span className="font-semibold text-slate-200">{p.health}</span> · RUL{" "}
                          <span className="font-semibold text-slate-200">{p.rul}</span> cyc
                        </div>
                        <div className="text-slate-400">
                          Risk @30 <span className="font-semibold text-slate-200">{p.risk}%</span>
                        </div>
                        <div className="mt-1 text-slate-500">{p.decision}</div>
                      </div>
                    );
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                {[
                  ...STATE_ORDER,
                  ...Object.keys(matrix).filter((s) => !STATE_ORDER.includes(s)),
                ]
                  .filter((s) => matrix[s]?.length)
                  .map((s) => (
                    <Scatter
                      key={s}
                      name={s}
                      data={matrix[s] as never}
                      fill={STATE_COLOR[s] ?? NEUTRAL}
                      fillOpacity={0.72}
                      stroke="#0f172a"
                      strokeWidth={2}
                      animationDuration={800}
                    />
                  ))}
              </ScatterChart>
            </ResponsiveContainer>
          </MetricCard>
        </Reveal>

        <Reveal index={2}>
          <MetricCard title="Remaining-life distribution" subtitle="Engines per 20-cycle band of predicted RUL">
            <ResponsiveContainer width="100%" height={330}>
              <BarChart data={rulBins} margin={{ top: 16, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                <XAxis dataKey="name" stroke={AXIS} fontSize={10} tickLine={false} axisLine={{ stroke: GRID }} />
                <YAxis stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip cursor={{ fill: "#ffffff08" }} content={<ChartTip />} />
                <Bar dataKey="engines" name="Engines" radius={[4, 4, 0, 0]} animationDuration={800}>
                  {rulBins.map((d) => (
                    <Cell
                      key={d.name}
                      fill={d.start < 40 ? CRITICAL : MAGNITUDE}
                      fillOpacity={d.start < 40 ? 0.9 : 0.75}
                    />
                  ))}
                  <LabelList dataKey="engines" position="top" fill="#94a3b8" fontSize={10} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-2 text-[11px] leading-snug text-slate-500">
              Bands under 40 cycles are highlighted: those engines fall inside the maintenance planning horizon.
            </p>
          </MetricCard>
        </Reveal>
      </div>

      {/* --------------------------- Decision layer --------------------------- */}
      <Section
        eyebrow="Decision layer"
        title="What the supervisor decided, and why"
        hint="Fused from RUL, failure-risk and anomaly evidence"
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Reveal index={1}>
          <MetricCard title="Supervisor decisions" subtitle="Escalation ladder, least to most urgent">
            <ResponsiveContainer width="100%" height={268}>
              <BarChart data={decisionChart} layout="vertical" margin={{ left: 44, right: 30, top: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
                <XAxis type="number" stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} />
                <YAxis
                  dataKey="name"
                  type="category"
                  stroke={AXIS}
                  width={132}
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip cursor={{ fill: "#ffffff08" }} content={<ChartTip />} />
                <Bar dataKey="value" name="Engines" radius={[0, 4, 4, 0]} maxBarSize={26} animationDuration={800}>
                  {decisionChart.map((d) => (
                    <Cell key={d.key} fill={d.color} />
                  ))}
                  <LabelList dataKey="value" position="right" fill="#94a3b8" fontSize={11} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </MetricCard>
        </Reveal>

        <Reveal index={2}>
          <MetricCard
            title="Attributed root causes"
            subtitle={`${attributed} of ${summary.engines} engines carry an attribution`}
          >
            {rootCauses.length ? (
              <ResponsiveContainer width="100%" height={268}>
                <BarChart data={rootCauses} layout="vertical" margin={{ left: 4, right: 30, top: 4, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
                  <XAxis type="number" stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    stroke={AXIS}
                    width={168}
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip cursor={{ fill: "#ffffff08" }} content={<ChartTip />} />
                  <Bar
                    dataKey="value"
                    name="Engines"
                    fill={MAGNITUDE}
                    radius={[0, 4, 4, 0]}
                    maxBarSize={22}
                    animationDuration={800}
                  >
                    <LabelList dataKey="value" position="right" fill="#94a3b8" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty label="No root cause attributed at the latest cycle." />
            )}
          </MetricCard>
        </Reveal>

        <Reveal index={3}>
          <MetricCard title="Decision quality" subtitle="Confidence, conflict and the review gate">
            <div className="grid grid-cols-2 gap-3">
              <Metric
                label="Mean confidence"
                value={pct(summary.avgConfidence)}
                tone={summary.avgConfidence > 0.7 ? "good" : summary.avgConfidence > 0.5 ? "warn" : "bad"}
              />
              <Metric label="Mean entropy" value={fmt(governance.entropy, 2)} hint="0 = fully decisive" />
              <Metric
                label="Review required"
                value={`${((governance.review / (governance.n || 1)) * 100).toFixed(0)}%`}
                hint={`${governance.review} of ${governance.n} engines`}
                tone={governance.review > governance.n / 2 ? "warn" : "good"}
              />
              <Metric
                label="Override rate"
                value={`${((governance.overridden / (governance.n || 1)) * 100).toFixed(0)}%`}
                hint="Safety rules beat the model"
              />
            </div>
            {Object.keys(governance.overrideReasons).length > 0 && (
              <div className="mt-4">
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                  Why fusion overrode the model
                </div>
                <div className="space-y-1.5">
                  {Object.entries(governance.overrideReasons)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 4)
                    .map(([reason, n]) => (
                      <div key={reason} className="flex items-center gap-2 text-xs">
                        <span className="flex-1 truncate text-slate-400">{reason}</span>
                        <span className="font-semibold tabular-nums text-slate-200">{n}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}
            <p className="mt-4 text-[11px] leading-snug text-slate-500">
              Every decision is published with its own confidence; the gate escalates anything below the threshold
              to a human rather than presenting it as settled.
            </p>
          </MetricCard>
        </Reveal>
      </div>

      {/* ----------------------------- Focus engine ----------------------------- */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Reveal index={1} className="lg:col-span-2">
          <MetricCard
            title={focus ? `Engine E${String(focus.engine_id)} — degradation trajectory` : "Trajectory"}
            subtitle="Health index and 30-cycle failure probability over the engine's own life · pick an engine in the board below"
          >
            {trajectory.length ? (
              <ResponsiveContainer width="100%" height={286}>
                <ComposedChart data={trajectory} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
                  <defs>
                    <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CRITICAL} stopOpacity={0.35} />
                      <stop offset="100%" stopColor={CRITICAL} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                  <XAxis dataKey="cycle" stroke={AXIS} fontSize={11} tickLine={false} axisLine={{ stroke: GRID }} />
                  {/* One axis only: both series are 0-100 scales (an index and a percentage). */}
                  <YAxis domain={[0, 100]} stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip content={<ChartTip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={50} stroke={WARNING} strokeDasharray="4 4" />
                  <Area
                    type="monotone"
                    dataKey="risk30"
                    name="Failure probability @30 (%)"
                    stroke={CRITICAL}
                    strokeWidth={2}
                    fill="url(#riskFill)"
                    animationDuration={800}
                  />
                  <Line
                    type="monotone"
                    dataKey="health"
                    name="Health index"
                    stroke={INFO}
                    strokeWidth={2}
                    dot={false}
                    animationDuration={800}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <Empty label="No cycle history for this engine." />
            )}
          </MetricCard>
        </Reveal>

        <Reveal index={2}>
          <MetricCard
            title="Selected engine"
            subtitle={focus ? `E${String(focus.engine_id)} at cycle ${focus.cycle}` : "—"}
          >
            {focus ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <StatusBadge status={statusFor(focus.health_state)} label={focus.health_state ?? "—"} size="sm" />
                  <Chip
                    label={String(focus.priority ?? "—")}
                    color={PRIORITY_COLOR[String(focus.priority)] ?? NEUTRAL}
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Metric label="Health index" value={fmt(focus.health_index)} />
                  <Metric label="Predicted RUL" value={`${fmt(focus.predicted_rul, 0)} cyc`} />
                  <Metric label="Risk @30" value={pct(focus.risk_30, 0)} />
                  <Metric label="Anomaly score" value={fmt(focus.anomaly_score, 2)} />
                </div>
                <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-slate-500">Decision</div>
                  <div className="mt-1 text-sm font-semibold text-slate-200">
                    {pretty(String(focus.final_decision ?? "—"))}
                  </div>
                  {focus.recommended_action && (
                    <p className="mt-1.5 text-[11px] leading-snug text-slate-400">{focus.recommended_action}</p>
                  )}
                  {focus.override_applied && (
                    <p className="mt-1.5 text-[11px] text-amber-400">
                      Overridden: {focus.override_reason ?? "safety rule"}
                    </p>
                  )}
                </div>
                <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-slate-500">Attributed cause</div>
                  <div className="mt-1 text-sm text-purple-300">{focus.root_cause ?? "No fault attributed"}</div>
                </div>
              </div>
            ) : (
              <Empty />
            )}
          </MetricCard>
        </Reveal>
      </div>

      {/* ---------------------------- Model evidence ---------------------------- */}
      <Section
        eyebrow="Assurance"
        title="Model evidence on held-out data"
        hint="Offline validation against ground truth the serving API never sees"
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Reveal index={1}>
          <MetricCard
            title="Anomaly & health detection"
            subtitle={
              anom?.rows
                ? `Held-out test split · ${anom.rows.toLocaleString()} rows · ${anom.engines} engines`
                : "Held-out test split"
            }
          >
            {anomalyEval.available ? (
              <>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Metric
                    label="AUC"
                    value={fmt(anom?.auc_healthy_vs_critical, 3)}
                    hint="Healthy vs critical"
                    tone={(anom?.auc_healthy_vs_critical ?? 0) > 0.9 ? "good" : "warn"}
                  />
                  <Metric
                    label="Detection rate"
                    value={pct(anom?.detection_rate, 0)}
                    hint="Failing engines caught"
                    tone={(anom?.detection_rate ?? 0) > 0.9 ? "good" : "warn"}
                  />
                  <Metric
                    label="False alarms"
                    value={pct(anom?.false_alarm_rate_healthy, 1)}
                    hint="Raised on healthy rows"
                    tone={(anom?.false_alarm_rate_healthy ?? 1) < 0.1 ? "good" : "warn"}
                  />
                  <Metric
                    label="Median lead time"
                    value={anom?.lead_time_median !== undefined ? `${anom.lead_time_median} cyc` : "—"}
                    hint="Warning raised before failure"
                    tone="good"
                  />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
                  <Metric
                    label="Health–RUL correlation"
                    value={fmt(healthEval?.rul_correlation, 2)}
                    hint="Index tracks true remaining life"
                  />
                  <Metric
                    label="Monotonicity"
                    value={fmt(healthEval?.monotonicity_score, 3)}
                    hint="Index degrades without rebounding"
                    tone={(healthEval?.monotonicity_score ?? 0) > 0.9 ? "good" : "warn"}
                  />
                  <Metric
                    label="Separation (Cohen's d)"
                    value={fmt(anom?.cohens_d, 2)}
                    hint="Healthy vs critical score gap"
                  />
                </div>
              </>
            ) : (
              <Empty label="Evaluation artefact has not been exported yet." />
            )}
          </MetricCard>
        </Reveal>

        <Reveal index={2}>
          <MetricCard
            title="Remaining-life prediction"
            subtitle={
              rulModel?.engines_scored
                ? `${rulModel.name ?? "RUL model"} · ${rulModel.engines_scored} held-out engines`
                : "Held-out C-MAPSS test engines"
            }
          >
            {rulEval.available && rulModel ? (
              <>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Metric label="MAE" value={`${fmt(rulModel.mae, 1)} cyc`} hint="Mean absolute error" />
                  <Metric label="RMSE" value={`${fmt(rulModel.rmse, 1)} cyc`} hint="Penalises large misses" />
                  <Metric
                    label="R²"
                    value={fmt(rulModel.r2, 3)}
                    hint="Variance explained"
                    tone={(rulModel.r2 ?? 0) > 0.5 ? "good" : "warn"}
                  />
                  <Metric label="NASA score" value={fmt(rulModel.nasa_score, 0)} hint="Late-prediction penalty" />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <Metric
                    label="Interval coverage (PICP)"
                    value={pct(rulEval.picp, 1)}
                    hint="Truth inside the predicted band"
                    tone={(rulEval.picp ?? 0) > 0.8 ? "good" : "warn"}
                  />
                  <Metric
                    label="Interval width (MPIW)"
                    value={`${fmt(rulEval.mpiw, 1)} cyc`}
                    hint="Mean prediction-interval width"
                  />
                </div>
                <p className="mt-3 text-[11px] leading-snug text-slate-500">
                  The interval is a fixed conformal width rather than one re-calibrated per prediction, so coverage
                  below the nominal level is reported here rather than smoothed away.
                </p>
              </>
            ) : (
              <Empty label="Evaluation artefact has not been exported yet." />
            )}
          </MetricCard>
        </Reveal>
      </div>

      {/* --------------------------- Operating context --------------------------- */}
      {modes.data && (
        <Reveal index={3}>
          <GlassCard variant={modes.data.drift.drift_detected ? "warning" : "success"}>
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-start gap-3">
                <div
                  className={`rounded-lg border p-2.5 ${
                    modes.data.drift.drift_detected
                      ? "border-amber-500/30 bg-amber-500/10"
                      : "border-emerald-500/30 bg-emerald-500/10"
                  }`}
                >
                  {modes.data.drift.drift_detected ? (
                    <AlertTriangle className="h-5 w-5 text-amber-400" />
                  ) : (
                    <BadgeCheck className="h-5 w-5 text-emerald-400" />
                  )}
                </div>
                <div>
                  <h3 className="font-semibold text-slate-200">Operating regime & input drift</h3>
                  <p className="mt-1 text-sm text-slate-400">
                    {Object.keys(modes.data.mode_distribution).length} flight regimes detected · PSI{" "}
                    <span className="font-semibold text-slate-200">{modes.data.drift.psi.toFixed(3)}</span> against a
                    threshold of {modes.data.drift.threshold} ·{" "}
                    {modes.data.drift.drift_detected ? (
                      <span className="text-amber-400">
                        drift {modes.data.drift.severity} — predictions may be off-distribution
                      </span>
                    ) : (
                      <span className="text-emerald-400">
                        stable — inputs match the distribution the models were fitted on
                      </span>
                    )}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(modes.data.mode_distribution)
                  .sort((a, b) => Number(a[0]) - Number(b[0]))
                  .map(([m, n], i) => (
                    <motion.div
                      key={m}
                      initial={{ opacity: 0, scale: 0.86 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.05 }}
                      className="min-w-[64px] rounded-lg border border-slate-700/50 bg-slate-800/40 px-3 py-2 text-center"
                    >
                      <div className="text-[10px] uppercase tracking-wide text-slate-500">Mode {m}</div>
                      <div className="font-semibold tabular-nums text-cyan-300">{n}</div>
                    </motion.div>
                  ))}
              </div>
            </div>
          </GlassCard>
        </Reveal>
      )}

      {/* ---------------------------- Priority board ---------------------------- */}
      <Section
        eyebrow="Work queue"
        title="Priority engines"
        hint="Select a row to drive the trajectory chart above"
      />

      <Reveal index={1}>
        <MetricCard
          title="Engine priority board"
          subtitle={`Showing ${tableRows.length} engines · ${TABS.find((t) => t.key === tab)?.label}`}
          actions={
            <div className="flex flex-wrap gap-1 rounded-lg border border-slate-800/60 bg-slate-900/40 p-1">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setTab(t.key)}
                  className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                    tab === t.key
                      ? "bg-cyan-500/15 text-cyan-300"
                      : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          }
        >
          {tableRows.length ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[880px]">
                <thead>
                  <tr className="border-b border-slate-800/60">
                    {[
                      "Engine",
                      "State",
                      "Health",
                      "RUL",
                      "Risk @30",
                      "Decision",
                      "Priority",
                      "Confidence",
                      "Attributed cause",
                    ].map((h) => (
                      <th
                        key={h}
                        className="px-3 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-500"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((r) => {
                    const active = String(r.engine_id) === String(focusId);
                    const hi = r.health_index ?? 0;
                    return (
                      <tr
                        key={String(r.engine_id)}
                        onClick={() => setPicked(r.engine_id)}
                        className={`cursor-pointer border-b border-slate-800/30 transition-colors ${
                          active ? "bg-cyan-500/10" : "hover:bg-slate-800/30"
                        }`}
                      >
                        <td className="px-3 py-2.5 text-sm font-semibold text-slate-200">
                          E{String(r.engine_id)}
                          <span className="ml-2 text-[10px] font-normal text-slate-500">c{r.cycle}</span>
                        </td>
                        <td className="px-3 py-2.5">
                          <StatusBadge status={statusFor(r.health_state)} label={r.health_state ?? "—"} size="sm" />
                        </td>
                        <td className="px-3 py-2.5">
                          <CellBar value={hi} color={hi > 70 ? HEALTHY : hi > 45 ? DEGRADING : CRITICAL} />
                        </td>
                        <td className="px-3 py-2.5 text-sm tabular-nums text-slate-300">{fmt(r.predicted_rul, 0)}</td>
                        <td className="px-3 py-2.5 text-sm tabular-nums text-slate-300">{pct(r.risk_30, 0)}</td>
                        <td className="px-3 py-2.5">
                          <Chip
                            label={pretty(String(r.final_decision ?? "—"))}
                            color={DECISION_COLOR[String(r.final_decision)] ?? NEUTRAL}
                          />
                        </td>
                        <td className="px-3 py-2.5">
                          <Chip
                            label={String(r.priority ?? "—")}
                            color={PRIORITY_COLOR[String(r.priority)] ?? NEUTRAL}
                          />
                        </td>
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-2">
                            <span className="text-sm tabular-nums text-slate-300">{pct(r.confidence_score, 0)}</span>
                            {r.requires_human_review && (
                              <span title="Escalated for human review">
                                <ClipboardList className="h-3.5 w-3.5 text-amber-400" />
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="max-w-[220px] truncate px-3 py-2.5 text-xs text-slate-400">
                          {r.root_cause ?? "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty label="No engines match this filter." />
          )}
          <p className="mt-3 text-[11px] text-slate-500">
            The clipboard marks a decision the confidence gate escalated for human review.
          </p>
        </MetricCard>
      </Reveal>
    </div>
  );
}
