/**
 * Failure Risk Agent -- dashboard.
 *
 * Single-engine view over the Failure Risk agent. Every figure is read from a
 * backend response; nothing is hardcoded. Where the backend cannot supply a
 * value the cell renders an em dash rather than inventing a number.
 *
 * Driven entirely by the last full-pipeline run: the fused decision table is
 * read back through `/integration/dashboard`, filtered to one engine. There is
 * no dataset picker, demo loader or "run prediction" control -- the subset is a
 * property of the uploaded file (the loader takes it from the filename so the
 * batch can select the matching per-subset model), so offering it as a choice
 * here let the screen score a window against a BiLSTM never fitted for it.
 *
 * Rows the batch sub-sampled past carry forward-filled risk scores; they are
 * dropped from the series rather than drawn, because a flat placeholder
 * segment reads as a real score on a chart.
 *
 * The remaining backend calls are unchanged:
 *
 *   thresholds         -> GET  /threshold/status         (EWA-adapted, per horizon)
 *   feedback           -> POST /threshold/feedback       (operator verdict)
 *   memory             -> GET  /memory/status
 *   artifacts          -> GET  /artifacts/status
 *
 * Component boundary: this page belongs to the Failure Risk module. It shows
 * `risk_10/30/50`, their thresholds, MC-dropout uncertainty, risk velocity and
 * `risk_state` -- all Failure-Risk-owned. It deliberately does NOT read
 * `predicted_rul`, `lower_bound`, `uncertainty_width` or `lifecycle_state`,
 * which belong to the RUL Lifecycle module and are shown on its own page. The
 * Engine Health gauge is therefore derived from the agent's own control risk,
 * not from a remaining-life estimate.
 */
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  useAgentStatus,
  useArtifactStatus,
  useEngineRows,
  useFleetSummary,
  useMemoryStatus,
  useThresholdFeedback,
  useThresholdStatus,
} from "../../lib/hooks";
import { useEngineParam } from "../../lib/engine-param";
import { NotificationCenter } from "../components/NotificationCenter";
import type { DecisionRow, HorizonThreshold } from "../../lib/types";

const HORIZONS = ["10", "30", "50"] as const;

/** Failure-risk states, in severity order -- the distribution donut. */
const RISK_STATES = ["Nominal", "Stable", "Rising", "Accelerating", "Critical"] as const;

/**
 * The fused decision table stores states lower-cased ("degrading") and adds a
 * "nominal" band of its own; the single-shot API this screen used to read
 * returned them capitalised. Every stage list, donut bucket and `===`
 * comparison below is written against the capitalised form, so the states are
 * normalised here at the data boundary rather than at a dozen render sites.
 */
const titleCase = (value: unknown) => {
  const text = String(value ?? "").trim();
  return text ? text[0].toUpperCase() + text.slice(1).toLowerCase() : "";
};

const DASH = "—";

const fmt = (v: number | null | undefined, d = 2) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : DASH;
const pct = (v: number | null | undefined, d = 0) =>
  typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : DASH;

function tone(state: string | undefined) {
  switch ((state ?? "").toLowerCase()) {
    case "critical":
      return { text: "text-red-400", dot: "bg-red-500", hex: "#ef4444" };
    case "accelerating":
      return { text: "text-orange-400", dot: "bg-orange-500", hex: "#f97316" };
    case "rising":
      return { text: "text-amber-400", dot: "bg-amber-500", hex: "#f59e0b" };
    default:
      return { text: "text-emerald-400", dot: "bg-emerald-500", hex: "#10b981" };
  }
}

/** Risk band for one horizon, judged against that horizon's live threshold. */
function riskBand(risk: number | undefined, threshold: number | undefined) {
  if (typeof risk !== "number") return { label: DASH, cls: "text-slate-500", ring: "border-slate-700" };
  if (typeof threshold === "number" && risk >= threshold)
    return { label: "HIGH RISK", cls: "text-red-400", ring: "border-red-500/60" };
  if (risk >= 0.25) return { label: "MEDIUM RISK", cls: "text-amber-400", ring: "border-amber-500/60" };
  return { label: "LOW RISK", cls: "text-emerald-400", ring: "border-emerald-500/60" };
}

function Panel({
  title,
  subtitle,
  right,
  footer,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex flex-col rounded-xl border border-slate-800/70 bg-slate-900/40 p-5 backdrop-blur-sm ${className}`}
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-200">{title}</h3>
          {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
        </div>
        {right}
      </div>
      <div className="flex-1">{children}</div>
      {footer && <div className="mt-3">{footer}</div>}
    </div>
  );
}

function Pill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs">
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-slate-600"}`} />
      <span className={ok ? "text-emerald-400" : "text-slate-500"}>{label}</span>
    </span>
  );
}

function DetailLink({ label, onClick }: { label: string; onClick?: () => void }) {
  return (
    <button onClick={onClick} className="text-xs text-slate-500 transition-colors hover:text-sky-400">
      {label} &rarr;
    </button>
  );
}

/** Ring gauge for the MC-dropout confidence. */
function ConfidenceRing({ value }: { value: number | null }) {
  const r = 46;
  const circ = 2 * Math.PI * r;
  const p = value === null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className="relative mx-auto h-32 w-32">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={r} className="fill-none stroke-slate-800" strokeWidth="11" />
        <circle
          cx="64"
          cy="64"
          r={r}
          className="fill-none stroke-sky-500"
          strokeWidth="11"
          strokeLinecap="round"
          strokeDasharray={`${circ * p} ${circ}`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-semibold text-slate-50">{pct(value)}</span>
        <span className="text-[10px] text-slate-500">confidence</span>
      </div>
    </div>
  );
}

/** Segmented donut over risk-state counts across the analysed trajectory. */
function RiskDonut({ counts, total }: { counts: Record<string, number>; total: number }) {
  const r = 52;
  const circ = 2 * Math.PI * r;
  let off = 0;
  const segs = RISK_STATES.map((s) => {
    const n = counts[s] ?? 0;
    const frac = total > 0 ? n / total : 0;
    const seg = { s, len: circ * frac, off, hex: tone(s).hex };
    off += circ * frac;
    return seg;
  }).filter((x) => x.len > 0);
  return (
    <div className="relative h-28 w-28 shrink-0">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={r} className="fill-none stroke-slate-800" strokeWidth="12" />
        {segs.map((x) => (
          <circle
            key={x.s}
            cx="64"
            cy="64"
            r={r}
            fill="none"
            stroke={x.hex}
            strokeWidth="12"
            strokeDasharray={`${x.len} ${circ - x.len}`}
            strokeDashoffset={-x.off}
          />
        ))}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-semibold text-slate-50">{total || 0}</span>
        <span className="text-[10px] text-slate-500">cycles</span>
      </div>
    </div>
  );
}

/** Semicircular gauge: engine health as the complement of control risk. */
function HealthGauge({ value }: { value: number | null }) {
  const p = value === null ? 0 : Math.max(0, Math.min(1, value / 100));
  const cx = 100;
  const cy = 92;
  const r = 68;
  const angle = Math.PI * (1 - p);
  const nx = cx + r * Math.cos(angle);
  const ny = cy - r * Math.sin(angle);
  const arc = (from: number, to: number, color: string) => {
    const a0 = Math.PI * (1 - from);
    const a1 = Math.PI * (1 - to);
    const x0 = cx + r * Math.cos(a0);
    const y0 = cy - r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy - r * Math.sin(a1);
    return <path key={color} d={`M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`} stroke={color} strokeWidth="14" fill="none" strokeLinecap="butt" />;
  };
  return (
    <div className="relative mx-auto w-[200px]">
      <svg viewBox="0 0 200 110" className="w-full">
        {arc(0, 0.33, "#ef4444")}
        {arc(0.33, 0.66, "#f59e0b")}
        {arc(0.66, 1, "#10b981")}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#e2e8f0" strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="5" fill="#e2e8f0" />
      </svg>
      <p className="-mt-1 text-center text-3xl font-semibold text-red-400">
        {value === null ? DASH : Math.round(value)}
      </p>
    </div>
  );
}

export function FailureRisk() {
  const [selected, setSelected] = useEngineParam();

  const status = useAgentStatus();
  const fleet = useFleetSummary();
  const memory = useMemoryStatus();
  const artifacts = useArtifactStatus();

  /** Engines the last full-pipeline run actually scored, in id order. */
  const engines = useMemo(
    () => [...fleet.latest].sort((a, b) => Number(a.engine_id) - Number(b.engine_id)),
    [fleet.latest],
  );

  // Falls back to the first scored engine until the operator picks one, so the
  // screen is populated as soon as the pipeline output loads.
  const engineId = selected || (engines.length ? String(engines[0].engine_id) : "");
  const engineRows = useEngineRows(engineId);
  const rows = engineRows.rows;

  /**
   * The engine's most recent scored cycle. `DecisionRow` carries every
   * risk-owned field this page reads, so it stands in for the single-shot
   * prediction response the screen used to be driven by.
   */
  const latestRow = rows.length ? rows[rows.length - 1] : null;
  const result: DecisionRow | null = useMemo(
    () => (latestRow ? { ...latestRow, risk_state: titleCase(latestRow.risk_state) } : null),
    [latestRow],
  );
  const fdSubset = String(result?.fd_subset ?? fleet.summary?.subsets[0] ?? "FD001");
  const thresholds = useThresholdStatus(fdSubset);
  const feedback = useThresholdFeedback();

  /**
   * Per-cycle series from the pipeline's own scored rows. Rows the batch
   * sub-sampled past carry forward-filled risk scores, so they are dropped
   * rather than drawn -- a flat placeholder segment reads as a real score.
   */
  const points = useMemo(
    () =>
      rows
        .filter((r) => typeof r.risk_30 === "number")
        .map((r) => ({
          cycle: r.cycle,
          risk_10: r.risk_10,
          risk_30: r.risk_30,
          risk_50: r.risk_50,
          risk_state: titleCase(r.risk_state),
          degradation_speed: r.degradation_speed ?? 0,
          degradation_acceleration: r.degradation_acceleration ?? 0,
        })),
    [rows],
  );

  const currentCycle = result ? Number(result.cycle) : null;
  const t = tone(result?.risk_state ?? undefined);

  const horizons = (thresholds.data?.horizons ?? {}) as Record<string, HorizonThreshold>;

  const series = useMemo(
    () =>
      points.map((p) => ({
        cycle: p.cycle,
        p10: p.risk_10 === null ? null : Number((p.risk_10 * 100).toFixed(2)),
        p30: p.risk_30 === null ? null : Number((p.risk_30 * 100).toFixed(2)),
        p50: p.risk_50 === null ? null : Number((p.risk_50 * 100).toFixed(2)),
        speed: Number(p.degradation_speed.toFixed(4)),
        accel: Number(p.degradation_acceleration.toFixed(4)),
      })),
    [points],
  );

  const riskCounts = useMemo(() => {
    const out: Record<string, number> = {};
    for (const p of points) if (p.risk_state) out[p.risk_state] = (out[p.risk_state] ?? 0) + 1;
    return out;
  }, [points]);

  /** Aggregates across the analysed trajectory. */
  const insights = useMemo(() => {
    if (!points.length) return null;
    const spd = points.map((p) => p.degradation_speed);
    const acc = points.map((p) => p.degradation_acceleration);
    const highRisk = points.filter((p) => (p.risk_30 ?? 0) >= 0.5).length;
    const critical = points.filter((p) => p.risk_state === "Critical").length;
    return {
      avgSpeed: spd.reduce((a, b) => a + b, 0) / spd.length,
      avgAccel: acc.reduce((a, b) => a + b, 0) / acc.length,
      highRisk,
      critical,
    };
  }, [points]);

  /** MC-dropout confidence: 1 - mean per-horizon standard deviation. */
  const confidence = result
    ? 1 -
      ((result.uncertainty_10 ?? 0) +
        (result.uncertainty_30 ?? 0) +
        (result.uncertainty_50 ?? 0)) /
        3
    : null;

  /** Engine health from the agent's own control risk -- see the boundary note. */
  const healthScore = result ? (1 - (result.control_risk ?? 0)) * 100 : null;

  const alertCounts = useMemo(() => {
    let warning = 0;
    let critical = 0;
    if (result) {
      if (result.predicted_fail_in_50 || result.predicted_fail_in_30) warning += 1;
      if (result.predicted_fail_in_10 || result.risk_state === "Critical") critical += 1;
    }
    return { warning, critical, total: warning + critical };
  }, [result]);

  const recommendation = useMemo(() => {
    const hint = result?.action_hint_for_supervisor;
    const map: Record<string, { priority: string; cls: string; text: string; body: string }> = {
      maintenance_supervisor_priority_review: {
        priority: "HIGH PRIORITY",
        cls: "text-red-400",
        text: "schedule maintenance review",
        body: "Failure-risk agent indicates a critical engine condition.",
      },
      increase_monitoring_and_prepare_inspection: {
        priority: "MEDIUM PRIORITY",
        cls: "text-amber-400",
        text: "prepare inspection",
        body: "Risk is accelerating; increase monitoring cadence.",
      },
      monitor_closely: {
        priority: "LOW PRIORITY",
        cls: "text-sky-400",
        text: "monitor closely",
        body: "Risk is rising but below the alert threshold.",
      },
      continue_monitoring: {
        priority: "NOMINAL",
        cls: "text-emerald-400",
        text: "continue monitoring",
        body: "No action required at the current risk level.",
      },
    };
    return hint ? map[String(hint)] ?? null : null;
  }, [result]);

  function sendFeedback(wasFalseAlarm: boolean) {
    feedback.mutate({ fd_subset: fdSubset, horizon: "30", was_false_alarm: wasFalseAlarm });
  }

  return (
    <div className="space-y-4">
      {/* ---------------- header ---------------- */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-sky-500">
            AERO-AI / FAILURE RISK
          </p>
          <h1 className="mt-1 text-3xl font-semibold text-slate-50">Failure Risk Agent</h1>
          <p className="mt-1 text-sm text-slate-400">
            Multi-horizon failure probability &middot; Monte-Carlo uncertainty &middot;
            Self-adaptive thresholds &middot; Risk trajectory
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-6 rounded-xl border border-slate-800/70 bg-slate-900/50 px-5 py-3">
            <div>
              <p className="text-xs text-slate-500">Model status</p>
              <Pill ok={status.modelActive} label={status.modelActive ? "Active" : "Unavailable"} />
            </div>
            <div>
              <p className="text-xs text-slate-500">API status</p>
              <Pill ok={status.apiConnected} label={status.apiConnected ? "Connected" : "Disconnected"} />
            </div>
            <div>
              <p className="text-xs text-slate-500">Scored cycles</p>
              <p className="text-sm font-semibold text-sky-400">
                {points.length ? points.length.toLocaleString() : DASH}
              </p>
            </div>
          </div>
          <NotificationCenter compact />
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-sky-600 text-xs font-semibold text-white">
            AE
          </div>
        </div>
      </div>

      {/* ---------------- pipeline run context ---------------- */}
      {/*
        This screen reports what the last full-pipeline run produced. There is
        deliberately no dataset picker, demo loader or "run prediction" button:
        the subset is a property of the uploaded file (the loader reads it from
        the filename and the batch selects the matching per-subset model), so
        offering it as a choice here let the screen score a window against a
        BiLSTM that was never fitted for it. Everything below is read back from
        the pipeline's own decision table.
      */}
      <div className="rounded-xl border border-slate-800/70 bg-slate-900/40 p-5">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <label className="block lg:col-span-2">
            <span className="text-xs text-slate-400">Engine</span>
            <select
              value={engineId}
              onChange={(e) => setSelected(e.target.value)}
              disabled={!engines.length}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-sky-500 disabled:opacity-50"
            >
              {engines.length ? (
                engines.map((e) => (
                  <option key={String(e.engine_id)} value={String(e.engine_id)}>
                    Engine {String(e.engine_id)}
                    {e.risk_state ? ` — ${titleCase(e.risk_state)}` : ""}
                  </option>
                ))
              ) : (
                <option value="">No pipeline results yet</option>
              )}
            </select>
          </label>
          <div>
            <p className="text-xs text-slate-400">Dataset</p>
            <p className="mt-1 text-sm font-medium text-slate-200">{fdSubset}</p>
            <p className="text-[11px] text-slate-500">from the uploaded file</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Current cycle</p>
            <p className="mt-1 text-2xl font-semibold text-sky-400">{currentCycle ?? DASH}</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Pipeline run</p>
            <p className="mt-1 text-sm font-medium text-sky-400">
              {fleet.dataUpdatedAt ? new Date(fleet.dataUpdatedAt).toLocaleString() : DASH}
            </p>
            <p className="text-[11px] text-slate-500">
              {fleet.summary
                ? `${fleet.summary.engines} engines · ${fleet.summary.rows.toLocaleString()} rows`
                : "awaiting results"}
            </p>
          </div>
        </div>

        {engineRows.isLoading && (
          <p className="mt-3 text-sm text-slate-400">Loading engine {engineId}…</p>
        )}
        {!engineRows.isLoading && !engines.length && (
          <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
            No pipeline results yet. Upload a C-MAPSS file and run the full pipeline to
            populate this dashboard.
          </p>
        )}
        {(fleet.error || engineRows.isError) && (
          <p className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {((fleet.error ?? engineRows.error) as Error)?.message ?? "Request failed."}
          </p>
        )}
      </div>

      {/* ---------------- row 1: status / probability / trajectory / uncertainty ---------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Panel title="Engine Status" right={<Pill ok={Boolean(result)} label={result ? "Running" : "Idle"} />}>
          <dl className="divide-y divide-slate-800/70 text-sm">
            {[
              ["Engine ID", engineId],
              ["FD subset", result?.fd_subset ?? fdSubset],
              ["Cycle", currentCycle ?? DASH],
              ["Trajectory state", result?.risk_state ?? DASH],
            ].map(([k, v], i) => (
              <div key={String(k)} className="flex justify-between py-2">
                <dt className="text-slate-400">{k}</dt>
                <dd className={`font-medium ${i === 3 ? t.text : "text-slate-100"}`}>{String(v)}</dd>
              </div>
            ))}
          </dl>
        </Panel>

        <Panel title="Failure Probability (Multi-Horizon)">
          <div className="grid grid-cols-3 gap-2">
            {HORIZONS.map((h) => {
              const risk = result?.[`risk_${h}`] as number | undefined;
              const thr = result?.[`threshold_${h}`] as number | undefined;
              const band = riskBand(risk, thr);
              return (
                <div key={h} className={`rounded-lg border-2 ${band.ring} bg-slate-950/40 p-2 text-center`}>
                  <p className="text-xl font-semibold text-slate-50">{pct(risk)}</p>
                  <p className="text-[10px] leading-tight text-slate-500">within {h} cycles</p>
                  <p className={`mt-1 text-[10px] font-semibold ${band.cls}`}>{band.label}</p>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Judged against each horizon&rsquo;s live threshold, not a fixed 50% cut.
          </p>
        </Panel>

        <Panel
          title="Risk Trajectory"
          right={
            <span className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs ${t.text}`}>
              <span className={`h-2 w-2 rounded-full ${t.dot}`} />
              {result?.risk_state ?? DASH}
            </span>
          }
        >
          <dl className="space-y-3">
            <div className="flex items-center justify-between">
              <dt className="text-sm text-slate-400">Velocity</dt>
              <dd className="text-sm font-semibold text-sky-400">
                {result ? `${(result.risk_velocity ?? 0) >= 0 ? "↑" : "↓"} ${fmt(result.risk_velocity, 3)} / cycle` : DASH}
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-sm text-slate-400">Acceleration</dt>
              <dd className="text-sm font-semibold text-sky-400">
                {result ? `${(result.risk_acceleration ?? 0) >= 0 ? "↑" : "↓"} ${fmt(result.risk_acceleration, 3)} / cycle` : DASH}
              </dd>
            </div>
            <div className="flex items-center justify-between border-t border-slate-800/70 pt-2">
              <dt className="text-sm text-slate-400">Control risk</dt>
              <dd className="text-sm font-medium text-slate-200">{pct(result?.control_risk, 1)}</dd>
            </div>
          </dl>
        </Panel>

        <Panel title="Uncertainty" right={<span className="text-xs text-slate-500">Monte Carlo dropout</span>}>
          <ConfidenceRing value={confidence} />
          <dl className="mt-3 space-y-1 border-t border-slate-800/70 pt-2 text-sm">
            {HORIZONS.map((h) => (
              <div key={h} className="flex justify-between">
                <dt className="text-slate-400">Std dev ({h}c)</dt>
                <dd className="font-medium text-slate-100">
                  {fmt(result?.[`uncertainty_${h}`] as number | undefined, 4)}
                </dd>
              </div>
            ))}
          </dl>
        </Panel>
      </div>

      {/* ---------------- row 2: threshold / memory / twin / recommendation ---------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Panel title="Self-Adaptive Threshold" subtitle="EWA-adapted (Module 5)">
          <dl className="space-y-2 text-sm">
            {HORIZONS.map((h) => {
              const row = horizons[h];
              return (
                <div key={h} className="flex items-center justify-between">
                  <dt className="text-slate-400">Threshold ({h}c)</dt>
                  <dd className="flex items-center gap-1.5">
                    <span className="text-xs text-slate-500">{pct(row?.original_threshold, 0)}</span>
                    <span className="text-slate-600">→</span>
                    <span className="font-semibold text-slate-100">{pct(row?.current_threshold, 1)}</span>
                    <span className="text-slate-600">
                      {row?.trend === "up" ? "↑" : row?.trend === "down" ? "↓" : "→"}
                    </span>
                  </dd>
                </div>
              );
            })}
          </dl>
          <p className="mt-3 text-xs text-slate-500">
            Feedback events{" "}
            <span className="font-medium text-slate-300">{horizons["30"]?.feedback_count ?? 0}</span>
            {"   "}False alarm rate{" "}
            <span className="font-medium text-slate-300">
              {horizons["30"]?.false_alarm_rate === null || horizons["30"]?.false_alarm_rate === undefined
                ? DASH
                : pct(horizons["30"]?.false_alarm_rate, 1)}
            </span>
          </p>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Feedback on the 30-cycle alert nudges its threshold via an exponentially-weighted
            average &mdash; a false alarm raises it, a confirmed event lowers it. This directly
            changes what the next prediction on {fdSubset} flags as a risk.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={() => sendFeedback(true)}
              disabled={feedback.isPending}
              className="rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-xs text-slate-200 hover:bg-slate-700/60 disabled:opacity-40"
            >
              Mark last alert as false alarm
            </button>
            <button
              onClick={() => sendFeedback(false)}
              disabled={feedback.isPending}
              className="rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
            >
              Confirm real event
            </button>
          </div>
        </Panel>

        <Panel
          title="Agent Memory"
          footer={<DetailLink label="View memory dashboard" onClick={() => memory.refetch?.()} />}
        >
          <dl className="divide-y divide-slate-800/70 text-sm">
            <div className="flex justify-between py-2">
              <dt className="text-slate-400">Stored episodes</dt>
              <dd className="font-semibold text-slate-100">{memory.status?.total_memories ?? DASH}</dd>
            </div>
            <div className="flex justify-between py-2">
              <dt className="text-slate-400">Memory confidence</dt>
              <dd className="font-semibold text-slate-100">
                {memory.status ? `${Math.round(memory.status.memory_confidence * 100)}%` : DASH}
              </dd>
            </div>
            <div className="flex justify-between py-2">
              <dt className="text-slate-400">Distinct engines</dt>
              <dd className="font-semibold text-slate-100">{memory.status?.distinct_engines ?? DASH}</dd>
            </div>
          </dl>
        </Panel>

        <Panel
          title="Digital Twin Status"
          right={<Pill ok={status.apiConnected} label={status.apiConnected ? "Connected" : "Offline"} />}
        >
          <dl className="divide-y divide-slate-800/70 text-sm">
            <div className="flex justify-between py-2">
              <dt className="text-slate-400">API</dt>
              <dd className={`font-medium ${status.apiConnected ? "text-emerald-400" : "text-slate-400"}`}>
                {status.apiConnected ? "Connected" : "Disconnected"}
              </dd>
            </div>
            <div className="flex justify-between py-2">
              <dt className="text-slate-400">Model artifacts</dt>
              <dd className={`font-medium ${artifacts.data ? "text-emerald-400" : "text-slate-400"}`}>
                {artifacts.isLoading ? "Checking…" : artifacts.data ? "Ready" : DASH}
              </dd>
            </div>
            <div className="flex justify-between py-2">
              <dt className="text-slate-400">Model in use</dt>
              <dd className="font-medium text-slate-200">{result?.model_used ?? DASH}</dd>
            </div>
          </dl>
        </Panel>

        <Panel
          title="AI Recommendation"
          right={
            <span className={`text-xs font-semibold ${recommendation?.cls ?? "text-slate-500"}`}>
              {recommendation?.priority ?? DASH}
            </span>
          }
          className="border-l-2 border-l-sky-500"
        >
          <p className="text-lg font-semibold text-slate-50">{recommendation?.text ?? DASH}</p>
          <p className="mt-2 text-xs leading-relaxed text-slate-400">
            {recommendation?.body ?? "Run a prediction to generate a recommendation."}
          </p>
          {result && (
            <p className="mt-3 border-t border-slate-800/70 pt-2 text-[11px] text-slate-500">
              Hint sent to supervisor:{" "}
              <code className="text-slate-400">{String(result.action_hint_for_supervisor ?? DASH)}</code>
            </p>
          )}
        </Panel>
      </div>

      {/* ---------------- row 3: risk over time / trajectory / health gauge ---------------- */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel
          title="Risk Prediction Over Time"
          subtitle="Failure probability at 10/30/50-cycle horizons"
          right={
            <div className="flex gap-2 text-[10px]">
              {[["P10", "#10b981"], ["P30", "#f59e0b"], ["P50", "#ef4444"]].map(([l, c]) => (
                <span key={l} className="flex items-center gap-1 text-slate-400">
                  <span className="h-2 w-2 rounded-sm" style={{ background: c }} />
                  {l}
                </span>
              ))}
            </div>
          }
        >
          <div className="h-48">
            {series.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="cycle" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} unit="%" />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }} />
                  <Line dataKey="p10" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line dataKey="p30" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line dataKey="p50" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-xs text-slate-600">
                No scored cycles for this engine yet.
              </p>
            )}
          </div>
        </Panel>

        <Panel
          title="Risk Trajectory Over Time"
          subtitle="Degradation speed vs. acceleration across the trajectory"
          right={
            <div className="flex gap-2 text-[10px]">
              {[["Speed", "#38bdf8"], ["Acceleration", "#f97316"]].map(([l, c]) => (
                <span key={l} className="flex items-center gap-1 text-slate-400">
                  <span className="h-2 w-2 rounded-sm" style={{ background: c }} />
                  {l}
                </span>
              ))}
            </div>
          }
        >
          <div className="h-48">
            {series.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="cycle" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }} />
                  <Line dataKey="speed" stroke="#38bdf8" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line dataKey="accel" stroke="#f97316" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-xs text-slate-600">
                No scored cycles for this engine yet.
              </p>
            )}
          </div>
        </Panel>

        <Panel title="Engine Health Gauge" right={<span className="text-xs text-slate-500">100 &minus; control risk</span>}>
          <HealthGauge value={healthScore} />
          <p className="mt-2 text-center text-[11px] leading-relaxed text-slate-500">
            Derived from this agent&rsquo;s own control risk, so the page stays independent of
            the RUL Lifecycle module.
          </p>
        </Panel>
      </div>

      {/* ---------------- row 4: analytics / velocity / acceleration ---------------- */}
      <Panel
        title="Risk Analytics &amp; Insights"
        right={<span className="text-xs text-slate-500">Aggregated across the analyzed trajectory</span>}
      >
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[
            ["Avg. degradation velocity", fmt(insights?.avgSpeed, 2), "text-slate-50"],
            ["Avg. degradation acceleration", fmt(insights?.avgAccel, 2), "text-slate-50"],
            ["High-risk cycles (P30 ≥ 50%)", insights ? String(insights.highRisk) : DASH, "text-slate-50"],
            ["Critical-state cycles", insights ? String(insights.critical) : DASH, "text-red-400"],
          ].map(([label, value, cls]) => (
            <div key={label}>
              <p className="text-xs text-slate-500">{label}</p>
              <p className={`mt-0.5 text-2xl font-semibold ${cls}`}>{value}</p>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel
          title="Degradation Velocity Over Time"
          subtitle="Real per-cycle rate of predicted change (proxy for risk velocity)"
        >
          <div className="h-44">
            {series.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="cycle" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }} />
                  <Line dataKey="speed" stroke="#38bdf8" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-xs text-slate-600">No trajectory yet.</p>
            )}
          </div>
        </Panel>

        <Panel
          title="Degradation Acceleration Over Time"
        >
          <div className="h-44">
            {series.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="cycle" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }} />
                  <Line dataKey="accel" stroke="#f97316" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-xs text-slate-600">No trajectory yet.</p>
            )}
          </div>
        </Panel>

        <Panel title="Risk State Distribution" right={<span className="text-xs text-slate-500">This engine&rsquo;s trajectory</span>}>
          <div className="flex items-center gap-5">
            <RiskDonut counts={riskCounts} total={points.length} />
            <ul className="flex-1 space-y-1.5">
              {RISK_STATES.map((s) => (
                <li key={s} className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 text-slate-400">
                    <span className={`h-2 w-2 rounded-full ${tone(s).dot}`} />
                    {s}
                  </span>
                  <span className="font-medium text-slate-200">{riskCounts[s] ?? 0}</span>
                </li>
              ))}
            </ul>
          </div>
        </Panel>
      </div>

      {/* ---------------- alerts summary ---------------- */}
      <Panel title="Alerts Summary">
        <dl className="divide-y divide-slate-800/70 text-sm">
          <div className="flex justify-between py-2">
            <dt className="text-slate-400">Warning</dt>
            <dd className="font-semibold text-amber-400">{alertCounts.warning}</dd>
          </div>
          <div className="flex justify-between py-2">
            <dt className="text-slate-400">Critical</dt>
            <dd className="font-semibold text-red-400">{alertCounts.critical}</dd>
          </div>
          <div className="flex justify-between py-2">
            <dt className="text-slate-400">Total active</dt>
            <dd className="font-semibold text-slate-100">{alertCounts.total}</dd>
          </div>
        </dl>
        <p className="mt-2 text-[11px] text-slate-500">
          Derived from the live horizon flags: a 30/50-cycle exceedance raises a warning, a
          10-cycle exceedance or a Critical risk state raises a critical alert.
        </p>
      </Panel>
    </div>
  );
}
