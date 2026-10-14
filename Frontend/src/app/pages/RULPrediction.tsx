/**
 * Autonomous RUL Reasoning & Lifecycle Planning Agent -- dashboard.
 *
 * Single-engine view over the Predictive Maintenance agent. Every figure is
 * read from a backend response; nothing is hardcoded. Where the backend cannot
 * supply a value the cell renders an em dash rather than inventing a number.
 *
 * Driven entirely by the last full-pipeline run: the fused decision table is
 * read back through `/integration/dashboard`, filtered to one engine. There is
 * no dataset picker, demo loader or "run prediction" control, and that is
 * deliberate -- the subset is a property of the uploaded file (the loader takes
 * it from the filename so the batch can select the matching per-subset model),
 * so offering it as a choice here let the screen score a window against a model
 * that was never fitted for it. The charts and the distribution donut come from
 * the engine's own scored cycles rather than a separate replay endpoint.
 *
 * Rows the batch sub-sampled past carry a forward-filled `predicted_rul`; they
 * are dropped from the series rather than drawn, because a flat placeholder
 * segment reads as a real prediction on a chart.
 *
 * Component boundary: this page belongs to the RUL Lifecycle module and shows
 * only RUL-owned fields. `risk_10/30/50` and `risk_state` belong to the
 * Failure Risk module and are presented on its own page -- so the trajectory
 * is requested with `include_risk: false`, which also stops the RUL module
 * invoking the Failure Risk model at all. The distribution donut therefore
 * counts `lifecycle_state`, the RUL agent's own scale.
 *
 * The previous fleet-wide view of this route is preserved unrouted at
 * `_RULPredictionFleet.tsx.bak`.
 */
import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  useAgentStatus,
  useEngineRows,
  useEvaluationMetrics,
  useFleetSummary,
  useMemoryStatus,
  useSimilarMemories,
} from "../../lib/hooks";
import { useEngineParam } from "../../lib/engine-param";
import { NotificationCenter } from "../components/NotificationCenter";
import type { DecisionRow } from "../../lib/types";

/** Lifecycle bands, in severity order -- the timeline strip. */
const LIFECYCLE_STAGES = [
  "Nominal",
  "Stable",
  "Degrading",
  "Accelerating",
  "Critical",
  "Unstable",
] as const;

/** Lifecycle bands used by the distribution donut -- RUL-owned, not risk. */
const DONUT_STATES = ["Nominal", "Stable", "Degrading", "Accelerating", "Critical", "Unstable"] as const;

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

const fmt = (v: number | null | undefined, digits = 1) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : DASH;

function stateTone(state: string | undefined) {
  switch ((state ?? "").toLowerCase()) {
    case "critical":
      return { text: "text-red-400", dot: "bg-red-500", ring: "stroke-red-500", hex: "#ef4444" };
    case "unstable":
      return { text: "text-fuchsia-400", dot: "bg-fuchsia-500", ring: "stroke-fuchsia-500", hex: "#d946ef" };
    case "accelerating":
      return { text: "text-orange-400", dot: "bg-orange-500", ring: "stroke-orange-500", hex: "#f97316" };
    case "rising":
      return { text: "text-amber-400", dot: "bg-amber-500", ring: "stroke-amber-500", hex: "#f59e0b" };
    case "degrading":
      return { text: "text-amber-400", dot: "bg-amber-500", ring: "stroke-amber-500", hex: "#f59e0b" };
    default:
      return { text: "text-emerald-400", dot: "bg-emerald-500", ring: "stroke-emerald-500", hex: "#10b981" };
  }
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
          <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-200">
            {title}
          </h3>
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

function FlagRow({ label, value }: { label: string; value: boolean | undefined }) {
  const on = value === true;
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-slate-400">{label}</span>
      <span
        className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium ${
          on ? "bg-red-500/15 text-red-400" : "bg-slate-700/40 text-slate-300"
        }`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${on ? "bg-red-500" : "bg-emerald-500"}`} />
        {value === undefined ? DASH : on ? "Yes" : "No"}
      </span>
    </div>
  );
}

function RulGauge({ value, state }: { value: number | null; state?: string }) {
  const max = 150;
  const pct = value === null ? 0 : Math.max(0, Math.min(1, value / max));
  const r = 52;
  const circ = 2 * Math.PI * r;
  const tone = stateTone(state);
  return (
    <div className="relative mx-auto h-36 w-36">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={r} className="fill-none stroke-slate-800" strokeWidth="10" />
        <circle
          cx="64"
          cy="64"
          r={r}
          className={`fill-none ${tone.ring}`}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${circ * pct} ${circ}`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-semibold text-slate-50">{fmt(value)}</span>
        <span className="text-xs text-slate-500">cycles</span>
      </div>
    </div>
  );
}

/** Segmented donut over the risk-state counts. */
function RiskDonut({ counts, total }: { counts: Record<string, number>; total: number }) {
  const r = 52;
  const circ = 2 * Math.PI * r;
  let offset = 0;
  const segments = DONUT_STATES.map((s) => {
    const n = counts[s] ?? 0;
    const frac = total > 0 ? n / total : 0;
    const seg = { state: s, len: circ * frac, off: offset, hex: stateTone(s).hex };
    offset += circ * frac;
    return seg;
  }).filter((s) => s.len > 0);

  return (
    <div className="relative h-28 w-28 shrink-0">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={r} className="fill-none stroke-slate-800" strokeWidth="12" />
        {segments.map((s) => (
          <circle
            key={s.state}
            cx="64"
            cy="64"
            r={r}
            fill="none"
            stroke={s.hex}
            strokeWidth="12"
            strokeDasharray={`${s.len} ${circ - s.len}`}
            strokeDashoffset={-s.off}
          />
        ))}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-semibold text-slate-50">{total || 0}</span>
        <span className="text-xs text-slate-500">cycles</span>
      </div>
    </div>
  );
}

/** One node in the self-correction flow. */
function FlowNode({
  label,
  value,
  done,
}: {
  label: string;
  value: string;
  done: boolean;
}) {
  return (
    <div className="flex flex-col items-center text-center">
      <span
        className={`flex h-7 w-7 items-center justify-center rounded-full border-2 text-xs ${
          done
            ? "border-sky-500 text-sky-400"
            : "border-slate-700 text-slate-600"
        }`}
      >
        ✓
      </span>
      <span className="mt-1.5 text-[11px] leading-tight text-slate-300">{label}</span>
      <span className="text-[10px] text-slate-500">{value}</span>
    </div>
  );
}

export function RULPrediction() {
  const [selected, setSelected] = useEngineParam();

  const status = useAgentStatus();
  const fleet = useFleetSummary();
  const memory = useMemoryStatus();
  const evalMetrics = useEvaluationMetrics();

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
   * RUL-owned field this page reads, so it stands in for the single-shot
   * prediction response the screen used to be driven by.
   */
  const latestRow = rows.length ? rows[rows.length - 1] : null;
  const result: DecisionRow | null = useMemo(
    () =>
      latestRow
        ? { ...latestRow, lifecycle_state: titleCase(latestRow.lifecycle_state) }
        : null,
    [latestRow],
  );

  /**
   * Per-cycle series, straight from the pipeline's own scored rows.
   *
   * Rows the batch did not score (it can sub-sample long trajectories) carry a
   * forward-filled `predicted_rul`, so they are dropped rather than drawn --
   * a flat placeholder segment reads as a real prediction on a chart.
   */
  const points = useMemo(
    () =>
      rows
        .filter((r) => typeof r.predicted_rul === "number")
        .map((r) => ({
          cycle: r.cycle,
          predicted_rul: r.predicted_rul as number,
          lower_bound: r.lower_bound ?? 0,
          upper_bound: r.upper_bound ?? 0,
          uncertainty_width: r.uncertainty_width ?? 0,
          lifecycle_state: titleCase(r.lifecycle_state),
        })),
    [rows],
  );

  const similar = useSimilarMemories(
    result ? result.fd_subset : null,
    result?.degradation_speed ?? null,
    result?.degradation_acceleration ?? null,
  );

  const currentCycle = result ? Number(result.cycle) : null;
  const tone = stateTone(result?.lifecycle_state ?? undefined);
  const halfWidth =
    typeof result?.uncertainty_width === "number" ? result.uncertainty_width / 2 : null;

  /** Chart series, keyed on the engine's real cycles. */
  const series = useMemo(
    () =>
      points.map((p) => ({
        cycle: p.cycle,
        rul: Number(p.predicted_rul.toFixed(2)),
        lower: Number(p.lower_bound.toFixed(2)),
        band: Number((p.upper_bound - p.lower_bound).toFixed(2)),
        width: Number(p.uncertainty_width.toFixed(2)),
      })),
    [points],
  );

  const riskCounts = useMemo(() => {
    const out: Record<string, number> = {};
    for (const p of points) {
      const k = p.lifecycle_state;
      if (k) out[k] = (out[k] ?? 0) + 1;
    }
    return out;
  }, [points]);

  return (
    <div className="space-y-4">
      {/* ---------------- header ---------------- */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-sky-500">
            AERO-AI / DASHBOARD
          </p>
          <h1 className="mt-1 text-3xl font-semibold text-slate-50">
            Autonomous RUL Reasoning &amp; Lifecycle Planning Agent
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            AI-powered remaining useful life intelligence &middot; Conformal uncertainty
            &middot; Lifecycle reasoning &middot; Self-correction
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
              <Pill
                ok={status.apiConnected}
                label={status.apiConnected ? "Connected" : "Disconnected"}
              />
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
        model that was never fitted for it. Everything below is read back from
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
                    {e.lifecycle_state ? ` — ${titleCase(e.lifecycle_state)}` : ""}
                  </option>
                ))
              ) : (
                <option value="">No pipeline results yet</option>
              )}
            </select>
          </label>
          <div>
            <p className="text-xs text-slate-400">Dataset</p>
            <p className="mt-1 text-sm font-medium text-slate-200">
              {result?.fd_subset ?? fleet.summary?.subsets.join(", ") ?? DASH}
            </p>
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

      {/* ---------------- RUL lifecycle detail ---------------- */}
      <p className="pt-1 text-center text-xs font-semibold uppercase tracking-widest text-slate-500">
        RUL Lifecycle Detail
      </p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <Panel title="Predicted RUL">
          <RulGauge
            value={typeof result?.predicted_rul === "number" ? result.predicted_rul : null}
            state={result?.lifecycle_state ?? undefined}
          />
          <p className="mt-2 text-center text-xs text-slate-500">Cycles remaining</p>
        </Panel>

        <Panel
          title="Prediction Interval"
          subtitle={halfWidth === null ? undefined : `±${fmt(halfWidth)} cycles`}
        >
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <p className="text-[11px] text-slate-500">Lower bound</p>
              <p className="text-lg font-semibold text-sky-400">{fmt(result?.lower_bound)}</p>
            </div>
            <div>
              <p className="text-[11px] text-slate-500">Predicted RUL</p>
              <p className="text-lg font-semibold text-slate-100">{fmt(result?.predicted_rul)}</p>
            </div>
            <div>
              <p className="text-[11px] text-slate-500">Upper bound</p>
              <p className="text-lg font-semibold text-sky-400">{fmt(result?.upper_bound)}</p>
            </div>
          </div>
          {/* Where the point estimate sits inside its own band. The conformal
              half-width is symmetric, so the marker is centred unless the lower
              bound was clamped at zero -- which is exactly the case worth
              seeing, because a clamped band cannot exclude immediate failure. */}
          <div className="relative mt-4 h-1.5 rounded-full bg-slate-800">
            {typeof result?.predicted_rul === "number" &&
              typeof result?.lower_bound === "number" &&
              typeof result?.upper_bound === "number" &&
              result.upper_bound > result.lower_bound && (
                <span
                  className="absolute -top-1 h-3.5 w-3.5 -translate-x-1/2 rounded-full border-2 border-slate-950 bg-sky-400"
                  style={{
                    left: `${Math.max(
                      0,
                      Math.min(
                        100,
                        ((result.predicted_rul - result.lower_bound) /
                          (result.upper_bound - result.lower_bound)) *
                          100,
                      ),
                    )}%`,
                  }}
                />
              )}
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            Interval width{" "}
            <span className="font-semibold text-slate-300">{fmt(result?.uncertainty_width)}</span>
            {evalMetrics.available && (
              <>
                {" "}PICP{" "}
                <span className="font-semibold text-slate-300">{fmt(evalMetrics.picp, 3)}</span>{" "}
                MPIW <span className="font-semibold text-slate-300">{fmt(evalMetrics.mpiw, 2)}</span>
              </>
            )}
          </p>
        </Panel>

        <Panel title="Lifecycle State">
          <span
            className={`inline-flex items-center gap-1.5 rounded-md bg-slate-800/60 px-2.5 py-1 text-sm font-medium ${tone.text}`}
          >
            <span className={`h-2 w-2 rounded-full ${tone.dot}`} />
            {result?.lifecycle_state ?? DASH}
          </span>
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            {result
              ? `Predicted RUL sits in the ${String(result.lifecycle_state ?? DASH).toLowerCase()} band.`
              : "Awaiting pipeline results."}
          </p>
          <a
            href="/ai-reasoning"
            className="mt-3 inline-flex rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
          >
            View Reasoning &rarr;
          </a>
        </Panel>

        <Panel title="Model in Use">
          <p className="truncate text-lg font-semibold text-sky-400" title={String(result?.model_used ?? "")}>
            {result?.model_used ? String(result.model_used) : DASH}
          </p>
          <div className="mt-3">
            <p className="text-[11px] text-slate-500">Fallback model</p>
            <p className="text-sm text-slate-200">
              {result?.fallback_model_used ? String(result.fallback_model_used) : "None"}
            </p>
          </div>
          <div className="mt-2">
            <p className="text-[11px] text-slate-500">Status</p>
            <Pill
              ok={Boolean(result?.fallback_allowed)}
              label={result?.fallback_allowed ? "Active" : "Not required"}
            />
          </div>
        </Panel>

        <Panel title="Health Reliability">
          {(() => {
            const flags = [
              result?.confidence_collapse_flag,
              result?.sudden_health_drop_flag,
              result?.instability_flag,
            ];
            const anyOn = flags.some((f) => f === true);
            return (
              <span
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium ${
                  anyOn ? "bg-red-500/15 text-red-400" : "bg-emerald-500/15 text-emerald-400"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${anyOn ? "bg-red-500" : "bg-emerald-500"}`}
                />
                {result ? (anyOn ? "Unreliable" : "Reliable") : DASH}
              </span>
            );
          })()}
          <div className="mt-2 divide-y divide-slate-800/70">
            <FlagRow
              label="Confidence collapse"
              value={result ? Boolean(result.confidence_collapse_flag) : undefined}
            />
            <FlagRow
              label="Sudden health drop"
              value={result ? Boolean(result.sudden_health_drop_flag) : undefined}
            />
            <FlagRow
              label="Instability"
              value={result ? Boolean(result.instability_flag) : undefined}
            />
          </div>
        </Panel>

        <Panel title="Degradation Overview">
          <div>
            <p className="text-[11px] text-slate-500">Speed</p>
            <p className="text-2xl font-semibold text-slate-100">
              {fmt(result?.degradation_speed, 2)}
            </p>
            <p className="text-[11px] text-slate-500">RUL lost per cycle</p>
          </div>
          <div className="mt-3">
            <p className="text-[11px] text-slate-500">Acceleration</p>
            <p className="text-2xl font-semibold text-amber-400">
              {fmt(result?.degradation_acceleration, 2)}
            </p>
            <p className="text-[11px] text-slate-500">Change in speed per cycle</p>
          </div>
        </Panel>
      </div>

      {/* ---------------- charts ---------------- */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel
          title="Remaining Useful Life Prediction"
          subtitle="Predicted RUL with the conformal-prediction confidence band"
        >
          <div className="h-48">
            {series.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={series}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="cycle" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Area dataKey="lower" stackId="b" stroke="none" fill="transparent" isAnimationActive={false} />
                  <Area
                    dataKey="band"
                    stackId="b"
                    stroke="none"
                    fill="#0ea5e9"
                    fillOpacity={0.18}
                    isAnimationActive={false}
                  />
                  <Line dataKey="rul" stroke="#38bdf8" strokeWidth={2} dot={false} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-xs text-slate-600">
                No scored cycles for this engine yet.
              </p>
            )}
          </div>
        </Panel>

        <Panel
          title="Uncertainty Evolution"
          subtitle="Conformal interval width over the engine's lifetime"
        >
          <div className="h-48">
            {series.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="cycle" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Line dataKey="width" stroke="#a78bfa" strokeWidth={2} dot={false} isAnimationActive={false} />
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
          title="Lifecycle State Distribution"
          right={<span className="text-xs text-slate-500">This engine&rsquo;s trajectory</span>}
        >
          <div className="flex items-center gap-5">
            <RiskDonut counts={riskCounts} total={points.length} />
            <ul className="flex-1 space-y-1.5">
              {DONUT_STATES.map((s) => (
                <li key={s} className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 text-slate-400">
                    <span className={`h-2 w-2 rounded-full ${stateTone(s).dot}`} />
                    {s}
                  </span>
                  <span className="font-medium text-slate-200">{riskCounts[s] ?? 0}</span>
                </li>
              ))}
            </ul>
          </div>
        </Panel>
      </div>

      {/* ---------------- bottom row ---------------- */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        <Panel title="Lifecycle Timeline">
          <p className="text-xs text-slate-500">Cycle {currentCycle ?? DASH}</p>
          <div className="mt-3 flex gap-1">
            {LIFECYCLE_STAGES.map((s) => {
              const active = result?.lifecycle_state === s;
              return (
                <div key={s} className="flex-1">
                  <div className={`h-1.5 rounded-full ${active ? stateTone(s).dot : "bg-slate-800"}`} />
                  <p
                    className={`mt-1 truncate text-[10px] ${
                      active ? stateTone(s).text : "text-slate-600"
                    }`}
                  >
                    {s}
                  </p>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-sm text-slate-400">
            Current state:{" "}
            <span className={`font-semibold ${tone.text}`}>{result?.lifecycle_state ?? DASH}</span>
          </p>
        </Panel>

        <Panel
          title="AI Reasoning Summary"
          subtitle="Generated from lifecycle state, flags, model used & degradation"
          footer={
            <button
              onClick={() =>
                document
                  .getElementById("self-correction")
                  ?.scrollIntoView({ behavior: "smooth", block: "center" })
              }
              className="rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
            >
              View Full Reasoning &rarr;
            </button>
          }
        >
          <p id="ai-reasoning" className="text-sm leading-relaxed text-slate-300">
            {result ? (
              <>
                Engine has entered a {(result.lifecycle_state ?? DASH).toLowerCase()} remaining-life state.
                Prediction uncertainty is{" "}
                {result.confidence_collapse_flag
                  ? "wider than the estimate it brackets"
                  : "within the acceptable band"}
                .{" "}
                {result.correction_applied
                  ? `Primary TCN read of ${fmt(result.primary_predicted_rul as number | null)} was corrected by the ${String(result.fallback_model_used)} fallback.`
                  : "No fallback correction was required."}
              </>
            ) : (
              <span className="text-slate-600">Run a prediction to generate reasoning.</span>
            )}
          </p>
        </Panel>

        <Panel
          title="Self-Correction Status"
          right={
            <span
              className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs ${
                result?.instability_flag
                  ? "bg-red-500/15 text-red-400"
                  : "bg-emerald-500/15 text-emerald-400"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  result?.instability_flag ? "bg-red-500" : "bg-emerald-500"
                }`}
              />
              {result ? (result.instability_flag ? "Active" : "Inactive") : DASH}
            </span>
          }
        >
          <div id="self-correction" className="grid grid-cols-2 gap-x-2 gap-y-3">
            <FlowNode
              label="TCN Prediction"
              value={result ? `${fmt(result.primary_predicted_rul as number | null)}c` : DASH}
              done={Boolean(result)}
            />
            <FlowNode
              label="Instability Detection"
              value={result ? (result.instability_flag ? "Detected" : "Clear") : DASH}
              done={Boolean(result)}
            />
            <FlowNode
              label="Fallback Validation"
              value={result ? String(result.fallback_model_used ?? "N/A") : DASH}
              done={Boolean(result)}
            />
            <FlowNode
              label="Decision"
              value={
                result ? (result.correction_applied ? "Used fallback" : "Kept TCN") : DASH
              }
              done={Boolean(result)}
            />
            <FlowNode
              label="Final Prediction"
              value={result ? `${fmt(result.predicted_rul)}c` : DASH}
              done={Boolean(result)}
            />
          </div>
          <div className="mt-3 space-y-1 border-t border-slate-800/70 pt-3 text-xs text-slate-400">
            <p>
              Fallback allowed{" "}
              <span className="font-medium text-slate-200">
                {result ? (result.fallback_allowed ? "Yes" : "No") : DASH}
              </span>
            </p>
            <p>
              Correction applied{" "}
              <span className="font-medium text-slate-200">
                {result ? (result.correction_applied ? "Yes" : "No") : DASH}
              </span>
            </p>
            <p>
              Active model{" "}
              <span className="font-medium text-slate-200">{result?.model_used ?? DASH}</span>
            </p>
            <p>
              Correction trigger Instability ={" "}
              <span className="font-medium text-slate-200">
                {result ? String(result.instability_flag) : DASH}
              </span>
            </p>
          </div>
        </Panel>

        <Panel
          title="Agent Memory Status"
          right={<span className="text-xs text-slate-500">Module 3</span>}
          footer={
            <>
              <button
                onClick={() => similar.refetch()}
                className="w-full rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700/60"
              >
                View Memory Explorer &rarr;
              </button>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                Sequential degradation history, instability event logging and trajectory
                history are stored today. Similar-case pattern matching above ranks past
                cases by Euclidean distance; adaptive threshold learning is not implemented
                yet.
              </p>
            </>
          }
        >
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Memory confidence</span>
            <span className="font-semibold text-slate-100">
              {memory.status ? `${Math.round(memory.status.memory_confidence * 100)}%` : DASH}
            </span>
          </div>
          <div className="mt-1.5 h-1.5 rounded-full bg-slate-800">
            <div
              className="h-1.5 rounded-full bg-sky-500"
              style={{
                width: memory.status
                  ? `${Math.min(100, memory.status.memory_confidence * 100)}%`
                  : "0%",
              }}
            />
          </div>
          <dl className="mt-3 divide-y divide-slate-800/70 text-sm">
            <div className="flex justify-between py-1.5">
              <dt className="text-slate-400">Total memories</dt>
              <dd className="font-medium text-slate-100">
                {memory.status?.total_memories ?? DASH}
              </dd>
            </div>
            <div className="flex justify-between py-1.5">
              <dt className="text-slate-400">Distinct engines</dt>
              <dd className="font-medium text-slate-100">
                {memory.status?.distinct_engines ?? DASH}
              </dd>
            </div>
            <div className="flex justify-between py-1.5">
              <dt className="text-slate-400">Similar cases found</dt>
              <dd className="font-medium text-slate-100">
                {result ? similar.matches.length : DASH}
              </dd>
            </div>
            <div className="flex justify-between py-1.5">
              <dt className="text-slate-400">Last memory update</dt>
              <dd className="font-medium text-slate-100">
                {memory.status?.last_updated ?? DASH}
              </dd>
            </div>
          </dl>
        </Panel>
      </div>
    </div>
  );
}
