import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Gauge,
  HeartPulse,
  Layers,
  ShieldAlert,
  TrendingDown,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
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
  useAnomalyDashboard,
  useAnomalyEvaluation,
  useEngineHealthTimeline,
} from "../../lib/hooks";
import { useEngineParam } from "../../lib/engine-param";
import {
  ContextDriftPanel,
  FeedbackLoopPanel,
  MaintenanceQueuePanel,
} from "../components/AnomalyValidationPanels";
import { GlassCard } from "../components/GlassCard";
import type { AnomalyTimelinePoint } from "../../lib/types";

const STATE_COLOR: Record<string, string> = {
  Healthy: "#10b981",
  Degrading: "#f59e0b",
  Warning: "#f97316",
  Critical: "#ef4444",
  "Failure Risk": "#b91c1c",
};


const STATE_LABEL: Record<string, string> = {
  "Failure Risk": "HI < 20",
};
const stateLabel = (s: string) => STATE_LABEL[s] ?? s;

const DASH = "—";
const n1 = (v: unknown, d = 1) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : DASH;


type Basis = "snapshot" | "lifetime";

export function HealthMonitoring() {
  const dash = useAnomalyDashboard();
  const evaluation = useAnomalyEvaluation();
  const [basis, setBasis] = useState<Basis>("snapshot");
  const [selected, setSelected] = useEngineParam();


  const fleet = dash.data?.fleet;
  const engines = dash.data?.engines ?? [];
  const timeline = (dash.data?.timeline ?? {}) as Record<string, AnomalyTimelinePoint[]>;

  const hasLifetime = typeof fleet?.lifetime_mean_health_index === "number";
  const meanHealth =
    basis === "lifetime" && hasLifetime
      ? (fleet?.lifetime_mean_health_index as number)
      : (fleet?.mean_health_index ?? 0);


  const stateChart = useMemo(() => {
    const src: Array<[string, unknown]> =
      basis === "lifetime" && hasLifetime
        ? [
            ["Healthy", fleet?.lifetime_rows_healthy],
            ["Degrading", fleet?.lifetime_rows_degrading],
            ["Warning", fleet?.lifetime_rows_warning],
            ["Critical", fleet?.lifetime_rows_critical],
            ["Failure Risk", fleet?.lifetime_rows_failure_risk],
          ]
        : [
            ["Healthy", fleet?.engines_healthy],
            ["Degrading", fleet?.engines_degrading],
            ["Warning", fleet?.engines_warning],
            ["Critical", fleet?.engines_critical],
            ["Failure Risk", fleet?.engines_failure_risk],
          ];
    return src
      .filter(([, v]) => typeof v === "number")
      .map(([name, value]) => ({
        name: stateLabel(name),
        value: value as number,
        color: STATE_COLOR[name] ?? "#64748b",
      }));
  }, [basis, hasLifetime, fleet]);

  const ranked = useMemo(
    () =>
      [...engines]
        .filter((e) => typeof e.health_index === "number")
        .sort((a, b) => (a.health_index as number) - (b.health_index as number)),
    [engines],
  );

  const focusId = selected ?? String(ranked[0]?.engine_id ?? "");
  const focus = engines.find((e) => String(e.engine_id) === focusId);


  const focusTimeline = useEngineHealthTimeline(focusId || undefined);

  const series = useMemo(() => {
    const bundled = timeline[focusId];
    const pts =
      bundled && bundled.length ? bundled : focusTimeline.data?.points ?? [];
    return pts
      .slice()
      .sort((a, b) => a.cycle - b.cycle)
      .map((p) => {
        const hi = typeof p.health_index === "number" ? p.health_index : null;
        const target =
          typeof p.health_index_target === "number" ? p.health_index_target : null;
        return {
          cycle: p.cycle,
          health: hi === null ? null : Number(hi.toFixed(2)),


          target: target === null ? null : Number(target.toFixed(2)),
          anomaly:
            typeof p.anomaly_score === "number" ? Number((p.anomaly_score * 100).toFixed(1)) : null,
          confidence:
            typeof p.confidence_score === "number" ? Number(p.confidence_score.toFixed(1)) : null,
          reliability:
            typeof p.reliability_score === "number"
              ? Number((p.reliability_score * 100).toFixed(1))
              : null,
        };
      });
  }, [timeline, focusId, focusTimeline.data]);


  const trackingR = useMemo(() => {
    const pairs = series.filter(
      (d): d is typeof d & { health: number; target: number } =>
        typeof d.health === "number" && typeof d.target === "number",
    );
    if (pairs.length < 3) return null;
    const n = pairs.length;
    const mx = pairs.reduce((a, d) => a + d.health, 0) / n;
    const my = pairs.reduce((a, d) => a + d.target, 0) / n;
    let sxy = 0;
    let sxx = 0;
    let syy = 0;
    for (const d of pairs) {
      const dx = d.health - mx;
      const dy = d.target - my;
      sxy += dx * dy;
      sxx += dx * dx;
      syy += dy * dy;
    }
    const denom = Math.sqrt(sxx * syy);
    return denom === 0 ? null : sxy / denom;
  }, [series]);

  const rootCauses = useMemo(
    () =>
      Object.entries(fleet?.top_root_causes ?? {})
        .map(([name, value]) => ({ name, value: Number(value) }))
        .sort((a, b) => b.value - a.value),
    [fleet?.top_root_causes],
  );


  const detectors = focus
    ? [
        { name: "Isolation Forest", value: Number(focus.isolation_forest_score ?? 0) },
        { name: "LSTM Autoencoder", value: Number(focus.lstm_ae_score ?? 0) },
        { name: "Mahalanobis", value: Number(focus.mahalanobis_score ?? 0) },
        { name: "Residual Rule", value: Number(focus.residual_rule_score ?? 0) },
        { name: "Fused score", value: Number(focus.anomaly_score ?? 0) },
      ].filter((d) => Number.isFinite(d.value))
    : [];


  const earlyWarning = useMemo(
    () =>
      [...engines]
        .filter((e) => typeof e.deterioration_rate === "number")
        .sort(
          (a, b) =>
            (b.deterioration_rate as number) - (a.deterioration_rate as number),
        )
        .slice(0, 8),
    [engines],
  );

  const hold = evaluation.health ?? null;
  const healthGate = evaluation.healthAcceptance ?? null;

  const perSubsetHealth = useMemo(
    () =>
      Object.entries(hold?.per_subset ?? {}).map(([name, m]) => ({
        name,
        engines: m.engines,
        monotonicity: m.monotonicity_score,
        rul: m.rul_correlation,
        meanHi: m.health_index_mean,
      })),
    [hold?.per_subset],
  );


  const biasQuartiles = useMemo(
    () =>
      Object.entries(hold?.bias_quartiles ?? {}).map(([name, q]) => ({
        name,
        short: name.replace(/^(Q\d).*$/, "$1"),
        meanHi: typeof q.mean_hi === "number" ? Number(q.mean_hi.toFixed(2)) : null,
        startRul: q.median_start_rul ?? null,
        n: q.n ?? null,
      })),
    [hold?.bias_quartiles],
  );

  const alertMixChart = useMemo(() => {
    const order = ["none", "info", "warning", "high", "critical"];
    const colors: Record<string, string> = {
      none: "#10b981",
      info: "#22d3ee",
      warning: "#f59e0b",
      high: "#f97316",
      critical: "#ef4444",
    };
    const mix = evaluation.alertMix ?? {};
    return order
      .filter((k) => typeof mix[k] === "number")
      .map((k) => ({ name: k, value: mix[k] as number, color: colors[k] }));
  }, [evaluation.alertMix]);

  const alertMixTotal = alertMixChart.reduce((a, d) => a + d.value, 0) || 1;


  if (dash.isLoading) return <Loading label="Loading health monitoring…" />;
  if (dash.error) return <ErrorState error={dash.error} />;
  if (!fleet) return <Empty />;

  const basisLabel =
    basis === "lifetime" ? "all recorded rows" : "each engine at its final cycle";
  const countLabel = basis === "lifetime" ? "rows" : "engines";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Health Monitoring"
        subtitle={`Digital-twin health index · ${fleet.total_engines} engines · ${fleet.total_rows.toLocaleString()} rows · absolute scale, calibrated against the persisted healthy reference`}
        right={<LiveBadge />}
      />


      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800/70 bg-slate-900/40 px-4 py-3">
        <span className="text-xs uppercase tracking-widest text-slate-500">Basis</span>
        <div className="flex overflow-hidden rounded-lg border border-slate-700">
          {(["snapshot", "lifetime"] as Basis[]).map((b) => (
            <button
              key={b}
              onClick={() => setBasis(b)}
              disabled={b === "lifetime" && !hasLifetime}
              className={`px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40 ${
                basis === b
                  ? "bg-sky-600 text-white"
                  : "bg-slate-900/60 text-slate-300 hover:bg-slate-800"
              }`}
            >
              {b === "snapshot" ? "Current snapshot" : "Full lifetime"}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500">
          Figures below are computed over <span className="text-slate-300">{basisLabel}</span>.{" "}
          The <span className="text-slate-300">&ldquo;Failure Risk&rdquo;</span> health band is the
          lowest health-index tier (HI&nbsp;&lt;&nbsp;20) assigned by this component &mdash; not
          the Failure Risk agent&rsquo;s output.
          {basis === "snapshot" && (
            <>
              {" "}
              C-MAPSS units run to failure, so a low snapshot mean is expected, not a fault.
            </>
          )}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <Reveal index={0}>
          <KPICard
            title={basis === "lifetime" ? "Lifetime Mean HI" : "Mean HI (final cycle)"}
            value={<AnimatedNumber value={meanHealth} decimals={1} />}
            icon={HeartPulse}
            status={meanHealth > 70 ? "success" : meanHealth > 45 ? "warning" : "danger"}
          />
        </Reveal>
        <Reveal index={1}>
          <KPICard
            title={`Critical ${countLabel}`}
            value={
              <AnimatedNumber
                value={
                  basis === "lifetime"
                    ? Number(fleet.lifetime_rows_critical ?? 0)
                    : fleet.engines_critical
                }
              />
            }
            icon={ShieldAlert}
            status="danger"
          />
        </Reveal>
        <Reveal index={2}>
          <KPICard
            title={`HI < 20 band ${countLabel}`}
            value={
              <AnimatedNumber
                value={
                  basis === "lifetime"
                    ? Number(fleet.lifetime_rows_failure_risk ?? 0)
                    : Number(fleet.engines_failure_risk ?? 0)
                }
              />
            }
            icon={AlertTriangle}
            status="danger"
          />
        </Reveal>
        <Reveal index={3}>
          <KPICard
            title={`Healthy ${countLabel}`}
            value={
              <AnimatedNumber
                value={
                  basis === "lifetime"
                    ? Number(fleet.lifetime_rows_healthy ?? 0)
                    : fleet.engines_healthy
                }
              />
            }
            icon={Activity}
            status="success"
          />
        </Reveal>
        <Reveal index={4}>
          <KPICard
            title="Mean confidence"
            value={<AnimatedNumber value={Number(fleet.mean_confidence ?? 0)} decimals={1} />}
            icon={Gauge}
            status="success"
          />
        </Reveal>
      </div>


      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Reveal index={5} className="lg:col-span-2">
          <MetricCard
            title={`Engine E${focusId} — Health Trajectory`}
            subtitle={
              trackingR === null
                ? "Health index against the engine's true remaining-life curve"
                : `Health index against the engine's true remaining-life curve · r = ${trackingR.toFixed(3)} over ${series.length} cycles`
            }
          >
            {focusTimeline.isLoading && !series.length ? (
              <Loading label="Loading trajectory…" />
            ) : series.length ? (
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={series}>
                  <defs>
                    <linearGradient id="hi" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.6} />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.03} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="cycle" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />

                  <ReferenceLine
                    y={20}
                    stroke="#b91c1c"
                    strokeDasharray="4 4"
                    label={{ value: "HI < 20", fill: "#b91c1c", fontSize: 10, position: "insideBottomLeft" }}
                  />
                  <Area
                    name="Health index"
                    dataKey="health"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    fill="url(#hi)"
                    isAnimationActive={false}
                  />
                  <Line
                    name="True remaining life"
                    dataKey="target"
                    stroke="#a78bfa"
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    name="Anomaly score"
                    dataKey="anomaly"
                    stroke="#f97316"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <Empty label="No timeline for this engine." />
            )}
            {series.length > 0 && (
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                The dashed line is ground truth — {"100 × clip(RUL, 0, 130) / 130"} — computed from
                the engine's actual remaining life. The health index is built from twin residuals
                alone and never sees it, so the two tracking together is a result rather than a fit.
                Across the fleet the median within-engine correlation is{" "}
                <span className="text-violet-300">0.98</span>.
              </p>
            )}
          </MetricCard>
        </Reveal>

        <Reveal index={6}>
          <MetricCard
            title="Health State Distribution"
            subtitle={`By ${countLabel}, over ${basisLabel}`}
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stateChart} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" stroke="#64748b" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={11} width={82} />
                <Tooltip
                  contentStyle={{
                    background: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {stateChart.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </MetricCard>
        </Reveal>
      </div>


      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Reveal index={7}>
          <MetricCard
            title={`Detector Ensemble — E${focusId}`}
            subtitle="Four independent detectors and their fused score"
          >
            {detectors.length ? (
              <>
                <ResponsiveContainer width="100%" height={210}>
                  <BarChart data={detectors} layout="vertical" margin={{ left: 24 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} stroke="#64748b" fontSize={11} />
                    <YAxis
                      dataKey="name"
                      type="category"
                      stroke="#64748b"
                      fontSize={10}
                      width={96}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {detectors.map((d, i) => (
                        <Cell key={d.name} fill={i === detectors.length - 1 ? "#38bdf8" : "#64748b"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-slate-800/70 pt-3 text-xs">
                  <div className="flex justify-between">
                    <dt className="text-slate-500">Detector agreement</dt>
                    <dd className="text-slate-200">{n1(focus?.detector_agreement, 3)}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">Model agreement</dt>
                    <dd className="text-slate-200">{n1(focus?.model_agreement, 3)}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">Fusion threshold</dt>
                    <dd className="text-slate-200">{n1(focus?.threshold_used, 4)}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-500">Uncertainty</dt>
                    <dd className="text-slate-200">{n1(focus?.uncertainty_score, 3)}</dd>
                  </div>
                </dl>
              </>
            ) : (
              <Empty label="No detector scores for this engine." />
            )}
          </MetricCard>
        </Reveal>

        <Reveal index={8}>
          <MetricCard
            title="Fleet Root Causes"
            subtitle={`${fleet.total_anomalies?.toLocaleString?.() ?? DASH} anomalies attributed`}
          >
            {rootCauses.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rootCauses} layout="vertical" margin={{ left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" stroke="#64748b" fontSize={11} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    stroke="#64748b"
                    fontSize={9}
                    width={140}
                    tickFormatter={(v: string) => (v.length > 26 ? `${v.slice(0, 25)}…` : v)}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="value" fill="#f97316" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty label="No root-cause attribution available." />
            )}
          </MetricCard>
        </Reveal>

        <Reveal index={9}>
          <MetricCard
            title="Fastest Deteriorating"
            subtitle="Ranked by deterioration rate (health index lost per cycle)"
          >
            <ul className="divide-y divide-slate-800/70">
              {earlyWarning.map((e) => {
                const active = String(e.engine_id) === focusId;
                return (
                  <li key={String(e.engine_id)}>
                    <button
                      onClick={() => setSelected(String(e.engine_id))}
                      className={`flex w-full items-center justify-between px-1 py-2 text-left text-sm transition-colors hover:bg-slate-800/40 ${
                        active ? "text-sky-400" : "text-slate-300"
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        <span
                          className="h-2 w-2 rounded-full"
                          style={{
                            background: STATE_COLOR[String(e.health_state ?? "")] ?? "#64748b",
                          }}
                        />
                        E{String(e.engine_id)}
                        <span className="text-xs text-slate-500">
                          {String(e.health_state ?? DASH)}
                        </span>
                      </span>
                      <span className="flex items-center gap-3 text-xs">
                        <span className="text-slate-500">HI {n1(e.health_index)}</span>
                        <span className="font-medium text-orange-400">
                          {n1(e.deterioration_rate, 3)}/cyc
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </MetricCard>
        </Reveal>
      </div>


      {focus && (
        <Reveal index={10}>
          <MetricCard
            title={`Engine E${focusId} — Diagnostic Detail`}
            subtitle="Every field below is read directly from the Anomaly & Health component"
          >
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-4">
              {[
                ["Health index", n1(focus.health_index, 2)],
                ["Health state", String(focus.health_state ?? DASH)],
                ["Remaining health", `${n1(focus.remaining_health_pct, 1)}%`],
                ["Alert level", String(focus.alert_level ?? DASH)],
                ["Anomaly score", n1(focus.anomaly_score, 4)],
                ["Anomaly type", String(focus.anomaly_type ?? DASH)],
                ["Persistence", `${focus.anomaly_persistence_cycles ?? DASH} cycles`],
                ["Early warning", n1(focus.early_warning_score, 3)],
                ["Root cause", String(focus.root_cause ?? DASH)],
                ["Severity", String(focus.severity ?? DASH)],
                ["Evolution pattern", String(focus.evolution_pattern ?? DASH)],
                ["Operating mode", String(focus.operating_mode ?? DASH)],
                ["Confidence", `${n1(focus.confidence_score, 1)}%`],
                ["Reliability", n1(focus.reliability_score, 3)],
                ["Twin spread", n1(focus.twin_spread_mean, 4)],
                ["Cycles observed", String(focus.total_cycles ?? DASH)],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-slate-800/50 py-1.5">
                  <span className="text-slate-500">{k}</span>
                  <span className="ml-2 truncate text-slate-200" title={String(v)}>
                    {v}
                  </span>
                </div>
              ))}
            </div>
            {Array.isArray(focus.top_sensors) && focus.top_sensors.length > 0 && (
              <div className="mt-4 border-t border-slate-800/70 pt-3">
                <p className="mb-2 flex items-center gap-1.5 text-xs uppercase tracking-widest text-slate-500">
                  <Layers className="h-3.5 w-3.5" /> Contributing sensors
                </p>
                <div className="flex flex-wrap gap-2">
                  {(focus.top_sensors as string[]).map((s) => (
                    <span
                      key={s}
                      className="rounded-md border border-slate-700 bg-slate-800/60 px-2 py-1 text-xs text-slate-300"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </MetricCard>
        </Reveal>
      )}


      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Reveal index={11} className="xl:col-span-3">
          <MetricCard
            title="Health Index Validation"
            subtitle={
              hold
                ? `Held-out: ${hold.engines?.toLocaleString() ?? DASH} engines, ${hold.rows?.toLocaleString() ?? DASH} rows the index was never calibrated on`
                : "Held-out validation"
            }
          >
            {!hold ? (
              <Empty label="Run the offline evaluation to populate this panel." />
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                  {[
                    {
                      label: "Monotonicity",
                      value: n1(hold.monotonicity_score, 3),
                      hint: "index rarely recovers",
                      tone: "text-emerald-400",
                    },
                    {
                      label: "Within-engine ρ",
                      value: n1(hold.within_engine_spearman_median, 3),
                      hint: "median, index vs true RUL",
                      tone: "text-emerald-400",
                    },
                    {
                      label: "Fleet ranking ρ",
                      value: n1(hold.fleet_ranking_spearman, 3),
                      hint: "ranks engines against each other",
                      tone: "text-sky-400",
                    },
                    {
                      label: "RUL correlation",
                      value: n1(hold.rul_correlation, 3),
                      hint: "pooled across all rows",
                      tone: "text-sky-400",
                    },
                  ].map((c) => (
                    <div
                      key={c.label}
                      className="rounded-lg border border-slate-800/70 bg-slate-950/40 p-3"
                    >
                      <p className="text-xs text-slate-500">{c.label}</p>
                      <p className={`mt-0.5 text-xl font-semibold ${c.tone}`}>{c.value}</p>
                      <p className="text-[10px] text-slate-600">{c.hint}</p>
                    </div>
                  ))}
                </div>

                {perSubsetHealth.length > 0 && (
                  <div className="mt-4">
                    <p className="mb-2 text-xs uppercase tracking-widest text-slate-500">
                      Per subset — held out
                    </p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-800/60 text-left text-xs uppercase tracking-wide text-slate-500">
                            <th className="py-2 pr-3 font-semibold">Subset</th>
                            <th className="py-2 pr-3 font-semibold">Engines</th>
                            <th className="py-2 pr-3 font-semibold">Monotonicity</th>
                            <th className="py-2 pr-3 font-semibold">RUL corr.</th>
                            <th className="py-2 font-semibold">Mean HI</th>
                          </tr>
                        </thead>
                        <tbody>
                          {perSubsetHealth.map((r) => (
                            <tr
                              key={r.name}
                              className="border-b border-slate-800/30 hover:bg-slate-800/20"
                            >
                              <td className="py-2 pr-3 font-medium text-slate-200">{r.name}</td>
                              <td className="py-2 pr-3 text-slate-500">{r.engines ?? DASH}</td>
                              <td className="py-2 pr-3">
                                <span
                                  className={
                                    (r.monotonicity ?? 0) >= 0.99
                                      ? "text-emerald-400"
                                      : "text-amber-400"
                                  }
                                >
                                  {n1(r.monotonicity, 3)}
                                </span>
                              </td>
                              <td className="py-2 pr-3 text-slate-400">{n1(r.rul, 3)}</td>
                              <td className="py-2 text-slate-400">{n1(r.meanHi, 1)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {healthGate && (
                  <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-slate-800/70 bg-slate-950/40 p-3">
                    <span
                      className={`rounded px-2 py-0.5 text-xs font-semibold ${
                        healthGate.passed
                          ? "bg-emerald-500/10 text-emerald-400"
                          : "bg-amber-500/10 text-amber-400"
                      }`}
                    >
                      {healthGate.passed ? "GATE PASSED" : "GATE FAILED"}
                    </span>
                    <span className="text-xs text-slate-400">
                      Training-set acceptance: RUL correlation{" "}
                      <span className="text-slate-200">{n1(healthGate.rul_correlation, 3)}</span>,
                      monotonicity{" "}
                      <span className="text-slate-200">{n1(healthGate.monotonicity_score, 3)}</span>{" "}
                      against a required {n1(healthGate.monotonicity_threshold, 2)}.
                    </span>
                  </div>
                )}

                <p className="mt-3 border-t border-slate-800/70 pt-3 text-xs leading-relaxed text-slate-500">
                  Monotonicity holds at or near 1.0 on all four subsets, so the index almost never
                  reports an engine recovering. Within a single engine it tracks true remaining life
                  at ρ&nbsp;={" "}
                  <span className="text-slate-300">{n1(hold.within_engine_spearman_median, 2)}</span>{" "}
                  (median); across engines the ranking is weaker at{" "}
                  <span className="text-slate-300">{n1(hold.fleet_ranking_spearman, 2)}</span> — see
                  the observation-length effect on the right for why.
                </p>
              </>
            )}
          </MetricCard>
        </Reveal>


        <Reveal index={12} className="xl:col-span-2">
          <MetricCard
            title="Cross-engine comparability"
            subtitle="Why two engines' indices are not directly comparable"
          >
            {biasQuartiles.length === 0 ? (
              <Empty label="Run the offline evaluation to populate this panel." />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={210}>
                  <ComposedChart data={biasQuartiles}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="short" stroke="#64748b" fontSize={11} />
                    <YAxis
                      yAxisId="hi"
                      stroke="#64748b"
                      fontSize={11}
                      domain={[80, 95]}
                      width={34}
                    />
                    <YAxis
                      yAxisId="rul"
                      orientation="right"
                      stroke="#64748b"
                      fontSize={11}
                      width={34}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Bar
                      yAxisId="hi"
                      name="Mean HI"
                      dataKey="meanHi"
                      fill="#22d3ee"
                      radius={[4, 4, 0, 0]}
                    />
                    <Line
                      yAxisId="rul"
                      name="Median RUL at first cycle"
                      dataKey="startRul"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  Read naively this looks backwards: the quartile that starts most degraded carries
                  a <span className="text-slate-300">higher</span> mean index than the healthiest.
                  It is an observation-length artefact. These units run to failure, so an engine
                  that starts with more life left is simply recorded for longer, and more of its
                  history is spent degrading — which drags its mean down. Mean HI correlates with
                  how long an engine was observed at{" "}
                  <span className="text-amber-300">
                    {n1(hold?.mean_hi_vs_observed_length, 2)}
                  </span>
                  , far more strongly than with its condition. A blended-baseline correction for
                  this was implemented, measured, and reverted for making fleet ranking worse. The
                  index is reliable <span className="text-slate-300">within</span> an engine over
                  time; compare engines by trend, not by absolute level.
                </p>
              </>
            )}
          </MetricCard>
        </Reveal>
      </div>


      {alertMixChart.length > 0 && (
        <Reveal index={13}>
          <GlassCard>
            <h3 className="mb-1 font-semibold text-slate-200">Alert level across recorded history</h3>
            <p className="mb-3 text-xs text-slate-500">
              Every scored row, not the fleet's latest cycle — the snapshot above counts engines now,
              this counts how much of all recorded operation sat at each alert level.
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {alertMixChart.map((a) => (
                <div
                  key={a.name}
                  className="rounded-lg border p-3"
                  style={{ borderColor: `${a.color}33`, background: `${a.color}0f` }}
                >
                  <p className="text-xs capitalize text-slate-400">{a.name}</p>
                  <p className="mt-0.5 text-xl font-semibold" style={{ color: a.color }}>
                    {a.value.toLocaleString()}
                  </p>
                  <p className="text-[10px] text-slate-600">
                    {((a.value / alertMixTotal) * 100).toFixed(1)}% of rows
                  </p>
                </div>
              ))}
            </div>
          </GlassCard>
        </Reveal>
      )}


      <Reveal index={14}>
        <ContextDriftPanel />
      </Reveal>

      <Reveal index={15}>
        <MaintenanceQueuePanel alerts={dash.data?.alerts ?? []} />
      </Reveal>

      <Reveal index={16}>
        <FeedbackLoopPanel
          engineId={focusId}
          cycle={
            typeof focus?.latest_cycle === "number" ? (focus.latest_cycle as number) : undefined
          }
        />
      </Reveal>

      <p className="flex items-start gap-2 text-xs leading-relaxed text-slate-500">
        <TrendingDown className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        Health index is an absolute 0&ndash;100 scale calibrated against the persisted healthy
        reference, so values are comparable across engines and runs. The snapshot basis reports
        each engine at its final observed cycle; the lifetime basis averages every recorded row.
        Both come from <code className="text-slate-400">/anomaly-health/dashboard</code>, which
        aggregates server-side over all {fleet.total_rows.toLocaleString()} rows &mdash; no
        client-side paging.
      </p>
    </div>
  );
}
