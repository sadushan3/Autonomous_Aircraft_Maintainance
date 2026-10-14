/**
 * Digital Twin — the evidence that the twin works.
 *
 * The twin's job is to answer one question: what would a *healthy* engine read
 * under the operating conditions this engine is actually running in? Everything
 * downstream (health index, detectors, root cause) consumes the gap between
 * that answer and the real reading, so this screen is built around making that
 * gap visible on a real engine, then showing that the residual stream it
 * produces passes the three gates it must pass before anyone may rely on it.
 *
 * All figures come from offline artefacts (`metrics/digital_twin.json`,
 * `metrics/evaluate_digital_twin.json`, `metrics/twin_comparison.json`) and the
 * stored prediction/residual tables. Nothing here is recomputed live.
 */
import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Cpu,
  Gauge,
  Layers,
  LineChart as LineChartIcon,
  Search,
  Sigma,
  Waves,
} from "lucide-react";
import { motion } from "motion/react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Pie,
  PieChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GlassCard } from "../components/GlassCard";
import { KPICard } from "../components/KPICard";
import { MetricCard } from "../components/MetricCard";
import {
  AnimatedNumber,
  Empty,
  ErrorState,
  Loading,
  PageHeader,
  Reveal,
} from "../components/Live";
import { useFleetModes, useTwinMetrics, useTwinTrace } from "../../lib/hooks";
import type { TwinEngineSummary } from "../../lib/types";

const MODEL_LABELS: Record<string, string> = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  tensorflow: "Keras MLP",
};

const PATTERN_META: Record<string, { label: string; color: string; hint: string }> = {
  none: { label: "No deviation", color: "#10b981", hint: "reading sits inside the twin's band" },
  narrow: { label: "Narrow", color: "#f59e0b", hint: "one or two sensors adrift — localised fault" },
  broad: { label: "Broad", color: "#ef4444", hint: "many sensors adrift at once — systemic" },
};

const sensorLabel = (s: string) => s.replace("sensor_", "Sensor ");

/**
 * Pooled runs offset unit ids per subset so they never collide, so engine
 * 30249 is really FD004's engine 249. The scheme comes from the backend rather
 * than being assumed here, so it cannot drift from the pipeline config.
 */
function makeEngineLabel(subsets: string[], step: number) {
  return (unit: number) => {
    if (!step || !subsets.length) return `Engine ${unit}`;
    const idx = Math.floor(unit / step);
    const local = unit % step;
    return idx < subsets.length ? `${subsets[idx]} · engine ${local}` : `Engine ${unit}`;
  };
}

const fmt = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d);

export function DigitalTwin() {
  const metrics = useTwinMetrics();
  const modes = useFleetModes();

  const [engineId, setEngineId] = useState<number | null>(null);
  const [sensor, setSensor] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<keyof TwinEngineSummary>("max_abs_residual");
  const [query, setQuery] = useState("");

  const data = metrics.data;
  const engines = data?.engines ?? [];
  const availableSensors = data?.available_sensors ?? [];
  const engineLabel = useMemo(
    () =>
      makeEngineLabel(
        data?.summary.subsets ?? [],
        data?.summary.unit_id_offset_step ?? 0,
      ),
    [data?.summary.subsets, data?.summary.unit_id_offset_step],
  );

  /**
   * Sensors whose standardised residual is trustworthy to *open* on.
   *
   * Two channels are quantised almost to constants, so dividing by their tiny
   * healthy variance inflates their z far past the ceiling. They stay in the
   * picker -- the spread chart below explains them -- but auto-selecting one
   * would open the page on a two-valued staircase instead of a degradation
   * curve.
   */
  const ceiling = data?.validation.spread.ceiling ?? 3.5;
  const p99 = data?.validation.spread.per_sensor_p99_abs_z;
  const wellScaledSensors = useMemo(
    () => availableSensors.filter((name) => (p99?.[name] ?? 0) <= ceiling),
    [availableSensors, p99, ceiling],
  );

  /**
   * Default to the worst-deviating engine that is best *shown* -- the largest
   * peak residual whose dominant sensor is one of the well-scaled channels. An
   * arbitrary "engine 1" often degrades so gently the chart looks like noise.
   */
  const defaultEngine = useMemo(() => {
    if (!engines.length) return null;
    const ranked = [...engines].sort((a, b) => b.max_abs_residual - a.max_abs_residual);
    return (
      ranked.find((e) => wellScaledSensors.includes(e.dominant_sensor_at_worst)) ??
      ranked[0]
    );
  }, [engines, wellScaledSensors]);

  const activeEngineId = engineId ?? defaultEngine?.unit ?? null;
  const activeEngine =
    engines.find((e) => e.unit === activeEngineId) ?? defaultEngine ?? null;

  const fallbackSensor = wellScaledSensors.includes("sensor_11")
    ? "sensor_11"
    : wellScaledSensors[0] ?? availableSensors[0] ?? "sensor_11";

  // Each engine's own worst sensor is the one worth showing first, provided it
  // is a channel whose residual scale means anything.
  const activeSensor =
    sensor ??
    (activeEngine && wellScaledSensors.includes(activeEngine.dominant_sensor_at_worst)
      ? activeEngine.dominant_sensor_at_worst
      : fallbackSensor);

  const trace = useTwinTrace(activeEngineId ?? undefined, activeSensor);

  const chart = useMemo(
    () =>
      (trace.data?.points ?? []).map((p) => ({
        cycle: p.cycle,
        expected: p.expected,
        actual: p.actual,
        band: [p.band_low, p.band_high] as [number, number],
        residual: p.residual,
        anomaly: p.anomaly,
      })),
    [trace.data],
  );

  /**
   * How far the residual travelled between the window the twin was fitted on
   * and this engine's final cycle. A twin that only fits noise would land near
   * 1x; the whole premise of the component is that it does not.
   */
  const growth = useMemo(() => {
    const pts = trace.data?.points ?? [];
    const until = trace.data?.healthy_until_cycle ?? 0;
    const healthy = pts.filter((p) => p.cycle <= until && p.residual !== null);
    const last = [...pts].reverse().find((p) => p.residual !== null);
    if (!healthy.length || !last?.residual) return null;
    const base =
      healthy.reduce((a, p) => a + Math.abs(p.residual as number), 0) / healthy.length;
    if (!base) return null;
    return { base, last: Math.abs(last.residual), factor: Math.abs(last.residual) / base };
  }, [trace.data]);

  const filteredEngines = useMemo(() => {
    const q = query.trim();
    const rows = q ? engines.filter((e) => String(e.unit).includes(q)) : engines;
    return [...rows]
      .sort((a, b) => Number(b[sortKey]) - Number(a[sortKey]))
      .slice(0, 12);
  }, [engines, sortKey, query]);

  if (metrics.isLoading) return <Loading label="Loading digital twin evidence…" />;
  if (metrics.error) return <ErrorState error={metrics.error} />;
  if (!data?.available) {
    return (
      <GlassCard variant="warning">
        <h3 className="mb-1 font-semibold text-slate-200">Digital twin not trained yet</h3>
        <p className="text-sm text-slate-400">
          {data?.message ?? "Run the offline pipeline to produce the twin artefacts."}
        </p>
      </GlassCard>
    );
  }

  const s = data.summary;
  const v = data.validation;

  // Three gates decide whether the residual stream may be consumed downstream.
  // Each bar is stated on the card so the pass is checkable, not asserted.
  const bias = v.centring.mean_absolute_bias;
  const exceed = v.spread.exceedance_fraction;
  const rising = v.growth.fraction_rising;
  const gates = [
    {
      id: "centring",
      title: "Residuals centred on zero",
      display: bias === null ? "—" : bias.toExponential(1),
      target: "target |bias| < 1e-6",
      passed: bias !== null && Math.abs(bias) < 1e-6,
      why: `Mean residual across ${(v.healthy_rows_checked ?? 0).toLocaleString()} healthy rows. Any standing offset here would be inherited as a false deviation by every downstream z-score.`,
    },
    {
      id: "spread",
      title: "Healthy spread stays bounded",
      display: exceed === null ? "—" : `${(exceed * 100).toFixed(2)}%`,
      target: `target ≤ 2% beyond ±${v.spread.ceiling ?? 3.5}σ`,
      passed: exceed !== null && exceed <= 0.02,
      why: `Fraction of healthy rows whose standardised residual exceeds the ±${v.spread.ceiling ?? 3.5}σ ceiling. A well-scaled residual leaves roughly this much in the tail — far more would mean the scale is wrong, far less that it is inflated.`,
    },
    {
      id: "growth",
      title: "Residual rises with age",
      display:
        v.growth.engines_with_rising_residual === null
          ? "—"
          : `${v.growth.engines_with_rising_residual}/${v.growth.engines_checked}`,
      target: "target ≥ 95% of engines",
      passed: rising !== null && rising >= 0.95,
      why: "Engines whose mean residual is larger in late life than in the healthy window. This is the property the whole component rests on: if residuals did not grow with degradation, no downstream detector could work.",
    },
  ];
  const gatesPassed = gates.filter((g) => g.passed).length;

  const maxRmse = Math.max(...data.models.map((m) => m.rmse ?? 0), 0.0001);

  const spreadBySensor = availableSensors.map((name) => ({
    name: sensorLabel(name),
    p99: Number((v.spread.per_sensor_p99_abs_z[name] ?? 0).toFixed(2)),
  }));

  const patternData = Object.entries(data.patterns).map(([k, n]) => ({
    name: PATTERN_META[k]?.label ?? k,
    key: k,
    value: n,
    color: PATTERN_META[k]?.color ?? "#64748b",
  }));
  const patternTotal = patternData.reduce((a, d) => a + d.value, 0) || 1;

  const drift = modes.data?.drift;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Digital Twin"
        subtitle="The healthy-behaviour reference every other layer is measured against"
        right={
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
              gatesPassed === gates.length
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                : "border-amber-500/30 bg-amber-500/10 text-amber-400"
            }`}
          >
            <CheckCircle2 className="h-3.5 w-3.5" />
            {gatesPassed}/{gates.length} residual gates passed
          </span>
        }
      />

      {/* ---------------- Headline: what the twin costs and how well it fits ---- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Reveal index={0}>
          <KPICard
            title="Ensemble RMSE"
            value={<AnimatedNumber value={s.weighted_mean_rmse ?? 0} decimals={4} />}
            icon={Sigma}
            status="info"
            subtitle={`best member ${MODEL_LABELS[s.best_member ?? ""] ?? s.best_member ?? "—"} at ${fmt(s.best_member_rmse, 4)}`}
          />
        </Reveal>
        <Reveal index={1}>
          <KPICard
            title="Twin Models Fused"
            value={<AnimatedNumber value={s.n_twins} />}
            icon={Cpu}
            status="neutral"
            subtitle={data.models.map((m) => MODEL_LABELS[m.name] ?? m.name).join(" · ")}
          />
        </Reveal>
        <Reveal index={2}>
          <KPICard
            title="Member Disagreement"
            value={<AnimatedNumber value={s.mean_twin_spread ?? 0} decimals={3} />}
            icon={Waves}
            status="info"
            subtitle="mean spread across the four twins — the band on the chart below"
          />
        </Reveal>
        <Reveal index={3}>
          <KPICard
            title="Sensors Modelled"
            value={
              <span>
                <AnimatedNumber value={s.informative_sensors ?? 0} />
                <span className="text-lg text-slate-500">/{s.sensors_modelled ?? 0}</span>
              </span>
            }
            icon={Layers}
            status="neutral"
            subtitle={`${s.operating_modes ?? "—"} operating regimes · ${(s.rows ?? 0).toLocaleString()} rows · ${s.engines ?? 0} engines`}
          />
        </Reveal>
      </div>

      {/* ---------------- Hero: the gap IS the signal ------------------------- */}
      <Reveal index={4}>
        <MetricCard
          title="Actual reading vs. twin's healthy expectation"
          subtitle={
            trace.data
              ? `${engineLabel(trace.data.engine_id)} · ${sensorLabel(trace.data.sensor)} · ${trace.data.cycles_total} cycles · ${trace.data.units}`
              : "Select an engine and sensor"
          }
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <select
                value={activeEngineId ?? ""}
                onChange={(e) => {
                  setEngineId(Number(e.target.value));
                  setSensor(null); // follow the new engine's own worst sensor
                }}
                className="rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cyan-500"
              >
                {engines.map((e) => (
                  <option key={e.unit} value={e.unit}>
                    {engineLabel(e.unit)}
                  </option>
                ))}
              </select>
              <select
                value={activeSensor}
                onChange={(e) => setSensor(e.target.value)}
                className="rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cyan-500"
              >
                {availableSensors.map((name) => (
                  <option key={name} value={name}>
                    {sensorLabel(name)}
                  </option>
                ))}
              </select>
            </div>
          }
        >
          {trace.isLoading ? (
            <Loading label="Joining prediction and residual tables…" />
          ) : trace.error ? (
            <ErrorState error={trace.error} />
          ) : !chart.length ? (
            <Empty />
          ) : (
            <>
              <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-400">
                <span className="flex items-center gap-2">
                  <svg width="22" height="8" aria-hidden>
                    <line x1="0" y1="4" x2="22" y2="4" stroke="#f59e0b" strokeWidth="2" />
                  </svg>
                  actual reading
                </span>
                <span className="flex items-center gap-2">
                  <svg width="22" height="8" aria-hidden>
                    <line
                      x1="0"
                      y1="4"
                      x2="22"
                      y2="4"
                      stroke="#22d3ee"
                      strokeWidth="2"
                      strokeDasharray="5 4"
                    />
                  </svg>
                  twin's healthy expectation
                </span>
                <span className="flex items-center gap-2">
                  <span className="h-3 w-5 rounded-sm bg-cyan-400/25" />
                  where the four twins disagree
                </span>
                <span className="flex items-center gap-2">
                  <span className="h-3 w-5 rounded-sm bg-emerald-500/20" />
                  window the twin was fitted on
                </span>
              </div>
              <ResponsiveContainer width="100%" height={340}>
                <ComposedChart data={chart} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="cycle"
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    label={{ value: "cycle", position: "insideBottomRight", fill: "#475569", fontSize: 11 }}
                  />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                    formatter={(value: unknown, name: string) => {
                      if (name === "band" && Array.isArray(value))
                        return [`${fmt(value[0])} … ${fmt(value[1])}`, "twin agreement band"];
                      return [fmt(value as number), name];
                    }}
                    labelFormatter={(c) => `Cycle ${c}`}
                  />

                  {/* The window the twin was actually fitted on. */}
                  {trace.data && (
                    <ReferenceArea
                      x1={chart[0].cycle}
                      x2={trace.data.healthy_until_cycle}
                      fill="#10b981"
                      fillOpacity={0.07}
                      label={{ value: "twin fitted here", fill: "#10b981", fontSize: 10, position: "insideTopLeft" }}
                    />
                  )}

                  {/* Where the four members disagree — the twin's own uncertainty. */}
                  <Area
                    dataKey="band"
                    stroke="none"
                    fill="#22d3ee"
                    fillOpacity={0.22}
                    isAnimationActive={false}
                    name="band"
                  />
                  <Line
                    type="monotone"
                    dataKey="expected"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    dot={false}
                    name="twin expectation (healthy)"
                    animationDuration={900}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                    name="actual reading"
                    animationDuration={900}
                  />

                  {trace.data?.onset_cycle != null && (
                    <ReferenceLine
                      x={trace.data.onset_cycle}
                      stroke="#ef4444"
                      strokeDasharray="4 4"
                      label={{
                        value: `anomaly raised · cycle ${trace.data.onset_cycle}`,
                        fill: "#ef4444",
                        fontSize: 10,
                        position: "top",
                      }}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>

              <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
                <TraceStat
                  label="Residual in fitted window"
                  value={growth ? fmt(growth.base, 2) : "—"}
                  tone="ok"
                />
                <TraceStat
                  label="Residual at last cycle"
                  value={growth ? fmt(growth.last, 2) : "—"}
                  tone="bad"
                />
                <TraceStat
                  label="Growth factor"
                  value={growth ? `${growth.factor.toFixed(1)}×` : "—"}
                  tone="bad"
                />
                <TraceStat
                  label="Warning lead time"
                  value={
                    trace.data?.onset_cycle != null
                      ? `${trace.data.last_cycle - trace.data.onset_cycle} cycles`
                      : "not flagged"
                  }
                  tone={trace.data?.onset_cycle != null ? "warn" : "ok"}
                />
              </div>

              <p className="mt-3 text-xs text-slate-500">
                The dashed line is what the twin says a <span className="text-cyan-300">healthy</span> engine would
                read under this engine's own operating conditions; the shaded band is where its four members disagree.
                Both series are corrected for this engine's early-life offset, so the vertical gap between them is
                exactly the residual every downstream layer consumes — inside the band while the engine is healthy,
                opening steadily as it degrades.
              </p>
            </>
          )}
        </MetricCard>
      </Reveal>

      {/* ---------------- Ensemble leaderboard -------------------------------- */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Reveal index={5} className="xl:col-span-3">
          <MetricCard
            title="Ensemble members"
            subtitle={`Trained on ${(s.train_rows ?? 0).toLocaleString()} healthy rows, validated on ${(s.val_rows ?? 0).toLocaleString()} held out · ${s.n_features ?? "—"} context features → ${s.sensors_modelled ?? "—"} sensor targets`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800/60 text-left text-xs uppercase tracking-wide text-slate-500">
                    <th className="py-2 pr-3 font-semibold">Model</th>
                    <th className="py-2 pr-3 font-semibold">RMSE</th>
                    <th className="py-2 pr-3 font-semibold">MAE</th>
                    <th className="py-2 pr-3 font-semibold">Fusion weight</th>
                    <th className="py-2 font-semibold">Fit</th>
                  </tr>
                </thead>
                <tbody>
                  {data.models.map((m) => (
                    <tr key={m.name} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                      <td className="py-3 pr-3">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-slate-200">
                            {MODEL_LABELS[m.name] ?? m.name}
                          </span>
                          {m.is_best && (
                            <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-400">
                              BEST
                            </span>
                          )}
                        </div>
                        {m.epochs_trained ? (
                          <div className="mt-0.5 text-[11px] text-slate-500">
                            {m.epochs_trained} epochs (early-stopped)
                          </div>
                        ) : null}
                      </td>
                      <td className="py-3 pr-3">
                        <div className="text-slate-300">{fmt(m.rmse, 4)}</div>
                        <div className="mt-1 h-1 w-24 overflow-hidden rounded-full bg-slate-800/70">
                          <div
                            className={`h-full rounded-full ${m.is_best ? "bg-emerald-500" : "bg-cyan-500/70"}`}
                            style={{ width: `${((m.rmse ?? 0) / maxRmse) * 100}%` }}
                          />
                        </div>
                      </td>
                      <td className="py-3 pr-3 text-slate-400">{fmt(m.mae, 4)}</td>
                      <td className="py-3 pr-3 text-slate-400">
                        {m.weight === null ? "—" : `${(m.weight * 100).toFixed(1)}%`}
                      </td>
                      <td className="py-3 text-slate-500">{fmt(m.fit_seconds, 1)}s</td>
                    </tr>
                  ))}
                  <tr className="bg-cyan-500/5">
                    <td className="py-3 pr-3 font-semibold text-cyan-300">Fused ensemble</td>
                    <td className="py-3 pr-3 font-semibold text-cyan-300">
                      {fmt(s.weighted_mean_rmse, 4)}
                    </td>
                    <td className="py-3 pr-3 text-slate-500">—</td>
                    <td className="py-3 pr-3 text-slate-500">100%</td>
                    <td className="py-3 text-slate-500">
                      {fmt(data.models.reduce((a, m) => a + (m.fit_seconds ?? 0), 0), 1)}s total
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Members are fused by inverse RMSE, which is why the weights sit so close to even — no member is
              materially stronger than the others. The point of keeping four is not accuracy but{" "}
              <span className="text-cyan-300">disagreement</span>: where they diverge, the twin is guessing, and that
              spread becomes the uncertainty band rather than being averaged away.
            </p>
          </MetricCard>
        </Reveal>

        {/* ---------------- Acceptance gates ---------------------------------- */}
        <Reveal index={6} className="xl:col-span-2">
          <MetricCard
            title="Residual acceptance gates"
            subtitle={`Checked on ${(v.healthy_rows_checked ?? 0).toLocaleString()} healthy rows of ${(v.total_rows ?? 0).toLocaleString()}`}
          >
            <div className="space-y-3">
              {gates.map((g, i) => (
                <motion.div
                  key={g.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className={`rounded-lg border p-3 ${
                    g.passed
                      ? "border-emerald-500/25 bg-emerald-500/[0.06]"
                      : "border-amber-500/25 bg-amber-500/[0.06]"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2">
                      {g.passed ? (
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" />
                      )}
                      <span className="text-sm font-medium text-slate-200">{g.title}</span>
                    </div>
                    <span
                      className={`shrink-0 font-mono text-sm font-semibold ${
                        g.passed ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {g.display}
                    </span>
                  </div>
                  <div className="mt-1 pl-6 text-[11px] uppercase tracking-wide text-slate-500">
                    {g.target}
                  </div>
                  <p className="mt-1.5 pl-6 text-xs leading-relaxed text-slate-400">{g.why}</p>
                </motion.div>
              ))}
            </div>
          </MetricCard>
        </Reveal>
      </div>

      {/* ---------------- Per-sensor spread + deviation patterns -------------- */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Reveal index={7}>
          <MetricCard
            title="Residual spread by sensor"
            subtitle={`99th percentile |z| on healthy rows, against the ±${v.spread.ceiling ?? 3.5}σ ceiling`}
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={spreadBySensor} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis
                  dataKey="name"
                  type="category"
                  stroke="#64748b"
                  width={78}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                <ReferenceLine
                  x={v.spread.ceiling ?? 3.5}
                  stroke="#ef4444"
                  strokeDasharray="4 4"
                  label={{ value: "ceiling", fill: "#ef4444", fontSize: 10 }}
                />
                <Bar dataKey="p99" radius={[0, 4, 4, 0]} animationDuration={900}>
                  {spreadBySensor.map((d) => (
                    <Cell
                      key={d.name}
                      fill={d.p99 > (v.spread.ceiling ?? 3.5) ? "#f59e0b" : "#22d3ee"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-2 text-xs text-slate-500">
              Fifteen of the seventeen modelled channels sit well inside the ceiling. Sensors 10 and 16 do not, and the
              reason is in the raw data rather than the twin: sensor 16 takes only two distinct values across the whole
              fleet and sensor 10 twenty-one, so there is almost no healthy variance to divide by and the standardised
              residual inflates. They are quantised status channels, not continuous measurements.
            </p>
          </MetricCard>
        </Reveal>

        <Reveal index={8}>
          <MetricCard
            title="Deviation pattern across the fleet"
            subtitle={`${(s.rows ?? 0).toLocaleString()} scored cycles · mean ${fmt(data.mean_deviating_sensors, 2)} sensors deviating at once`}
          >
            <ResponsiveContainer width="100%" height={230}>
              <PieChart>
                <Pie
                  data={patternData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={62}
                  outerRadius={95}
                  paddingAngle={3}
                  animationDuration={900}
                >
                  {patternData.map((d) => (
                    <Cell key={d.key} fill={d.color} stroke="#0f172a" />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                  formatter={(value: unknown) => [
                    `${(value as number).toLocaleString()} cycles (${(((value as number) / patternTotal) * 100).toFixed(1)}%)`,
                    "count",
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-2 space-y-2">
              {patternData.map((d) => (
                <div key={d.key} className="flex items-center justify-between gap-3 text-xs">
                  <span className="flex items-center gap-2 text-slate-300">
                    <span className="h-2 w-2 rounded-full" style={{ background: d.color }} />
                    {d.name}
                    <span className="text-slate-500">— {PATTERN_META[d.key]?.hint}</span>
                  </span>
                  <span className="font-mono text-slate-400">
                    {((d.value / patternTotal) * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </MetricCard>
        </Reveal>
      </div>

      {/* ---------------- Per-engine fidelity table -------------------------- */}
      <Reveal index={9}>
        <MetricCard
          title="Per-engine twin fidelity"
          subtitle={`All ${engines.length} engines ranked — click a row to trace it above`}
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-600" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Engine id"
                  className="w-28 rounded-lg border border-slate-700 bg-slate-950/60 py-1.5 pl-8 pr-2 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-cyan-500"
                />
              </div>
              <select
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value as keyof TwinEngineSummary)}
                className="rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cyan-500"
              >
                <option value="max_abs_residual">Sort by peak residual</option>
                <option value="mean_abs_residual">Sort by mean residual</option>
                <option value="mean_deviating_sensors">Sort by sensors deviating</option>
                <option value="last_cycle">Sort by life length</option>
              </select>
            </div>
          }
        >
          {filteredEngines.length === 0 ? (
            <Empty label="No engine matches that number." />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800/60 text-left text-xs uppercase tracking-wide text-slate-500">
                    <th className="py-2 pr-3 font-semibold">Engine</th>
                    <th className="py-2 pr-3 font-semibold">Life</th>
                    <th className="py-2 pr-3 font-semibold">Mean |residual|</th>
                    <th className="py-2 pr-3 font-semibold">Peak |residual|</th>
                    <th className="py-2 pr-3 font-semibold">Worst cycle</th>
                    <th className="py-2 pr-3 font-semibold">Dominant sensor</th>
                    <th className="py-2 font-semibold">Pattern</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEngines.map((e) => {
                    const active = e.unit === activeEngineId;
                    const meta = PATTERN_META[e.dominant_pattern];
                    return (
                      <tr
                        key={e.unit}
                        onClick={() => {
                          setEngineId(e.unit);
                          setSensor(null);
                        }}
                        className={`cursor-pointer border-b border-slate-800/30 transition-colors ${
                          active ? "bg-cyan-500/10" : "hover:bg-slate-800/25"
                        }`}
                      >
                        <td className="py-2.5 pr-3 font-medium text-slate-200">
                          {active && <span className="mr-1.5 text-cyan-400">▸</span>}
                          {engineLabel(e.unit)}
                        </td>
                        <td className="py-2.5 pr-3 text-slate-500">{e.last_cycle}</td>
                        <td className="py-2.5 pr-3 text-slate-400">{fmt(e.mean_abs_residual, 3)}</td>
                        <td className="py-2.5 pr-3">
                          <span className="font-mono text-amber-400">
                            {fmt(e.max_abs_residual, 3)}
                          </span>
                        </td>
                        <td className="py-2.5 pr-3 text-slate-500">
                          {e.worst_cycle}
                          <span className="ml-1 text-[11px] text-slate-600">
                            ({Math.round((e.worst_cycle / e.last_cycle) * 100)}% of life)
                          </span>
                        </td>
                        <td className="py-2.5 pr-3 text-slate-400">
                          {sensorLabel(e.dominant_sensor_at_worst)}
                        </td>
                        <td className="py-2.5">
                          <span
                            className="rounded px-1.5 py-0.5 text-[11px] font-medium"
                            style={{
                              color: meta?.color ?? "#94a3b8",
                              background: `${meta?.color ?? "#64748b"}18`,
                            }}
                          >
                            {meta?.label ?? e.dominant_pattern}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <p className="mt-3 text-xs text-slate-500">
            Showing the top 12 of {engines.length}. "Worst cycle" is where the twin's expectation and the real reading
            diverge most; that it clusters near the end of life is the fleet-wide version of the single-engine chart
            above.
          </p>
        </MetricCard>
      </Reveal>

      {/* ---------------- How it works + operating context -------------------- */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Reveal index={10} className="lg:col-span-2">
          <GlassCard variant="primary">
            <h3 className="mb-3 flex items-center gap-2 font-semibold text-slate-200">
              <LineChartIcon className="h-4 w-4 text-cyan-400" />
              How the twin produces a residual
            </h3>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
              {[
                {
                  n: "01",
                  t: "Fit on healthy only",
                  d: `${(s.train_rows ?? 0).toLocaleString()} rows where RUL is high and the engine is in the first 30% of its life.`,
                },
                {
                  n: "02",
                  t: "Predict per regime",
                  d: `${s.n_features ?? "—"} operating-context features in, ${s.sensors_modelled ?? "—"} sensor values out, across ${s.operating_modes ?? "—"} regimes.`,
                },
                {
                  n: "03",
                  t: "Fuse four members",
                  d: "Inverse-RMSE weighting; the spread between them is kept as the uncertainty band.",
                },
                {
                  n: "04",
                  t: "Centre and standardise",
                  d: "Against the engine's own early-life baseline, then a reference frozen at training time.",
                },
              ].map((step, i) => (
                <motion.div
                  key={step.n}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.05 * i }}
                  className="rounded-lg border border-slate-800/60 bg-slate-950/40 p-3"
                >
                  <div className="text-[10px] font-bold tracking-widest text-cyan-500/70">{step.n}</div>
                  <div className="mt-1 text-sm font-medium text-slate-200">{step.t}</div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-500">{step.d}</p>
                </motion.div>
              ))}
            </div>
            <p className="mt-3 text-xs text-slate-500">
              Freezing the standardisation reference at training time is what stops an engine's score from depending on
              which other engines happened to be scored alongside it.
            </p>
          </GlassCard>
        </Reveal>

        <Reveal index={11}>
          <MetricCard title="Operating context" subtitle="Conditions the twin is currently predicting under">
            {modes.isLoading ? (
              <Loading label="Loading context…" />
            ) : !drift ? (
              <Empty label="Context model not available." />
            ) : (
              <div className="space-y-3">
                <ContextRow
                  icon={Layers}
                  label="Operating regimes"
                  value={String(Object.keys(modes.data?.mode_distribution ?? {}).length)}
                />
                <ContextRow
                  icon={Gauge}
                  label="Context drift (PSI)"
                  value={fmt(drift.psi, 3)}
                  sub={`threshold ${drift.threshold}`}
                  tone={drift.drift_detected ? "warn" : "ok"}
                />
                <ContextRow
                  icon={Activity}
                  label="Drift severity"
                  value={drift.drift_detected ? drift.severity : "stable"}
                  tone={drift.drift_detected ? "warn" : "ok"}
                />
                <ContextRow
                  icon={Boxes}
                  label="Engines scored"
                  value={(s.engines ?? 0).toLocaleString()}
                />
                <p className="pt-1 text-xs leading-relaxed text-slate-500">
                  A shifted regime mix is expected when the scored fleet is not the one the reference was built from —
                  it changes what the twin is asked to predict, not whether it is right.
                </p>
              </div>
            )}
          </MetricCard>
        </Reveal>
      </div>
    </div>
  );
}

function TraceStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "ok" | "warn" | "bad";
}) {
  const colors = {
    ok: "text-emerald-400 border-emerald-500/20 bg-emerald-500/[0.06]",
    warn: "text-amber-400 border-amber-500/20 bg-amber-500/[0.06]",
    bad: "text-rose-400 border-rose-500/20 bg-rose-500/[0.06]",
  } as const;
  return (
    <div className={`rounded-lg border p-3 ${colors[tone]}`}>
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 font-mono text-xl font-semibold ${colors[tone].split(" ")[0]}`}>
        {value}
      </div>
    </div>
  );
}

function ContextRow({
  icon: Icon,
  label,
  value,
  sub,
  tone = "neutral",
}: {
  icon: typeof Layers;
  label: string;
  value: string;
  sub?: string;
  tone?: "ok" | "warn" | "neutral";
}) {
  const color =
    tone === "warn" ? "text-amber-400" : tone === "ok" ? "text-emerald-400" : "text-slate-200";
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-800/60 bg-slate-950/40 p-3">
      <span className="flex items-center gap-2 text-sm text-slate-400">
        <Icon className="h-4 w-4 text-slate-500" />
        {label}
      </span>
      <span className="text-right">
        <span className={`font-semibold capitalize ${color}`}>{value}</span>
        {sub && <span className="block text-[11px] text-slate-600">{sub}</span>}
      </span>
    </div>
  );
}
