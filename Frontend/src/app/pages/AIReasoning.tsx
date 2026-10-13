import { useState } from "react";
import {
  Activity,
  Brain,
  CheckCircle2,
  Clock,
  History,
  Lightbulb,
  Network,
  ScrollText,
  ShieldCheck,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
  LiveBadge,
  Loading,
  PageHeader,
  Reveal,
} from "../components/Live";
import {
  useAnomalyDashboard,
  useAnomalyEvaluation,
  useEngineReasoning,
  useSensorGraph,
} from "../../lib/hooks";
import { useEngineParam } from "../../lib/engine-param";
import { FeedbackControl } from "../components/FeedbackControl";
import { SensorGraph } from "../components/SensorGraph";
import { EngineAnatomy } from "../components/EngineAnatomy";
import type { AnomalyAlert, RootCauseRecord } from "../../lib/types";

const PALETTE = ["#ef4444", "#f97316", "#f59e0b", "#a855f7", "#22d3ee", "#10b981", "#6366f1", "#ec4899"];

/** top_sensors arrives as a real array or as a stringified Python list. */
function sensorList(v: unknown): string[] {
  if (Array.isArray(v)) return v.map(String);
  if (typeof v === "string") {
    const m = v.match(/'([^']+)'|"([^"]+)"/g);
    if (m) return m.map((s) => s.replace(/['"]/g, ""));
    return v.split(",").map((s) => s.trim()).filter(Boolean);
  }
  return [];
}

const sevColor = (s: string | null) =>
  s === "critical" ? "danger" : s === "high" ? "warning" : s === "medium" ? "primary" : "default";

export function AIReasoning() {
  const dash = useAnomalyDashboard();
  const graph = useSensorGraph();
  const [selected, setSelected] = useEngineParam();

  const evaluation = useAnomalyEvaluation();

  const alerts: AnomalyAlert[] = dash.data?.alerts ?? [];
  const explained = alerts.filter((a) => a.explanation_text);
  const focusId = selected ?? String(explained[0]?.engine_id ?? "");
  const focus = explained.find((a) => String(a.engine_id) === focusId) ?? explained[0];
  const reasoning = useEngineReasoning(focusId);

  /**
   * The reasoning record for the cycle the focus alert was raised on.
   *
   * `/reasoning/{id}` returns the engine's history newest-first; matching the
   * alert's own cycle keeps the chain, the narrative and the sensor ranking
   * describing the same moment rather than three different ones.
   */
  const record: RootCauseRecord | null = (() => {
    const records = reasoning.data?.records ?? [];
    if (!records.length) return null;
    return records.find((r) => r.cycle === focus?.cycle) ?? records[0];
  })();

  /** Largest deviation in the focused alert, so the contribution bars scale to it. */
  const zMax = (() => {
    const zs = (record?.sensor_contributions ?? []).map((c) => Math.abs(c.residual_z ?? 0));
    const top = Math.max(...zs, 0);
    return top > 0 ? top : 1;
  })();

  if (dash.isLoading) return <Loading label="Loading explainability layer…" />;
  if (dash.error) return <ErrorState error={dash.error} />;
  if (!dash.data) return <Empty />;

  const fleet = dash.data.fleet;

  /**
   * `fleet.top_root_causes` is capped at five server-side -- it is a "top
   * causes" widget field, not the full set. Counting its keys reported five
   * distinct causes when the reasoner actually distinguishes seven, and the
   * chart silently dropped the two rarest. The reasoning-stage rollup carries
   * all of them, so prefer it and fall back only when it is unavailable.
   */
  const causes = evaluation.reasoning?.cause_distribution ?? fleet.top_root_causes ?? {};
  const causeTruncated = !evaluation.reasoning?.cause_distribution;
  const causeTotal = Object.values(causes).reduce((a, b) => a + Number(b), 0) || 1;
  const causeChart = Object.entries(causes)
    .map(([name, value]) => ({ name, value: Number(value) }))
    .sort((a, b) => b.value - a.value);

  const evolutionChart = Object.entries(evaluation.reasoning?.evolution_distribution ?? {})
    .map(([name, value]) => ({ name, value: Number(value) }))
    .sort((a, b) => b.value - a.value);
  const evolutionTotal = evolutionChart.reduce((a, d) => a + d.value, 0) || 1;

  const rGate = evaluation.reasoningGate;
  const xGate = evaluation.explainGate;

  // Signed local group attributions for the focused alert and sensor.
  const shapRanking = (() => {
    const raw = focus?.shap_top_features;
    const list = Array.isArray(raw) ? raw : [];
    const parsed = list
      .map((f) => {
        const item = f as { feature?: unknown; shap_value?: unknown };
        return {
          feature: String(item.feature ?? ""),
          value: Number(item.shap_value ?? 0),
        };
      })
      .filter((f) => f.feature && Number.isFinite(f.value));
    const informative = parsed.some((f) => Math.abs(f.value) > 0);
    return { rows: parsed, informative };
  })();

  const engine = dash.data.engines.find((e) => String(e.engine_id) === focusId);
  const topSensors = sensorList(engine?.top_sensors ?? focus?.top_sensors);

  const nodes = (graph.data?.n_nodes as number | undefined) ?? 0;
  const edges = (graph.data?.n_edges as number | undefined) ?? 0;
  const graphNodes = (graph.data?.nodes as string[] | undefined) ?? [];
  const graphEdges =
    (graph.data?.edges as Array<{ source: string; target: string; relation: string; weight: number }> | undefined) ?? [];

  // The root cause is prose ("High-Pressure Turbine degradation"); match it to
  // a graph node id so the propagation path can be highlighted.
  const CAUSE_TO_NODE: Array<[RegExp, string]> = [
    [/high-pressure turbine|hpt/i, "hpt"],
    [/low-pressure turbine|lpt/i, "lpt"],
    [/high-pressure compressor|hpc/i, "hpc"],
    [/low-pressure compressor|lpc/i, "lpc"],
    [/combustor|burner|fuel-air/i, "burner"],
    [/bypass|fan-flow/i, "bypass"],
    [/bleed/i, "bleed"],
    [/fan/i, "fan"],
  ];
  const faultyNode =
    CAUSE_TO_NODE.find(([re]) => re.test(String(focus?.root_cause ?? "")))?.[1] ?? null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="AI Reasoning & Explainability"
        subtitle="Why each anomaly was raised — root cause, contributing sensors, subsystem and generated narrative"
        right={<LiveBadge />}
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Reveal index={0}>
          <KPICard title="Explained Alerts" value={<AnimatedNumber value={explained.length} />} icon={ScrollText} status="info" />
        </Reveal>
        <Reveal index={1}>
          <KPICard title="Fault Anomalies" value={<AnimatedNumber value={fleet.fault_anomalies} />} icon={Brain} status="danger" />
        </Reveal>
        <Reveal index={2}>
          <KPICard
            title="Distinct Root Causes"
            value={
              <AnimatedNumber
                value={rGate?.n_distinct_causes ?? Object.keys(causes).length}
              />
            }
            icon={Lightbulb}
            status="warning"
            subtitle={`across ${rGate?.n_distinct_subsystems ?? Object.keys(evaluation.reasoning?.subsystem_distribution ?? {}).length} subsystems`}
          />
        </Reveal>
        <Reveal index={3}>
          <KPICard
            title="Hard Causal Claims"
            value={<AnimatedNumber value={rGate?.hard_causal_claims ?? 0} />}
            icon={ShieldCheck}
            status="success"
            subtitle={`every one of ${(rGate?.rows_reasoned ?? 0).toLocaleString()} conclusions is a rule-supported hypothesis`}
          />
        </Reveal>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Reveal index={4}>
          <MetricCard
            title="Root Cause Attribution — Fleet"
            subtitle="Fault cycles per attributed cause"
          >
            {causeChart.length ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={causeChart} layout="vertical" margin={{ left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" stroke="#64748b" />
                  <YAxis dataKey="name" type="category" stroke="#64748b" width={210} tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]} animationDuration={900}>
                    {causeChart.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty label="No root causes attributed in this run." />
            )}
            <p className="mt-2 text-xs text-slate-500">
              {causeTruncated ? (
                <>Top causes only — run the offline evaluation to show the full distribution.</>
              ) : (
                <>
                  All {causeChart.length} causes the reasoner distinguishes, over{" "}
                  {causeTotal.toLocaleString()} attributed fault cycles. The leading cause accounts
                  for {((causeChart[0]?.value ?? 0) / causeTotal * 100).toFixed(1)}%.
                </>
              )}
            </p>
          </MetricCard>
        </Reveal>

        <Reveal index={5}>
          <MetricCard title="Anomaly Breakdown">
            <div className="grid grid-cols-3 gap-3">
              {[
                ["Total", fleet.total_anomalies, "primary"],
                ["Genuine faults", fleet.fault_anomalies, "danger"],
                ["Operational", fleet.operational_anomalies ?? 0, "warning"],
              ].map(([label, val, variant], i) => (
                <motion.div
                  key={String(label)}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.08 }}
                >
                  <GlassCard variant={variant as never}>
                    <div className="text-xs uppercase tracking-wide text-slate-500">{String(label)}</div>
                    <div className="mt-1 text-2xl font-bold text-slate-100">{Number(val).toLocaleString()}</div>
                  </GlassCard>
                </motion.div>
              ))}
            </div>
            <p className="mt-4 text-xs text-slate-500">
              The operational-vs-fault gate suppresses flags that coincide with a regime transition and have not yet
              persisted, so a genuine operating-mode change is not reported as a fault.
            </p>
          </MetricCard>
        </Reveal>
      </div>

      <Reveal index={6}>
        <MetricCard title="Explained Alerts — select an engine">
          <div className="mb-4 flex flex-wrap gap-2">
            {explained.slice(0, 18).map((a) => (
              <button
                key={`${a.engine_id}-${a.cycle}`}
                onClick={() => setSelected(String(a.engine_id))}
                className={`rounded-lg border px-3 py-1 text-xs transition-all hover:scale-105 ${
                  focusId === String(a.engine_id)
                    ? "border-purple-500/50 bg-purple-500/20 text-purple-300"
                    : "border-slate-700/50 bg-slate-800/40 text-slate-400"
                }`}
              >
                E{String(a.engine_id)}
                {a.severity === "critical" && <span className="ml-1 text-red-400">●</span>}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {focus ? (
              <motion.div
                key={focusId}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="space-y-4"
              >
                <GlassCard variant={sevColor(focus.severity) as never}>
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                    {[
                      ["Alert Level", focus.alert_level ?? "—"],
                      ["Severity", focus.severity ?? "—"],
                      ["Cycle", String(focus.cycle)],
                      ["Evolution", focus.evolution_pattern ?? "—"],
                    ].map(([k, v], i) => (
                      <motion.div
                        key={k}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.06 }}
                      >
                        <div className="text-xs text-slate-500">{k}</div>
                        <div className="mt-0.5 font-semibold capitalize text-slate-100">{v}</div>
                      </motion.div>
                    ))}
                  </div>
                </GlassCard>

                {/* The generated narrative — the actual explainability output. */}
                <div className="rounded-lg border border-purple-500/25 bg-gradient-to-br from-purple-900/20 to-slate-900/20 p-5">
                  <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-purple-300">
                    <Sparkles className="h-4 w-4" /> Generated Explanation
                  </div>
                  <p className="leading-relaxed text-slate-200">{focus.explanation_text}</p>
                </div>

                {/* ---- the actual reasoning: what else it considered, and why ---- */}
                {record && record.candidate_causes.length > 0 && (
                  <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-4">
                    <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-slate-300">
                      <Brain className="h-4 w-4 text-purple-400" />
                      Reasoning chain — {record.candidate_causes.length} hypotheses considered
                    </div>
                    <p className="mb-3 text-xs text-slate-500">
                      Each hypothesis was raised by a named rule and carries the evidence that fired
                      it. The winner is the highest likelihood, not a certainty — the alternatives
                      stay on the record.
                    </p>
                    <div className="space-y-2">
                      {[...record.candidate_causes]
                        .sort((a, b) => b.likelihood - a.likelihood)
                        .map((c, i) => {
                          const leading = i === 0;
                          return (
                            <motion.div
                              key={`${c.rule_id}-${c.cause}`}
                              initial={{ opacity: 0, x: -8 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.07 }}
                              className={`rounded-lg border p-3 ${
                                leading
                                  ? "border-purple-500/40 bg-purple-500/[0.07]"
                                  : "border-slate-800/70 bg-slate-950/40"
                              }`}
                            >
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <span className="flex items-center gap-2">
                                  <span
                                    className={`text-sm font-medium ${
                                      leading ? "text-purple-200" : "text-slate-300"
                                    }`}
                                  >
                                    {c.cause}
                                  </span>
                                  {leading && (
                                    <span className="rounded bg-purple-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-purple-300">
                                      SELECTED
                                    </span>
                                  )}
                                </span>
                                <span
                                  className={`font-mono text-sm font-semibold ${
                                    leading ? "text-purple-300" : "text-slate-400"
                                  }`}
                                >
                                  {c.likelihood.toFixed(2)} support
                                </span>
                              </div>

                              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800/70">
                                <motion.div
                                  className={`h-full rounded-full ${
                                    leading ? "bg-purple-400" : "bg-slate-600"
                                  }`}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.min(c.likelihood * 100, 100)}%` }}
                                  transition={{ duration: 0.7, delay: i * 0.07 }}
                                />
                              </div>

                              {c.evidence.length > 0 && (
                                <ul className="mt-2 space-y-1">
                                  {c.evidence.map((e) => (
                                    <li
                                      key={e}
                                      className="flex gap-2 text-xs leading-relaxed text-slate-400"
                                    >
                                      <span className="mt-[3px] h-1 w-1 shrink-0 rounded-full bg-slate-600" />
                                      {e}
                                    </li>
                                  ))}
                                </ul>
                              )}

                              <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px]">
                                {c.rule_id && (
                                  <code className="rounded bg-slate-800/70 px-1.5 py-0.5 font-mono text-slate-400">
                                    {c.rule_id}
                                  </code>
                                )}
                                <span className="rounded bg-slate-800/40 px-1.5 py-0.5 uppercase tracking-wide text-slate-500">
                                  {c.subsystem}
                                </span>
                                {c.inspection_focus && (
                                  <span className="text-slate-500">
                                    inspect: <span className="text-slate-400">{c.inspection_focus}</span>
                                  </span>
                                )}
                              </div>
                            </motion.div>
                          );
                        })}
                    </div>
                  </div>
                )}

                {/* ---- when it started, how long, how fast ---------------------- */}
                {record && (
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    {[
                      {
                        icon: Clock,
                        label: "Onset cycle",
                        value: record.onset_cycle === null ? "—" : String(record.onset_cycle),
                        hint:
                          record.onset_cycle !== null && focus
                            ? `${focus.cycle - record.onset_cycle} cycles ago`
                            : "deviation start",
                      },
                      {
                        icon: Activity,
                        label: "Persisted",
                        value: `${record.duration_cycles} cycles`,
                        hint: "how long it has held",
                      },
                      {
                        icon: TrendingUp,
                        label: "Trend slope",
                        value: record.trend_slope.toFixed(4),
                        hint: "anomaly-score change per cycle",
                      },
                      {
                        icon: Sparkles,
                        label: "Rule support",
                        value: record.cause_likelihood.toFixed(2),
                        hint: "heuristic score, not fault probability",
                      },
                    ].map((c, i) => (
                      <motion.div
                        key={c.label}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.06 }}
                        className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-3"
                      >
                        <div className="flex items-center gap-1.5 text-xs text-slate-500">
                          <c.icon className="h-3.5 w-3.5" />
                          {c.label}
                        </div>
                        <div className="mt-0.5 font-semibold text-slate-100">{c.value}</div>
                        <div className="text-[10px] text-slate-600">{c.hint}</div>
                      </motion.div>
                    ))}
                  </div>
                )}

                {Array.isArray(focus.subsystem_attribution) &&
                  focus.subsystem_attribution.length > 0 && (
                    <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-4">
                      <div className="mb-3 text-sm font-semibold text-slate-300">
                        Where the deviation sits
                      </div>
                      <EngineAnatomy attribution={focus.subsystem_attribution} />
                    </div>
                  )}

                {(focus.why || focus.what_pattern || focus.how_severe) && (
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                    {[
                      ["Why", focus.why],
                      ["Pattern", focus.what_pattern],
                      ["Severity", focus.how_severe],
                    ].map(([label, text], i) =>
                      text ? (
                        <motion.div
                          key={String(label)}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.07 }}
                          className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-3"
                        >
                          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                            {String(label)}
                          </div>
                          <p className="text-sm leading-relaxed text-slate-300">{String(text)}</p>
                        </motion.div>
                      ) : null,
                    )}
                  </div>
                )}

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-4">
                  <FeedbackControl
                    engineId={focus.engine_id}
                    cycle={focus.cycle}
                    systemRootCause={focus.root_cause}
                  />
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-4">
                    <div className="mb-2 text-sm font-semibold text-slate-300">Root Cause</div>
                    <p className="text-slate-200">{focus.root_cause ?? "—"}</p>
                    {record && (
                      <p className="mt-1 text-xs text-slate-500">
                        {record.subsystem} &middot; rule support{" "}
                        {record.cause_likelihood.toFixed(2)} (heuristic)
                      </p>
                    )}
                    {focus.recommended_action && (
                      <p className="mt-3 text-sm">
                        <span className="text-slate-500">Recommended action: </span>
                        <span className="text-cyan-300">{focus.recommended_action}</span>
                      </p>
                    )}
                    {record?.inspection_focus && (
                      <p className="mt-1.5 text-sm">
                        <span className="text-slate-500">Inspection focus: </span>
                        <span className="text-amber-300">{record.inspection_focus}</span>
                      </p>
                    )}
                  </div>

                  <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-4">
                    <div className="mb-2 text-sm font-semibold text-slate-300">
                      Contributing Sensors <span className="text-xs font-normal text-slate-500">(ranked by deviation)</span>
                    </div>
                    {record && record.sensor_contributions.length > 0 ? (
                      <div className="space-y-2">
                        {record.sensor_contributions.slice(0, 6).map((c, i) => {
                          const z = Math.abs(c.residual_z ?? 0);
                          return (
                            <motion.div
                              key={c.sensor}
                              initial={{ opacity: 0, x: -8 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.06 }}
                            >
                              <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
                                <span className="font-mono text-slate-300">
                                  #{i + 1} {c.sensor}
                                  {c.subsystem && (
                                    <span className="ml-1.5 text-[10px] uppercase tracking-wide text-slate-600">
                                      {c.subsystem}
                                    </span>
                                  )}
                                </span>
                                <span
                                  className="font-mono"
                                  style={{ color: PALETTE[i % PALETTE.length] }}
                                >
                                  {z.toFixed(2)}
                                  <span className="text-slate-600">z</span>
                                </span>
                              </div>
                              <div className="h-1.5 overflow-hidden rounded-full bg-slate-800/70">
                                <motion.div
                                  className="h-full rounded-full"
                                  style={{ background: PALETTE[i % PALETTE.length] }}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.min((z / zMax) * 100, 100)}%` }}
                                  transition={{ duration: 0.7, delay: i * 0.06 }}
                                />
                              </div>
                            </motion.div>
                          );
                        })}
                        <p className="pt-1 text-[10px] text-slate-600">
                          Bars show each residual as a robust z-score, scaled to the largest
                          deviation in this alert.
                        </p>
                      </div>
                    ) : topSensors.length ? (
                      <div className="flex flex-wrap gap-2">
                        {topSensors.map((s, i) => (
                          <motion.span
                            key={s}
                            initial={{ opacity: 0, scale: 0.85 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: i * 0.07 }}
                            className="rounded-lg border px-2.5 py-1 font-mono text-xs"
                            style={{
                              borderColor: `${PALETTE[i % PALETTE.length]}55`,
                              background: `${PALETTE[i % PALETTE.length]}18`,
                              color: PALETTE[i % PALETTE.length],
                            }}
                          >
                            #{i + 1} {s}
                          </motion.span>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-slate-500">No sensor ranking for this alert.</p>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {[
                    ["Model Agreement", focus.model_agreement, "%"],
                    ["Detector Agreement", focus.detector_agreement, "%"],
                    ["Reliability", focus.reliability_score, ""],
                    ["Twin Spread", focus.twin_spread_mean, ""],
                  ].map(([label, val, unit], i) => (
                    <motion.div
                      key={String(label)}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.06 }}
                      className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-3"
                    >
                      <div className="text-xs text-slate-500">{String(label)}</div>
                      <div className="mt-0.5 font-semibold text-cyan-300">
                        {typeof val === "number"
                          ? unit === "%"
                            ? `${(val * 100).toFixed(1)}%`
                            : val.toFixed(4)
                          : "—"}
                      </div>
                    </motion.div>
                  ))}
                </div>

                {record && (
                  <div
                    className={`rounded-lg border p-4 ${
                      record.similar_past_anomaly_id
                        ? "border-cyan-500/25 bg-cyan-500/[0.05]"
                        : "border-slate-800/60 bg-slate-900/40"
                    }`}
                  >
                    <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-300">
                      <History className="h-4 w-4 text-cyan-400" />
                      Case memory
                    </div>
                    {record.similar_past_anomaly_id ? (
                      <>
                        <p className="text-sm leading-relaxed text-slate-300">
                          This pattern matches a remembered case at{" "}
                          <span className="font-semibold text-cyan-300">
                            {(record.similarity_score * 100).toFixed(1)}%
                          </span>{" "}
                          similarity, seen{" "}
                          <span className="font-semibold text-cyan-300">
                            {record.times_seen_before}
                          </span>{" "}
                          {record.times_seen_before === 1 ? "time" : "times"} before.
                        </p>
                        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                          <code className="rounded bg-slate-800/70 px-1.5 py-0.5 font-mono text-slate-400">
                            {record.similar_past_anomaly_id}
                          </code>
                          <span
                            className={`rounded px-1.5 py-0.5 ${
                              record.previously_false_alarm
                                ? "bg-amber-500/15 text-amber-300"
                                : "bg-emerald-500/10 text-emerald-400"
                            }`}
                          >
                            {record.previously_false_alarm
                              ? "an operator previously rejected this pattern"
                              : "never rejected by an operator"}
                          </span>
                        </div>
                      </>
                    ) : (
                      <p className="text-sm text-slate-400">
                        No close match in memory: this pattern is being seen for the first time, so
                        the conclusion rests on the rule evidence above rather than on precedent.
                      </p>
                    )}
                    <p className="mt-2 text-xs text-slate-500">
                      Matches feed the confidence estimate: a pattern an operator has already
                      rejected lowers it instead of raising an identical alert again.
                    </p>
                  </div>
                )}
              </motion.div>
            ) : (
              <Empty label="No explained alerts — no faults were detected in this run." />
            )}
          </AnimatePresence>
        </MetricCard>
      </Reveal>


      {/* ---- Does the reasoning layer hold up? Its own acceptance gates ------ */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Reveal index={7} className="xl:col-span-3">
          <MetricCard
            title="Reasoning & Explainability Validation"
            subtitle={
              rGate
                ? `Every one of ${(rGate.rows_reasoned ?? 0).toLocaleString()} fault cycles, checked offline`
                : "Offline acceptance gates"
            }
          >
            {!rGate && !xGate ? (
              <Empty label="Run the offline evaluation to populate this panel." />
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                  {[
                    {
                      label: "Coverage",
                      value:
                        rGate?.coverage === undefined
                          ? "—"
                          : `${(rGate.coverage * 100).toFixed(1)}%`,
                      hint: "fault cycles given a cause",
                      tone: "text-emerald-400",
                    },
                    {
                      label: "Narrative completeness",
                      value:
                        xGate?.completeness?.completeness_fraction === undefined
                          ? "—"
                          : `${(xGate.completeness.completeness_fraction * 100).toFixed(1)}%`,
                      hint: "all four fields present",
                      tone: "text-emerald-400",
                    },
                    {
                      label: "Consistency",
                      value:
                        xGate?.consistency?.consistency_fraction === undefined
                          ? "—"
                          : `${((xGate.consistency.consistency_fraction ?? 0) * 100).toFixed(1)}%`,
                      hint: "sensor lists match reasoning records",
                      tone: "text-emerald-400",
                    },
                    {
                      label: "Memory match rate",
                      value:
                        rGate?.memory_match_rate === undefined
                          ? "—"
                          : `${(rGate.memory_match_rate * 100).toFixed(1)}%`,
                      hint: `${(evaluation.reasoning?.memory_matches ?? 0).toLocaleString()} cases recognised`,
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

                {xGate?.fidelity && (
                  <p className="mt-4 text-xs leading-relaxed text-slate-400">
                    Local SHAP: {xGate.fidelity.shap_rows_computed ?? 0} alerts
                    ({((xGate.fidelity.shap_coverage_fraction ?? 0) * 100).toFixed(1)}% coverage);
                    {" "}{xGate.fidelity.shap_rows_failed ?? 0} failed.
                    Health reconstruction coverage: {((xGate.fidelity.health_coverage_fraction ?? 0) * 100).toFixed(1)}%.
                    These checks measure completeness and numeric fidelity; root causes still require inspection or expert validation.
                  </p>
                )}

                {/* No field is silently blank -- the per-field misses are shown. */}
                {xGate?.completeness?.missing_by_field && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {Object.entries(xGate.completeness.missing_by_field).map(([field, missing]) => (
                      <span
                        key={field}
                        className={`rounded-lg border px-2.5 py-1 text-xs ${
                          missing === 0
                            ? "border-emerald-500/25 bg-emerald-500/[0.07] text-emerald-300"
                            : "border-amber-500/25 bg-amber-500/[0.07] text-amber-300"
                        }`}
                      >
                        {missing === 0 ? "\u2713" : "\u26a0"} {field.replace(/_/g, " ")}
                        {missing > 0 && <span className="ml-1">{missing} missing</span>}
                      </span>
                    ))}
                  </div>
                )}

                <div className="mt-4 rounded-lg border border-emerald-500/25 bg-emerald-500/[0.06] p-4">
                  <div className="flex items-start gap-3">
                    <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
                    <div>
                      <p className="text-sm font-semibold text-emerald-300">
                        {rGate?.hard_causal_claims ?? 0} hard causal claims
                      </p>
                      <p className="mt-1 text-xs leading-relaxed text-slate-400">
                        The dependency graph encodes physical couplings between engine sections, not
                        demonstrated causation, so the reasoner is not permitted to assert that one
                        thing caused another. Every conclusion above is a ranked likelihood with its
                        evidence attached, and this counter is the check that none of them
                        overstepped across all{" "}
                        {(rGate?.rows_reasoned ?? 0).toLocaleString()} reasoned cycles.
                      </p>
                    </div>
                  </div>
                </div>

                {rGate?.acceptance && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {Object.entries(rGate.acceptance).map(([gate, passed]) => (
                      <span
                        key={gate}
                        className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs ${
                          passed
                            ? "border-emerald-500/25 bg-emerald-500/[0.07] text-emerald-300"
                            : "border-red-500/25 bg-red-500/[0.07] text-red-300"
                        }`}
                      >
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        {gate.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}
          </MetricCard>
        </Reveal>

        {/* ---- how deviations evolve, fleet-wide -------------------------- */}
        <Reveal index={8} className="xl:col-span-2">
          <MetricCard
            title="How deviations evolve"
            subtitle="Evolution pattern across every reasoned fault cycle"
          >
            {evolutionChart.length === 0 ? (
              <Empty label="Run the offline evaluation to populate this panel." />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={evolutionChart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(v: unknown) => [
                        `${(v as number).toLocaleString()} cycles`,
                        "count",
                      ]}
                    />
                    <Bar dataKey="value" radius={[6, 6, 0, 0]} animationDuration={900}>
                      {evolutionChart.map((d) => (
                        <Cell
                          key={d.name}
                          fill={
                            d.name === "abrupt"
                              ? "#ef4444"
                              : d.name === "gradual"
                              ? "#f59e0b"
                              : "#22d3ee"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-2 space-y-1">
                  {evolutionChart.map((d) => (
                    <div key={d.name} className="flex justify-between text-xs">
                      <span className="capitalize text-slate-400">{d.name}</span>
                      <span className="font-mono text-slate-500">
                        {((d.value / evolutionTotal) * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  Degradation is overwhelmingly gradual or stable; genuinely abrupt shifts are rare.
                  That is what makes early warning possible at all — a fault that only ever appeared
                  abruptly would leave no window to act in.
                </p>
              </>
            )}
          </MetricCard>
        </Reveal>
      </div>


      {focus && (
        <Reveal index={9}>
          <MetricCard
            title="Why the monitoring twin expected this reading"
            subtitle={`Local SHAP for ${focus.shap_sensor ?? "the selected sensor"}, engine ${focus.engine_id}, cycle ${focus.cycle}`}
          >
            {focus.shap_status === "computed" && shapRanking.rows.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={210}>
                  <BarChart data={shapRanking.rows} layout="vertical" margin={{ left: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" stroke="#64748b" fontSize={11} />
                    <YAxis dataKey="feature" type="category" stroke="#64748b" width={140} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
                      formatter={(v: unknown) => [(v as number).toFixed(5), "signed SHAP"]}
                    />
                    <Bar dataKey="value" animationDuration={900}>
                      {shapRanking.rows.map((row, i) => (
                        <Cell key={i} fill={row.value >= 0 ? "#22d3ee" : "#fbbf24"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <p className="mt-2 text-xs leading-relaxed text-slate-400">
                  Baseline {focus.shap_base_value?.toFixed(4)} + signed contributions = expected reading {focus.shap_prediction?.toFixed(4)}.
                  The baseline includes the fixed commissioning offset ({focus.shap_commissioning_offset?.toFixed(4)}).
                  Positive values raise the expected reading; negative values lower it. Values use mode-scaled sensor units.
                </p>
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  Operating settings and mode are explained together; cycle age is a separate group.
                  This explains the ensemble's expected sensor value. The residual evidence explains the observed deviation;
                  these SHAP values do not establish a physical fault cause.
                </p>
              </>
            ) : (
              <p className="text-sm text-slate-400">
                {focus.shap_status === "not_sampled"
                  ? "This alert was outside the local SHAP sample. Numeric residual and health evidence remain available."
                  : focus.shap_status === "disabled"
                    ? "SHAP was disabled for this run."
                    : "Local SHAP is unavailable for this alert. Check the generation report for failure details."}
              </p>
            )}
            {focus.health_explanation && (
              <div className="mt-4 border-t border-slate-800 pt-4">
                <p className="mb-1 text-sm font-semibold text-slate-300">Health-index evidence</p>
                <p className="text-sm leading-relaxed text-slate-400">{focus.health_explanation}</p>
              </div>
            )}
          </MetricCard>
        </Reveal>
      )}

      <Reveal index={10}>
        <MetricCard
          title={
            faultyNode
              ? `Fault Propagation — origin: ${faultyNode.toUpperCase()}`
              : "Subsystem Dependency Graph"
          }
          subtitle={`${nodes} nodes · ${edges} physical couplings between engine sections`}
        >
          <SensorGraph nodes={graphNodes} edges={graphEdges} faultySubsystem={faultyNode} />
        </MetricCard>
      </Reveal>

      {explained.length > 0 && (
        <Reveal index={11}>
          <MetricCard title="Recent Explained Alerts">
            <div className="space-y-2">
              {explained.slice(0, 8).map((a, i) => (
                <motion.button
                  key={`${a.engine_id}-${a.cycle}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => setSelected(String(a.engine_id))}
                  className="w-full rounded-lg border border-slate-800/60 bg-slate-900/40 p-3 text-left transition-colors hover:border-purple-500/40"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-slate-200">
                      E{String(a.engine_id)} · cycle {a.cycle}
                    </span>
                    <span
                      className={`rounded border px-2 py-0.5 text-xs ${
                        a.severity === "critical"
                          ? "border-red-500/30 bg-red-500/10 text-red-300"
                          : a.severity === "high"
                          ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                          : "border-slate-600/30 bg-slate-700/20 text-slate-300"
                      }`}
                    >
                      {a.severity ?? "—"}
                    </span>
                  </div>
                  <div className="mt-1 text-sm text-slate-400">{a.root_cause ?? "—"}</div>
                  <div className="mt-1 line-clamp-2 text-xs text-slate-500">{a.explanation_text}</div>
                </motion.button>
              ))}
            </div>
          </MetricCard>
        </Reveal>
      )}
    </div>
  );
}
