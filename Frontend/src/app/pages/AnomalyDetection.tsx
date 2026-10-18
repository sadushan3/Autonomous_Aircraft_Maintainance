import { useState } from "react";
import { Activity, Radar, ShieldAlert, SignalHigh, Waves } from "lucide-react";
import { motion } from "motion/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar as RadarShape,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DataTable } from "../components/DataTable";
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
  useDetectorConfig,
} from "../../lib/hooks";
import { useEngineParam } from "../../lib/engine-param";
import { FeedbackControl } from "../components/FeedbackControl";
import { DetectionPerformancePanel } from "../components/AnomalyValidationPanels";
import type { AnomalyAlert } from "../../lib/types";


const DETECTORS = [
  { configKey: "isolation_forest", key: "isolation_forest_score", label: "Isolation Forest", color: "#22d3ee" },
  { configKey: "mahalanobis", key: "mahalanobis_score", label: "Mahalanobis", color: "#a855f7" },
  { configKey: "lstm_autoencoder", key: "lstm_ae_score", label: "LSTM Autoencoder", color: "#f59e0b" },
  { configKey: "residual_rule", key: "residual_rule_score", label: "Residual Rule", color: "#10b981" },
] as const;


const SEV_ORDER = ["none", "low", "medium", "high", "critical"] as const;

const SEV_COLOR: Record<string, string> = {
  critical: "#ef4444",
  high: "#f97316",
  medium: "#f59e0b",
  low: "#22d3ee",
  none: "#64748b",
};

export function AnomalyDetection() {
  const dash = useAnomalyDashboard();
  const detectorCfg = useDetectorConfig();
  const evaluation = useAnomalyEvaluation();
  const [selected, setSelected] = useEngineParam();


  const weightOf: Record<string, number | undefined> = {};
  const principleOf: Record<string, string | undefined> = {};
  const activeOf: Record<string, boolean> = {};
  for (const d of DETECTORS) {
    const cfg = detectorCfg.data?.detectors.find((x) => x.key === d.configKey);
    weightOf[d.key] = cfg?.weight;
    principleOf[d.key] = cfg?.principle ?? undefined;
    activeOf[d.key] = cfg?.enabled === true && cfg.trained === true;
  }

  const alerts: AnomalyAlert[] = dash.data?.alerts ?? [];
  const focus = alerts.find((a) => String(a.engine_id) === selected) ?? alerts[0];
  const focusId = String(focus?.engine_id ?? "");

  if (dash.isLoading) return <Loading label="Loading anomaly detection…" />;
  if (dash.error) return <ErrorState error={dash.error} />;
  if (!dash.data) return <Empty />;

  const fleet = dash.data.fleet;
  const engines = dash.data.engines;


  const sevCounts: Record<string, number> = {};
  for (const e of engines) {
    const k = String(e.severity ?? "none");
    sevCounts[k] = (sevCounts[k] ?? 0) + 1;
  }
  const sevChart = Object.entries(sevCounts)
    .map(([name, value]) => ({ name, value, color: SEV_COLOR[name] ?? "#64748b" }))
    .sort((a, b) => b.value - a.value);


  const detectorMeans = DETECTORS.map((d) => {
    const vals = alerts
      .map((a) => a[d.key] as number | null)
      .filter((n): n is number => typeof n === "number" && Number.isFinite(n));
    const mean = vals.length ? vals.reduce((x, y) => x + y, 0) / vals.length : null;
    return {
      name: d.label,
      score: mean === null ? null : Number(mean.toFixed(4)),
      color: d.color,
      weight: weightOf[d.key],
    };
  });

  const focusRadar = focus
    ? DETECTORS.filter((d) => typeof focus[d.key] === "number" && Number.isFinite(focus[d.key])).map((d) => ({
        detector: d.label.split(" ")[0],
        score: Number((((focus[d.key] as number) ?? 0) * 100).toFixed(1)),
      }))
    : [];

  const threshold = focus?.threshold_used ?? detectorCfg.data?.fusion.calibrated_threshold ?? null;


  const severityRul: Array<{ name: string; rul: number; color: string }> = [];
  for (const band of SEV_ORDER) {
    const rul = evaluation.test?.severity_mean_rul?.[band];
    if (typeof rul === "number" && Number.isFinite(rul)) {
      severityRul.push({ name: band, rul, color: SEV_COLOR[band] ?? "#64748b" });
    }
  }

  const monotone = severityRul.length >= 2 && severityRul.every(
    (d, i) => i === 0 || d.rul <= severityRul[i - 1].rul,
  );

  const gate = evaluation.acceptance?.gate_behaviour;
  const rulAlign = evaluation.acceptance?.rul_alignment;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Anomaly Detection"
        subtitle={`${detectorCfg.data?.fusion.active_detectors ?? 0} active detector models, fused against a calibrated threshold`}
        right={<LiveBadge />}
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <Reveal index={0}>
          <KPICard title="Total Anomalies" value={<AnimatedNumber value={fleet.total_anomalies} />} icon={Activity} status="info" />
        </Reveal>
        <Reveal index={1}>
          <KPICard title="Genuine Faults" value={<AnimatedNumber value={fleet.fault_anomalies} />} icon={ShieldAlert} status="danger" />
        </Reveal>
        <Reveal index={2}>
          <KPICard
            title="Operational (gated)"
            value={<AnimatedNumber value={fleet.operational_anomalies ?? 0} />}
            icon={Waves}
            status="warning"
          />
        </Reveal>
        <Reveal index={3}>
          <KPICard
            title="Fusion Threshold"
            value={threshold === null ? "—" : <AnimatedNumber value={threshold} decimals={4} />}
            icon={SignalHigh}
            status="neutral"
            subtitle="healthy p99, calibrated"
          />
        </Reveal>
        <Reveal index={4}>
          <KPICard
            title="Critical Engines"
            value={<AnimatedNumber value={sevCounts["critical"] ?? 0} />}
            icon={Radar}
            status="danger"
          />
        </Reveal>
      </div>


      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Reveal index={5} className="xl:col-span-3">
          <DetectionPerformancePanel />
        </Reveal>

        <Reveal index={6} className="xl:col-span-2">
          <MetricCard
            title="Do the severity bands mean anything?"
            subtitle="Mean remaining life behind each band, held-out engines"
          >
            {severityRul.length === 0 ? (
              <Empty label={evaluation.isError
                ? "Could not load validation results. Check the backend connection."
                : evaluation.test
                  ? "The saved evaluation report does not include mean RUL by severity band."
                  : "No held-out severity results are available in the saved evaluation report."} />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={severityRul}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis
                      stroke="#64748b"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: "mean RUL",
                        angle: -90,
                        position: "insideLeft",
                        fill: "#475569",
                        fontSize: 11,
                      }}
                    />
                    <Tooltip
                      contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                      formatter={(v: unknown) => [`${(v as number).toFixed(1)} cycles`, "mean RUL"]}
                    />
                    <Bar dataKey="rul" radius={[6, 6, 0, 0]} animationDuration={900}>
                      {severityRul.map((d) => (
                        <Cell key={d.name} fill={d.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                {evaluation.test?.severity_row_counts && (
                  <p className="mt-2 text-xs text-slate-500">
                    Evaluated cycles: {severityRul.map((d) =>
                      `${d.name} ${evaluation.test?.severity_row_counts?.[d.name]?.toLocaleString() ?? "—"}`
                    ).join(" · ")}
                  </p>
                )}
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  {monotone ? (
                    <>
                      Mean remaining life decreases across the observed bands, from{" "}
                      <span className="text-emerald-300">
                        {severityRul[0].rul.toFixed(0)} cycles
                      </span>{" "}
                      at <em>{severityRul[0].name}</em> down to{" "}
                      <span className="text-red-300">
                        {severityRul[severityRul.length - 1].rul.toFixed(0)} cycles
                      </span>{" "}
                      at <em>{severityRul[severityRul.length - 1].name}</em>.
                    </>
                  ) : (
                    <>
                      The bands do not order monotonically by remaining life on held-out engines, so severity should
                      be read as a score band rather than a life estimate.
                    </>
                  )}
                </p>
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  {evaluation.test?.severity_summary_note
                    ?? "Mean RUL by severity is a descriptive association, rather than a calibrated remaining-life estimate."}
                </p>
              </>
            )}
          </MetricCard>
        </Reveal>
      </div>


      {(gate || rulAlign) && (
        <Reveal index={7}>
          <GlassCard>
            <h3 className="mb-3 font-semibold text-slate-200">Fault vs. operational gating</h3>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {[
                {
                  label: "Raw flags",
                  value: gate?.total_flagged?.toLocaleString() ?? "—",
                  hint: "cycles over the fused threshold",
                  tone: "text-slate-200",
                },
                {
                  label: "Kept as faults",
                  value: gate?.fault_anomalies?.toLocaleString() ?? "—",
                  hint: "genuine degradation",
                  tone: "text-red-400",
                },
                {
                  label: "Reclassified operational",
                  value:
                    gate?.operational_suppression_rate === undefined
                      ? "—"
                      : `${(gate.operational_suppression_rate * 100).toFixed(1)}%`,
                  hint: `${gate?.operational_anomalies?.toLocaleString() ?? "—"} cycles explained by a regime change`,
                  tone: "text-amber-400",
                },
                {
                  label: "Mean RUL at a fault flag",
                  value:
                    rulAlign?.fault_mean_rul === undefined
                      ? "—"
                      : `${rulAlign.fault_mean_rul.toFixed(1)}`,
                  hint: `vs ${rulAlign?.fleet_mean_rul?.toFixed(1) ?? "—"} across the fleet`,
                  tone: "text-emerald-400",
                },
              ].map((c) => (
                <div key={c.label} className="rounded-lg border border-slate-800/70 bg-slate-950/40 p-3">
                  <p className="text-xs text-slate-500">{c.label}</p>
                  <p className={`mt-0.5 text-xl font-semibold ${c.tone}`}>{c.value}</p>
                  <p className="text-[10px] text-slate-600">{c.hint}</p>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs leading-relaxed text-slate-500">
              {(100 * (gate?.operational_suppression_rate ?? 0)).toFixed(1)}% of raw flags are classified as operational
              excursions around regime changes. The remaining sustained flags are assessed as faults. The last
              tile is the sanity check on what survives: flagged faults sit at a mean{" "}
              {rulAlign?.fault_mean_rul?.toFixed(0) ?? "—"} cycles of remaining life against a fleet mean of{" "}
              {rulAlign?.fleet_mean_rul?.toFixed(0) ?? "—"} — they cluster near end of life, where faults belong.
            </p>
          </GlassCard>
        </Reveal>
      )}

      <Reveal index={8}>
        <GlassCard variant="primary">
          <h3 className="mb-3 font-semibold text-slate-200">Detector Ensemble</h3>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
            {DETECTORS.map((d, i) => (
              <motion.div
                key={d.key}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className="rounded-lg border p-3"
                style={{ borderColor: `${d.color}44`, background: `${d.color}12` }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold" style={{ color: d.color }}>
                    {d.label}
                  </span>
                  <span className="rounded bg-slate-900/60 px-1.5 py-0.5 text-xs text-slate-400">
                    {weightOf[d.key] === undefined ? "w=—" : `w=${weightOf[d.key]}`}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-500">{principleOf[d.key] ?? "—"}</p>
                <p className={`mt-2 text-xs ${activeOf[d.key] ? "text-emerald-400" : "text-amber-400"}`}>
                  {activeOf[d.key] ? "Active · trained model" : "Unavailable or disabled"}
                </p>
              </motion.div>
            ))}
          </div>
          <p className="mt-3 text-xs text-slate-500">
            {detectorCfg.data?.fusion.normalisation ?? "Loading the saved detector calibration."}
          </p>
        </GlassCard>
      </Reveal>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Reveal index={9}>
          <MetricCard title="Mean Detector Score Across Alerts">
            {detectorMeans.some((d) => d.score !== null) ? <ResponsiveContainer width="100%" height={280}>
              <BarChart data={detectorMeans} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" domain={[0, 1]} stroke="#64748b" />
                <YAxis dataKey="name" type="category" stroke="#64748b" width={130} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                <Bar dataKey="score" radius={[0, 6, 6, 0]} animationDuration={900}>
                  {detectorMeans.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer> : <Empty label="No detector scores are available in the current alerts." />}
          </MetricCard>
        </Reveal>

        <Reveal index={10}>
          <MetricCard title="Severity Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={sevChart}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]} animationDuration={900}>
                  {sevChart.map((d) => (
                    <Cell key={d.name} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-2 text-xs text-slate-500">
              Severity bands are fractions of the distance from the calibrated threshold to 1.0, so they move with the
              threshold instead of being fixed cuts.
            </p>
          </MetricCard>
        </Reveal>
      </div>

      <Reveal index={11}>
        <MetricCard title={`Engine E${focusId} — Detector Breakdown`}>
          <div className="mb-4 flex flex-wrap gap-2">
            {alerts.slice(0, 18).map((a) => (
              <button
                key={`${a.engine_id}-${a.cycle}`}
                onClick={() => setSelected(String(a.engine_id))}
                className={`rounded-lg border px-3 py-1 text-xs transition-all hover:scale-105 ${
                  focusId === String(a.engine_id)
                    ? "border-cyan-500/50 bg-cyan-500/20 text-cyan-300"
                    : "border-slate-700/50 bg-slate-800/40 text-slate-400"
                }`}
              >
                E{String(a.engine_id)}
              </button>
            ))}
          </div>

          {focus ? (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={focusRadar}>
                  <PolarGrid stroke="#1e293b" />
                  <PolarAngleAxis dataKey="detector" stroke="#64748b" />
                  <PolarRadiusAxis stroke="#334155" domain={[0, 100]} />
                  <RadarShape
                    name="Detector score (%)"
                    dataKey="score"
                    stroke="#22d3ee"
                    fill="#22d3ee"
                    fillOpacity={0.35}
                    animationDuration={900}
                  />
                  <Legend />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                </RadarChart>
              </ResponsiveContainer>

              <div className="space-y-3">
                {DETECTORS.map((d, i) => {
                  const reading = focus[d.key];
                  const v = typeof reading === "number" && Number.isFinite(reading) ? reading : null;
                  return (
                    <motion.div
                      key={d.key}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.07 }}
                    >
                      <div className="mb-1 flex justify-between text-xs">
                        <span className="text-slate-400">{d.label}</span>
                        <span className="font-mono" style={{ color: d.color }}>
                          {v === null ? "—" : v.toFixed(4)}
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ background: d.color }}
                          initial={{ width: 0 }}
                          animate={{ width: `${v === null ? 0 : Math.min(v * 100, 100)}%` }}
                          transition={{ duration: 0.8, delay: i * 0.07 }}
                        />
                      </div>
                    </motion.div>
                  );
                })}

                <div className="mt-4 grid grid-cols-2 gap-3">
                  {[
                    ["Fused Score", focus.anomaly_score, 4],
                    ["Threshold", focus.threshold_used, 4],
                    ["Model Agreement", focus.model_agreement, 3],
                    ["Detector Agreement", focus.detector_agreement, 3],
                    ["Uncertainty", focus.uncertainty_score, 3],
                    ["Reliability", focus.reliability_score, 3],
                    ["Persistence (cycles)", focus.anomaly_persistence_cycles, 0],
                    ["Early Warning", focus.early_warning_score, 2],
                  ].map(([label, val, dp], i) => (
                    <motion.div
                      key={String(label)}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-2.5"
                    >
                      <div className="text-xs text-slate-500">{String(label)}</div>
                      <div className="mt-0.5 font-semibold text-cyan-300">
                        {typeof val === "number" ? val.toFixed(Number(dp)) : "—"}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <Empty label="No alerts to break down." />
          )}

          {focus && (
            <div className="mt-5 rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-4">
              <FeedbackControl
                engineId={focus.engine_id}
                cycle={focus.cycle}
                systemRootCause={focus.root_cause}
              />
            </div>
          )}
        </MetricCard>
      </Reveal>

      <Reveal index={12}>
        <MetricCard title="Anomaly Feed">
          <DataTable
            columns={[
              { header: "Engine", accessor: "id" },
              { header: "Cycle", accessor: "cycle" },
              { header: "Severity", accessor: "sev" },
              { header: "Fused", accessor: "fused" },
              { header: "IF", accessor: "iso" },
              { header: "Mahal.", accessor: "mah" },
              { header: "LSTM-AE", accessor: "lstm" },
              { header: "Rule", accessor: "rule" },
              { header: "Persist", accessor: "persist" },
              { header: "Agreement", accessor: "agree" },
            ]}
            data={alerts.slice(0, 20).map((a) => ({
              id: `E${a.engine_id}`,
              cycle: a.cycle,
              sev: a.severity ?? "—",
              fused: (a.anomaly_score ?? 0).toFixed(4),
              iso: typeof a.isolation_forest_score === "number" ? a.isolation_forest_score.toFixed(3) : "—",
              mah: typeof a.mahalanobis_score === "number" ? a.mahalanobis_score.toFixed(3) : "—",
              lstm: typeof a.lstm_ae_score === "number" ? a.lstm_ae_score.toFixed(3) : "—",
              rule: typeof a.residual_rule_score === "number" ? a.residual_rule_score.toFixed(3) : "—",
              persist: a.anomaly_persistence_cycles ?? "—",
              agree: typeof a.detector_agreement === "number" ? `${(a.detector_agreement * 100).toFixed(1)}%` : "—",
            }))}
          />
        </MetricCard>
      </Reveal>
    </div>
  );
}
