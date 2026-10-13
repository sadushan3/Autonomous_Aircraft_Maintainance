/**
 * Anomaly & Health -- validation, drift, triage and feedback panels.
 *
 * Split out of `HealthMonitoring.tsx` to keep that page readable. Every figure
 * is read from a backend response; nothing is computed in the browser beyond
 * grouping and formatting, and nothing is hardcoded.
 *
 * All four panels belong to the Anomaly & Health component. None of them read
 * `risk_10/30/50`, `risk_state` or any other Failure Risk field -- the two
 * modules stay independent.
 */
import { useMemo, useState } from "react";
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
import { AlertTriangle, CheckCircle2, Radar, ThumbsDown, ThumbsUp, Waves } from "lucide-react";
import { MetricCard } from "./MetricCard";
import { Empty } from "./Live";
import {
  useAnomalyEvaluation,
  useContextStatus,
  useFeedbackSummary,
  useFeedbackThresholds,
  useSubmitFeedback,
} from "../../lib/hooks";
import type { AnomalyAlert, ContextDrift } from "../../lib/types";
import { criticalEngineDetection } from "../../lib/anomalyEvaluation";

const DASH = "—";
const pct = (v: unknown, d = 1) =>
  typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : DASH;
const num = (v: unknown, d = 3) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : DASH;

const ALERT_COLOR: Record<string, string> = {
  critical: "#ef4444",
  high: "#f97316",
  warning: "#f59e0b",
  low: "#10b981",
};

/* ------------------------------------------------------------------ */
/* 1. Detection performance + lead time                                */
/* ------------------------------------------------------------------ */

export function DetectionPerformancePanel() {
  const ev = useAnomalyEvaluation();
  const { test, train, perSubset, health } = ev;

  if (ev.isLoading) return <MetricCard title="Detection Performance"><Empty label="Loading…" /></MetricCard>;
  if (!ev.available || !test) {
    return (
      <MetricCard
        title="Detection Performance"
        subtitle="Held-out validation"
      >
        <Empty label={ev.isError ? "Could not load validation results. Check the backend connection." : "No held-out detection results are available in the saved evaluation report."} />
      </MetricCard>
    );
  }

  const subsetChart = Object.entries(perSubset ?? {}).map(([fd, m]) => ({
    fd,
    auc: Number((m.auc * 100).toFixed(2)),
    falseAlarm: Number((m.false_alarm * 100).toFixed(2)),
  }));

  return (
    <MetricCard
      title="Detection Performance"
      subtitle="Saved held-out benchmark · RUL condition proxies"
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["AUC", pct(test.auc_healthy_vs_critical, 2), "healthy vs critical", "text-emerald-400"],
          ["False alarms", pct(test.false_alarm_rate_healthy, 2), "on healthy rows", "text-emerald-400"],
          ["Detection rate", pct(criticalEngineDetection(test), 1), "of engines reaching critical RUL", "text-amber-400"],
          ["Separation", num(test.cohens_d, 2), "Cohen's d", "text-sky-400"],
        ].map(([label, value, hint, tone]) => (
          <div key={label as string} className="rounded-lg border border-slate-800/70 bg-slate-950/40 p-3">
            <p className="text-xs text-slate-500">{label}</p>
            <p className={`mt-0.5 text-xl font-semibold ${tone}`}>{value}</p>
            <p className="text-[10px] text-slate-600">{hint}</p>
          </div>
        ))}
      </div>

      {/* lead time -- the headline metric for predictive maintenance */}
      <div className="mt-4 rounded-lg border border-slate-800/70 bg-slate-950/40 p-4">
        <p className="text-xs uppercase tracking-widest text-slate-500">Early-warning lead time</p>
        <p className="mt-1 text-sm text-slate-400">
          The first alert occurs at a median remaining life of{" "}
          <span className="text-2xl font-semibold text-sky-400">{test.lead_time_median ?? DASH}</span>{" "}
          cycles.
        </p>
        <p className="mt-2 text-xs text-slate-500">
          {test.condition_proxy_metrics?.engine_events?.lead_time_note
            ?? "Remaining life at the first alert can include false alerts; it does not establish correct fault onset."}
        </p>
        {typeof test.lead_time_p25 === "number" && typeof test.lead_time_p75 === "number" && (
          <>
            <div className="relative mt-3 h-2 rounded-full bg-slate-800">
              <div
                className="absolute h-2 rounded-full bg-sky-500/40"
                style={{
                  left: `${Math.min(95, (test.lead_time_p25 / 200) * 100)}%`,
                  width: `${Math.min(
                    100 - (test.lead_time_p25 / 200) * 100,
                    ((test.lead_time_p75 - test.lead_time_p25) / 200) * 100,
                  )}%`,
                }}
              />
              <div
                className="absolute -top-0.5 h-3 w-1 rounded-full bg-sky-400"
                style={{ left: `${Math.min(98, ((test.lead_time_median ?? 0) / 200) * 100)}%` }}
              />
            </div>
            <p className="mt-1.5 text-xs text-slate-500">
              Interquartile range {test.lead_time_p25}&ndash;{test.lead_time_p75} cycles · axis 0&ndash;200
            </p>
          </>
        )}
      </div>

      {subsetChart.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-xs uppercase tracking-widest text-slate-500">Per subset</p>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={subsetChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="fd" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} unit="%" />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
              />
              <Bar name="AUC %" dataKey="auc" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar name="False alarm %" dataKey="falseAlarm" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <p className="mt-3 text-xs text-slate-500">
        {test.label_basis ?? "Healthy and critical conditions are defined using RUL bands, rather than independently confirmed fault labels."}
      </p>
      {/* Compare only rates with the same critical-engine denominator. */}
      {train && criticalEngineDetection(train) != null && criticalEngineDetection(test) != null && (
        <p className="mt-3 border-t border-slate-800/70 pt-3 text-xs leading-relaxed text-slate-500">
          Critical-engine detection: {" "}
          <span className="text-slate-300">{pct(criticalEngineDetection(train), 0)}</span> on training engines and{" "}
          <span className="text-slate-300">{pct(criticalEngineDetection(test), 0)}</span> held out. Held-out AUC is{" "}
          <span className="text-slate-300">{pct(test.auc_healthy_vs_critical, 1)}</span> — the ranking
          generalises, the operating point is conservative.
          {typeof health?.fleet_ranking_spearman === "number" && (
            <>
              {" "}
              Health-index fleet ranking ρ = {num(health.fleet_ranking_spearman, 3)}, monotonicity{" "}
              {num(health.monotonicity_score, 3)}.
            </>
          )}
        </p>
      )}
    </MetricCard>
  );
}

/* ------------------------------------------------------------------ */
/* 2. Operating-context drift                                          */
/* ------------------------------------------------------------------ */

export function ContextDriftPanel() {
  const ctx = useContextStatus();
  const drift = (ctx.data as { drift?: ContextDrift } | undefined)?.drift;

  if (ctx.isLoading) return <MetricCard title="Operating-Context Drift"><Empty label="Loading…" /></MetricCard>;
  if (!drift) {
    return (
      <MetricCard title="Operating-Context Drift">
        <Empty label="No context reference persisted yet." />
      </MetricCard>
    );
  }

  const modes = Array.from(
    new Set([
      ...Object.keys(drift.reference_proportions ?? {}),
      ...Object.keys(drift.current_proportions ?? {}),
    ]),
  ).sort();

  const chart = modes.map((m) => ({
    mode: `Mode ${m}`,
    reference: Number((((drift.reference_proportions ?? {})[m] ?? 0) * 100).toFixed(2)),
    current: Number((((drift.current_proportions ?? {})[m] ?? 0) * 100).toFixed(2)),
  }));

  const bad = drift.drift_detected;

  return (
    <MetricCard
      title="Operating-Context Drift"
      subtitle="Population Stability Index between the persisted reference regime mix and the current upload"
      actions={
        <span
          className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium ${
            bad ? "bg-red-500/15 text-red-400" : "bg-emerald-500/15 text-emerald-400"
          }`}
        >
          {bad ? <AlertTriangle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
          {bad ? `Drift — ${drift.severity}` : "Stable"}
        </span>
      }
    >
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-slate-800/70 bg-slate-950/40 p-3">
          <p className="text-xs text-slate-500">PSI</p>
          <p className={`text-xl font-semibold ${bad ? "text-red-400" : "text-emerald-400"}`}>
            {num(drift.psi, 3)}
          </p>
        </div>
        <div className="rounded-lg border border-slate-800/70 bg-slate-950/40 p-3">
          <p className="text-xs text-slate-500">Threshold</p>
          <p className="text-xl font-semibold text-slate-200">{num(drift.threshold, 2)}</p>
        </div>
        <div className="rounded-lg border border-slate-800/70 bg-slate-950/40 p-3">
          <p className="text-xs text-slate-500">Severity</p>
          <p className="text-xl font-semibold capitalize text-slate-200">{drift.severity ?? DASH}</p>
        </div>
      </div>

      <p className="mb-2 mt-4 text-xs uppercase tracking-widest text-slate-500">
        Regime mix · reference vs current
      </p>
      <ResponsiveContainer width="100%" height={170}>
        <BarChart data={chart}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis dataKey="mode" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} unit="%" />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
          />
          <Bar name="Reference" dataKey="reference" fill="#64748b" radius={[4, 4, 0, 0]} />
          <Bar name="Current" dataKey="current" fill="#38bdf8" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      {bad && (
        <p className="mt-3 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-300">
          <Waves className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          Residual thresholds were calibrated on the reference regime mix and may no longer hold.
          The current upload concentrates in a single operating mode, so anomaly scores are being
          judged against a distribution the twin did not see. Refitting the twin is advisable before
          trusting flag rates from this batch.
        </p>
      )}
    </MetricCard>
  );
}

/* ------------------------------------------------------------------ */
/* 3. Maintenance action queue                                         */
/* ------------------------------------------------------------------ */

const LEVEL_ORDER = ["critical", "high", "warning", "low"];

export function MaintenanceQueuePanel({ alerts }: { alerts: AnomalyAlert[] }) {
  const [level, setLevel] = useState<string>("all");

  const counts = useMemo(() => {
    const out: Record<string, number> = {};
    for (const a of alerts) {
      const k = String(a.alert_level ?? "unknown").toLowerCase();
      out[k] = (out[k] ?? 0) + 1;
    }
    return out;
  }, [alerts]);

  const rows = useMemo(() => {
    const filtered =
      level === "all"
        ? alerts
        : alerts.filter((a) => String(a.alert_level ?? "").toLowerCase() === level);
    return [...filtered]
      .sort((a, b) => {
        const ai = LEVEL_ORDER.indexOf(String(a.alert_level ?? "").toLowerCase());
        const bi = LEVEL_ORDER.indexOf(String(b.alert_level ?? "").toLowerCase());
        if (ai !== bi) return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
        return Number(b.anomaly_score ?? 0) - Number(a.anomaly_score ?? 0);
      })
      .slice(0, 12);
  }, [alerts, level]);

  return (
    <MetricCard
      title="Maintenance Action Queue"
      subtitle={`${alerts.length} open alerts, most urgent first`}
      actions={
        <div className="flex gap-1">
          {["all", ...LEVEL_ORDER.filter((l) => counts[l])].map((l) => (
            <button
              key={l}
              onClick={() => setLevel(l)}
              className={`rounded-md px-2 py-0.5 text-xs capitalize transition-colors ${
                level === l ? "bg-sky-600 text-white" : "bg-slate-800/60 text-slate-400 hover:text-slate-200"
              }`}
            >
              {l}
              {l !== "all" && counts[l] ? ` ${counts[l]}` : ""}
            </button>
          ))}
        </div>
      }
    >
      {rows.length ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wider text-slate-500">
                <th className="py-2 pr-3">Engine</th>
                <th className="py-2 pr-3">Level</th>
                <th className="py-2 pr-3">Score</th>
                <th className="py-2 pr-3">Persist.</th>
                <th className="py-2">Recommended action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {rows.map((a, i) => {
                const lvl = String(a.alert_level ?? "").toLowerCase();
                return (
                  <tr key={`${a.engine_id}-${i}`} className="hover:bg-slate-800/30">
                    <td className="py-2 pr-3 font-medium text-slate-200">E{String(a.engine_id)}</td>
                    <td className="py-2 pr-3">
                      <span
                        className="inline-flex items-center gap-1.5 text-xs capitalize"
                        style={{ color: ALERT_COLOR[lvl] ?? "#94a3b8" }}
                      >
                        <span
                          className="h-1.5 w-1.5 rounded-full"
                          style={{ background: ALERT_COLOR[lvl] ?? "#94a3b8" }}
                        />
                        {lvl || DASH}
                      </span>
                    </td>
                    <td className="py-2 pr-3 text-slate-300">{num(a.anomaly_score, 3)}</td>
                    <td className="py-2 pr-3 text-slate-400">
                      {a.anomaly_persistence_cycles ?? DASH}
                    </td>
                    <td className="py-2 text-slate-300">
                      {String(a.recommended_action ?? DASH)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty label="No alerts at this level." />
      )}
      <p className="mt-3 text-xs text-slate-500">
        Persistence is the number of consecutive cycles the engine has stayed flagged &mdash; a
        sustained count separates a real fault from a transient spike.
      </p>
    </MetricCard>
  );
}

/* ------------------------------------------------------------------ */
/* 4. Human-in-the-loop feedback                                       */
/* ------------------------------------------------------------------ */

export function FeedbackLoopPanel({
  engineId,
  cycle,
}: {
  engineId?: string;
  cycle?: number;
}) {
  const summary = useFeedbackSummary();
  const thresholds = useFeedbackThresholds();
  const submit = useSubmitFeedback();
  const [note, setNote] = useState("");

  const s = summary.data as { feedback?: { total?: number; by_verdict?: Record<string, number> }; memory?: Record<string, number> } | undefined;
  const t = thresholds.data as Record<string, unknown> | undefined;

  // The backend's verdict vocabulary is accepted | false_alarm | missed |
  // unknown -- "accepted" means the operator confirms the raised alert was a
  // genuine fault. Both engine_id and cycle are required, so the panel is
  // inert until a focus engine with a known latest cycle is selected.
  const numericId = Number(engineId);
  const canSubmit = Number.isFinite(numericId) && typeof cycle === "number";

  const send = (verdict: "accepted" | "false_alarm") => {
    if (!canSubmit) return;
    submit.mutate({
      engine_id: numericId,
      cycle: cycle as number,
      verdict,
      comment: note.trim() || undefined,
    });
    setNote("");
  };

  return (
    <MetricCard
      title="Analyst Feedback Loop"
      subtitle="Operator verdicts adapt the fusion threshold — closed-loop, persisted server-side"
      actions={<Radar className="h-4 w-4 text-slate-500" />}
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["Base threshold", num(t?.base_threshold, 4), "calibrated, frozen"],
          ["Live threshold", num(t?.fusion_threshold, 4), "after feedback"],
          ["Effective scale", num(t?.effective_scale, 4), t?.clamped ? "clamped at ceiling" : "unclamped"],
          ["Verdicts applied", String(t?.feedback_applied ?? 0), `${s?.feedback?.total ?? 0} submitted`],
        ].map(([label, value, hint]) => (
          <div key={label} className="rounded-lg border border-slate-800/70 bg-slate-950/40 p-3">
            <p className="text-xs text-slate-500">{label}</p>
            <p className="mt-0.5 text-lg font-semibold text-slate-100">{value}</p>
            <p className="text-[10px] text-slate-600">{hint}</p>
          </div>
        ))}
      </div>

      {s?.feedback?.by_verdict && Object.keys(s.feedback.by_verdict).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(s.feedback.by_verdict).map(([k, v]) => (
            <span
              key={k}
              className="rounded-md border border-slate-700 bg-slate-800/60 px-2 py-1 text-xs text-slate-300"
            >
              {k.replace("_", " ")}: <span className="font-medium text-slate-100">{v}</span>
            </span>
          ))}
        </div>
      )}

      <div className="mt-4 border-t border-slate-800/70 pt-3">
        <p className="text-xs text-slate-500">
          Submit a verdict for{" "}
          <span className="text-slate-300">
            {canSubmit ? `engine E${engineId} at cycle ${cycle}` : "the selected engine"}
          </span>
          . A false alarm raises the bar; a confirmed fault lowers it.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Optional note…"
            className="min-w-[10rem] flex-1 rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-1.5 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-sky-500"
          />
          <button
            onClick={() => send("false_alarm")}
            disabled={!canSubmit || submit.isPending}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-xs text-slate-200 hover:bg-slate-700/60 disabled:opacity-40"
          >
            <ThumbsDown className="h-3.5 w-3.5" /> False alarm
          </button>
          <button
            onClick={() => send("accepted")}
            disabled={!canSubmit || submit.isPending}
            className="inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
          >
            <ThumbsUp className="h-3.5 w-3.5" /> Confirm fault
          </button>
        </div>
        {submit.isError && (
          <p className="mt-2 text-xs text-red-400">
            {(submit.error as Error)?.message ?? "Feedback submission failed."}
          </p>
        )}
        {submit.isSuccess && (
          <p className="mt-2 text-xs text-emerald-400">
            Verdict recorded — threshold state refreshed.
          </p>
        )}
      </div>
    </MetricCard>
  );
}
