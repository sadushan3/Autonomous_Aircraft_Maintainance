import { useState } from "react";
import { motion } from "motion/react";
import { LineChart as LineChartIcon, Loader2, Play } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MetricCard } from "./MetricCard";
import { Empty } from "./Live";
import { useTrajectory } from "../../lib/hooks";

/**
 * Forward RUL / risk trajectory for one engine.
 *
 * Deliberately on-demand rather than polled: the service replays the engine's
 * whole sequence through both the RUL and Failure-Risk models at growing
 * window sizes, which is orders of magnitude heavier than reading the
 * dashboard, and the result barely changes between pipeline runs.
 */
export function TrajectoryPanel({ engineId }: { engineId?: number | string }) {
  const [armed, setArmed] = useState(false);
  const q = useTrajectory(armed ? engineId : undefined, 40);

  const points = q.data?.points ?? [];
  const chart = points.map((p) => ({
    cycle: p.cycle,
    rul: Number(p.predicted_rul.toFixed(2)),
    lower: Number(p.lower_bound.toFixed(2)),
    band: Number((p.upper_bound - p.lower_bound).toFixed(2)),
    risk30: Number((p.risk_30 * 100).toFixed(2)),
  }));

  const first = points[0];
  const last = points[points.length - 1];
  const delta = first && last ? last.predicted_rul - first.predicted_rul : 0;

  return (
    <MetricCard
      title={engineId ? `Engine E${engineId} — RUL Trajectory Forecast` : "Trajectory Forecast"}
      actions={
        <button
          onClick={() => setArmed(true)}
          disabled={!engineId || q.isFetching}
          className="flex items-center gap-2 rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-1.5 text-xs text-cyan-300 transition-transform hover:scale-105 disabled:opacity-40"
        >
          {q.isFetching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
          {q.isFetching ? "Forecasting…" : armed ? "Recompute" : "Run forecast"}
        </button>
      }
    >
      {!armed ? (
        <div className="flex flex-col items-center justify-center gap-3 py-14 text-center">
          <LineChartIcon className="h-10 w-10 text-slate-700" />
          <p className="max-w-md text-sm text-slate-500">
            Replays this engine's full sensor history through the RUL and Failure-Risk models at growing window
            sizes, showing how the prediction evolved cycle over cycle. Heavier than a dashboard read, so it runs
            only when you ask.
          </p>
        </div>
      ) : q.isLoading ? (
        <div className="flex items-center justify-center gap-3 py-16 text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin text-cyan-400" />
          <span className="text-sm">Replaying sequence through both models…</span>
        </div>
      ) : q.error ? (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {q.error instanceof Error ? q.error.message : String(q.error)}
        </div>
      ) : chart.length === 0 ? (
        <Empty label="No trajectory points returned for this engine." />
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
            {[
              ["Points", String(points.length)],
              ["Stride", String(q.data?.stride ?? "—")],
              ["RUL change", `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}`],
              ["Final state", last?.lifecycle_state ?? "—"],
            ].map(([k, v], i) => (
              <motion.div
                key={k}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06 }}
                className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-2.5"
              >
                <div className="text-xs text-slate-500">{k}</div>
                <div className="mt-0.5 font-semibold text-cyan-300">{v}</div>
              </motion.div>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={chart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="cycle" stroke="#64748b" />
              <YAxis yAxisId="l" stroke="#64748b" />
              <YAxis yAxisId="r" orientation="right" stroke="#64748b" domain={[0, 100]} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
              <Legend />
              {/* Stacked lower + band renders the conformal interval as a ribbon. */}
              <Area yAxisId="l" type="monotone" dataKey="lower" stackId="ci" stroke="none" fill="transparent" name=" " />
              <Area
                yAxisId="l"
                type="monotone"
                dataKey="band"
                stackId="ci"
                stroke="none"
                fill="#3b82f6"
                fillOpacity={0.16}
                name="Conformal interval"
                animationDuration={900}
              />
              <Line
                yAxisId="l"
                type="monotone"
                dataKey="rul"
                name="Predicted RUL"
                stroke="#22d3ee"
                strokeWidth={2.5}
                dot={{ r: 2 }}
                animationDuration={900}
              />
              <Line
                yAxisId="r"
                type="monotone"
                dataKey="risk30"
                name="Risk @30 (%)"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                animationDuration={900}
              />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="mt-2 text-xs text-slate-500">
            The interval width is a fixed split-conformal band, so it does not narrow as the estimate sharpens.
          </p>
        </>
      )}
    </MetricCard>
  );
}
