import { motion } from "motion/react";
import { Loader2, ThumbsDown, ThumbsUp, TrendingDown, TrendingUp, Minus } from "lucide-react";
import { toast } from "sonner";
import { MetricCard } from "./MetricCard";
import { Empty } from "./Live";
import { useThresholdFeedback, useThresholdStatus } from "../../lib/hooks";

const TREND_ICON: Record<string, typeof Minus> = {
  rising: TrendingUp,
  falling: TrendingDown,
  flat: Minus,
};

/**
 * Adaptive risk-threshold state and the operator control that drives it.
 *
 * The Predictive-Maintenance component keeps its own per-horizon thresholds,
 * separate from the anomaly component's fused threshold. Marking an alert a
 * false alarm raises that horizon's bar; confirming one lowers it back toward
 * catching more. Nothing in the UI reached this loop before, so it had never
 * run on a real verdict.
 */
export function RiskThresholdPanel({ fdSubset }: { fdSubset?: string }) {
  const status = useThresholdStatus(fdSubset);
  const feedback = useThresholdFeedback();

  if (!fdSubset) return null;

  const horizons = status.data?.horizons ?? {};
  const entries = Object.entries(horizons).sort(([a], [b]) => Number(a) - Number(b));

  const send = (horizon: string, wasFalseAlarm: boolean) => {
    feedback.mutate(
      { fd_subset: fdSubset, horizon, was_false_alarm: wasFalseAlarm },
      {
        onSuccess: () =>
          toast.success(
            wasFalseAlarm ? "Marked as false alarm" : "Alert confirmed",
            {
              description: `${fdSubset} · ${horizon}-cycle horizon — threshold ${
                wasFalseAlarm ? "raised" : "lowered"
              }.`,
            },
          ),
        onError: (e) =>
          toast.error("Could not record verdict", {
            description: e instanceof Error ? e.message : String(e),
          }),
      },
    );
  };

  return (
    <MetricCard title={`Adaptive Risk Thresholds — ${fdSubset}`}>
      {entries.length === 0 ? (
        <Empty label="No threshold state for this subset." />
      ) : (
        <div className="space-y-3">
          {entries.map(([h, t], i) => {
            const Trend = TREND_ICON[t.trend] ?? Minus;
            const moved = t.current_threshold !== t.original_threshold;
            return (
              <motion.div
                key={h}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.07 }}
                className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="font-semibold text-slate-200">
                      {h}-cycle horizon
                    </div>
                    <div className="mt-0.5 text-xs text-slate-500">
                      threshold{" "}
                      <span className="font-mono text-cyan-300">
                        {t.current_threshold.toFixed(4)}
                      </span>
                      {moved && (
                        <>
                          {" "}
                          <span className="text-slate-600">
                            (from {t.original_threshold.toFixed(4)})
                          </span>
                        </>
                      )}
                      {" · "}
                      {t.feedback_count} verdict{t.feedback_count === 1 ? "" : "s"}
                      {t.false_alarm_rate !== null && (
                        <> · {(t.false_alarm_rate * 100).toFixed(0)}% false alarms</>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className={`flex items-center gap-1 rounded border px-2 py-0.5 text-xs ${
                        t.trend === "rising"
                          ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                          : t.trend === "falling"
                          ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                          : "border-slate-600/30 bg-slate-700/20 text-slate-400"
                      }`}
                    >
                      <Trend className="h-3 w-3" />
                      {t.trend}
                    </span>

                    <button
                      disabled={feedback.isPending}
                      onClick={() => send(h, false)}
                      title="This alert was real — lower the bar so more are caught."
                      className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-300 transition-transform hover:scale-105 disabled:opacity-50"
                    >
                      {feedback.isPending ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <ThumbsUp className="h-3.5 w-3.5" />
                      )}
                      Confirmed
                    </button>
                    <button
                      disabled={feedback.isPending}
                      onClick={() => send(h, true)}
                      title="Nothing was wrong — raise the bar to reduce false alarms."
                      className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs text-amber-300 transition-transform hover:scale-105 disabled:opacity-50"
                    >
                      <ThumbsDown className="h-3.5 w-3.5" />
                      False alarm
                    </button>
                  </div>
                </div>

                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-800">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(t.current_threshold * 100, 100)}%` }}
                    transition={{ duration: 0.7, delay: i * 0.07 }}
                  />
                </div>
                <div className="mt-1 flex justify-between text-[10px] text-slate-600">
                  <span>{t.confirmed_count} confirmed</span>
                  <span>{t.false_alarm_count} false alarms</span>
                </div>
              </motion.div>
            );
          })}
          <p className="text-xs text-slate-500">
            These thresholds were recalibrated against held-out ground truth. Operator verdicts nudge them from
            there — confirming an alert lowers the bar, a false alarm raises it.
          </p>
        </div>
      )}
    </MetricCard>
  );
}
