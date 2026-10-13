import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, Loader2, MessageSquarePlus, ThumbsDown, ThumbsUp, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import { useSubmitFeedback } from "../../lib/hooks";

type Verdict = "accepted" | "false_alarm" | "missed";

const VERDICTS: Array<{
  key: Verdict;
  label: string;
  hint: string;
  icon: typeof ThumbsUp;
  tone: string;
}> = [
  {
    key: "accepted",
    label: "Correct",
    hint: "The anomaly was real and the diagnosis was right.",
    icon: ThumbsUp,
    tone: "emerald",
  },
  {
    key: "false_alarm",
    label: "False Alarm",
    hint: "Nothing was actually wrong. Repeated verdicts raise the threshold.",
    icon: ThumbsDown,
    tone: "amber",
  },
  {
    key: "missed",
    label: "Missed",
    hint: "Something was wrong that the system did not flag.",
    icon: TriangleAlert,
    tone: "red",
  },
];

const TONE: Record<string, { border: string; bg: string; text: string }> = {
  emerald: { border: "border-emerald-500/40", bg: "bg-emerald-500/10", text: "text-emerald-300" },
  amber: { border: "border-amber-500/40", bg: "bg-amber-500/10", text: "text-amber-300" },
  red: { border: "border-red-500/40", bg: "bg-red-500/10", text: "text-red-300" },
};

/**
 * Operator verdict control for one raised alert.
 *
 * Submitting here drives the component's learning loop: the verdict is stored,
 * matched into alert memory, and the adaptive anomaly threshold is recomputed
 * from the full feedback history. A sustained false-alarm rate also nudges the
 * health-state boundaries.
 */
export function FeedbackControl({
  engineId,
  cycle,
  systemRootCause,
  compact = false,
}: {
  engineId: number | string;
  cycle: number;
  systemRootCause?: string | null;
  compact?: boolean;
}) {
  const submit = useSubmitFeedback();
  const [chosen, setChosen] = useState<Verdict | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [comment, setComment] = useState("");
  const [actualCause, setActualCause] = useState("");
  const [done, setDone] = useState(false);

  const send = (verdict: Verdict) => {
    setChosen(verdict);
    submit.mutate(
      {
        engine_id: Number(engineId),
        cycle,
        verdict,
        operator: "dashboard_operator",
        comment: comment.trim(),
        // Only send a corrected cause when the operator actually typed one;
        // an empty string would be recorded as a confirmed (blank) diagnosis.
        confirmed_root_cause: actualCause.trim() || null,
      },
      {
        onSuccess: (res) => {
          setDone(true);
          const total = (res as { total_feedback_records?: number }).total_feedback_records;
          const changed = (res as { thresholds_updated?: number }).thresholds_updated ?? 0;
          toast.success("Verdict recorded", {
            description:
              `E${engineId} @ cycle ${cycle} → ${verdict.replace("_", " ")}` +
              (total ? ` · ${total} record${total === 1 ? "" : "s"} total` : "") +
              (changed ? ` · ${changed} threshold(s) adapted` : ""),
          });
        },
        onError: (e) => {
          setChosen(null);
          toast.error("Could not record verdict", {
            description: e instanceof Error ? e.message : String(e),
          });
        },
      },
    );
  };

  if (done) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300"
      >
        <CheckCircle2 className="h-4 w-4" />
        Verdict recorded — fed into threshold adaptation.
      </motion.div>
    );
  }

  return (
    <div className="space-y-3">
      {!compact && (
        <div className="text-sm font-semibold text-slate-300">Was this alert correct?</div>
      )}

      <div className="flex flex-wrap gap-2">
        {VERDICTS.map((v, i) => {
          const t = TONE[v.tone];
          const busy = submit.isPending && chosen === v.key;
          return (
            <motion.button
              key={v.key}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              disabled={submit.isPending}
              onClick={() => send(v.key)}
              title={v.hint}
              className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition-colors disabled:opacity-50 ${t.border} ${t.bg} ${t.text}`}
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <v.icon className="h-4 w-4" />}
              {v.label}
            </motion.button>
          );
        })}

        <button
          onClick={() => setExpanded((e) => !e)}
          className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-400 transition-colors hover:text-slate-200"
        >
          <MessageSquarePlus className="h-4 w-4" />
          {expanded ? "Hide detail" : "Add detail"}
        </button>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-2 overflow-hidden"
          >
            <input
              value={actualCause}
              onChange={(e) => setActualCause(e.target.value)}
              placeholder={
                systemRootCause
                  ? `Actual cause, if not "${systemRootCause}"`
                  : "Actual root cause found (optional)"
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 outline-none transition-colors focus:border-cyan-500/60"
            />
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={2}
              placeholder="Notes for the maintenance record (optional)"
              className="w-full resize-none rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 outline-none transition-colors focus:border-cyan-500/60"
            />
            <p className="text-xs text-slate-500">
              A corrected cause is written into anomaly memory so future matching alerts inherit it.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
