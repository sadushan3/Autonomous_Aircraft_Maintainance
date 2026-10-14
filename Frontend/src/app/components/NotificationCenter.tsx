/**
 * The bell: a live notification centre derived from data the UI already holds.
 *
 * Nothing here polls a new endpoint, computes a model output or changes a
 * threshold. Every item is a reading of the fleet summary, the operating-regime
 * drift check, the maintenance queue and the backend health ping -- the same
 * four queries the dashboards already subscribe to, so opening the bell costs
 * no extra requests and can never disagree with the page behind it.
 *
 * Read state lives in `localStorage`, keyed by a signature that includes the
 * figure the item is about. That is deliberate: dismissing "10 engines need
 * immediate maintenance" should not also dismiss the same alert once it becomes
 * eleven engines. A stale acknowledgement of a changed situation is worse than
 * no acknowledgement at all.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { AnimatePresence, motion } from "motion/react";
import {
  AlertTriangle,
  Bell,
  CheckCheck,
  ClipboardList,
  ServerCrash,
  ShieldAlert,
  Waves,
  Wrench,
} from "lucide-react";
import {
  useFleetModes,
  useFleetSummary,
  useHealthCheck,
  useMaintenanceQueue,
} from "../../lib/hooks";

type Severity = "critical" | "warning" | "info";

interface Note {
  /** Stable across renders, but changes when the underlying figure changes. */
  id: string;
  severity: Severity;
  icon: typeof Bell;
  title: string;
  detail: string;
  /** Where the operator should go to act on it. */
  to: string;
}

const STORAGE_KEY = "aerotwin.notifications.read";

const SEVERITY_STYLE: Record<Severity, { dot: string; text: string; ring: string }> = {
  critical: { dot: "bg-rose-500", text: "text-rose-400", ring: "border-rose-500/30" },
  warning: { dot: "bg-amber-500", text: "text-amber-400", ring: "border-amber-500/30" },
  info: { dot: "bg-cyan-500", text: "text-cyan-400", ring: "border-cyan-500/30" },
};

/** localStorage can throw (private mode, blocked site data), so never let it break the page. */
function readDismissed(): string[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function writeDismissed(ids: string[]) {
  try {
    // Bounded: ids carry the figure they describe, so they accumulate over time.
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(ids.slice(-200)));
  } catch {
    /* preference only -- losing it costs the operator nothing */
  }
}

export function NotificationCenter({ compact = false }: { compact?: boolean }) {
  const navigate = useNavigate();
  const { summary, latest, error } = useFleetSummary();
  const modes = useFleetModes();
  const queue = useMaintenanceQueue();
  const health = useHealthCheck();

  const [open, setOpen] = useState(false);
  const [dismissed, setDismissed] = useState<string[]>(readDismissed);
  const boxRef = useRef<HTMLDivElement>(null);

  // Close on outside click and on Escape, so the panel never traps the page.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const notes = useMemo<Note[]>(() => {
    const out: Note[] = [];

    if (health.isError || (!health.isLoading && !health.data)) {
      out.push({
        id: "backend-offline",
        severity: "critical",
        icon: ServerCrash,
        title: "Backend unreachable",
        detail: "The API is not answering. Panels are showing the last values received.",
        to: "/settings",
      });
      // Nothing below can be trusted while the API is down.
      return out;
    }

    if (error) {
      out.push({
        id: "fleet-summary-failed",
        severity: "warning",
        icon: AlertTriangle,
        title: "Fleet summary unavailable",
        detail: (error as Error)?.message ?? "The fused decision table could not be read.",
        to: "/upload",
      });
    }

    if (!summary) return out;

    const act = latest.filter((r) => r.final_decision === "immediate_maintenance").length;
    if (act > 0) {
      out.push({
        id: `immediate-maintenance:${act}`,
        severity: "critical",
        icon: Wrench,
        title: `${act} engine${act === 1 ? "" : "s"} need immediate maintenance`,
        detail: "The supervisor escalated these past scheduling. Review them first.",
        to: "/maintenance",
      });
    }

    // "Failure Risk" is the bottom health band, below Critical -- worth its own
    // line rather than being folded into a single critical total.
    const failureRisk = Object.entries(summary.healthStateCounts).find(
      ([k]) => k.toLowerCase() === "failure risk",
    )?.[1];
    if (failureRisk) {
      out.push({
        id: `failure-risk:${failureRisk}`,
        severity: "critical",
        icon: ShieldAlert,
        title: `${failureRisk} engine${failureRisk === 1 ? "" : "s"} in the failure-risk band`,
        detail: "Health index below 20 — the worst band the health model reports.",
        to: "/health",
      });
    }

    const shortRul = latest.filter(
      (r) => typeof r.predicted_rul === "number" && (r.predicted_rul as number) <= 20,
    ).length;
    if (shortRul > 0) {
      out.push({
        id: `short-rul:${shortRul}`,
        severity: "warning",
        icon: AlertTriangle,
        title: `${shortRul} engine${shortRul === 1 ? "" : "s"} under 20 cycles of predicted life`,
        detail: "Inside the planning horizon — schedule before the window closes.",
        to: "/rul",
      });
    }

    if (modes.data?.drift.drift_detected) {
      out.push({
        id: `drift:${modes.data.drift.severity}:${modes.data.drift.psi.toFixed(3)}`,
        severity: "warning",
        icon: Waves,
        title: `Operating-regime drift: ${modes.data.drift.severity}`,
        detail: `PSI ${modes.data.drift.psi.toFixed(3)} against a threshold of ${modes.data.drift.threshold}. Predictions may be off-distribution.`,
        to: "/digital-twin",
      });
    }

    // Only worth raising when it is not simply "everything", which says more
    // about the confidence calibration than about any individual engine.
    if (summary.reviewCount > 0 && summary.reviewCount < summary.engines) {
      out.push({
        id: `review:${summary.reviewCount}`,
        severity: "warning",
        icon: ClipboardList,
        title: `${summary.reviewCount} decisions flagged for human review`,
        detail: "The confidence gate could not settle these on its own.",
        to: "/confidence",
      });
    } else if (summary.reviewCount >= summary.engines && summary.engines > 0) {
      out.push({
        id: `review-all:${summary.engines}`,
        severity: "warning",
        icon: ClipboardList,
        title: "Every decision is flagged for review",
        detail: `Mean confidence is ${(summary.avgConfidence * 100).toFixed(0)}%. Treat the decision layer as advisory until this is calibrated.`,
        to: "/confidence",
      });
    }

    const queued = queue.data?.rows ?? 0;
    if (queued > 0) {
      const engines = new Set((queue.data?.queue ?? []).map((r) => String(r.engine_id))).size;
      out.push({
        id: `queue:${engines}:${queued}`,
        severity: "info",
        icon: Wrench,
        title: `${engines} engine${engines === 1 ? "" : "s"} in the maintenance queue`,
        detail: `${queued.toLocaleString()} flagged cycles awaiting action.`,
        to: "/maintenance",
      });
    }

    return out;
  }, [summary, latest, modes.data, queue.data, health.isError, health.isLoading, health.data, error]);

  const unread = notes.filter((n) => !dismissed.includes(n.id));

  function markAllRead() {
    const next = Array.from(new Set([...dismissed, ...notes.map((n) => n.id)]));
    setDismissed(next);
    writeDismissed(next);
  }

  function openNote(note: Note) {
    const next = Array.from(new Set([...dismissed, note.id]));
    setDismissed(next);
    writeDismissed(next);
    setOpen(false);
    navigate(note.to);
  }

  const trigger = compact
    ? "flex h-9 w-9 items-center justify-center rounded-full border border-slate-800 bg-slate-900/60 text-slate-400 transition-colors hover:text-slate-200"
    : "rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-slate-200";

  return (
    <div className="relative" ref={boxRef}>
      <button
        onClick={() => setOpen((o) => !o)}
        title={unread.length ? `${unread.length} new notification(s)` : "No new notifications"}
        aria-label="Notifications"
        aria-expanded={open}
        className={`relative ${trigger}`}
      >
        <Bell className={compact ? "h-4 w-4" : "h-[18px] w-[18px]"} />
        {unread.length > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-red-500 px-1 text-[9px] font-bold text-white"
          >
            {unread.length > 9 ? "9+" : unread.length}
          </motion.span>
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 z-50 mt-2 w-[22rem] overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900/95 shadow-2xl backdrop-blur-xl"
          >
            <div className="flex items-center justify-between border-b border-slate-800/70 px-4 py-2.5">
              <span className="text-sm font-semibold text-slate-200">Notifications</span>
              {notes.length > 0 && (
                <button
                  onClick={markAllRead}
                  className="flex items-center gap-1 text-[11px] text-slate-500 transition-colors hover:text-cyan-400"
                >
                  <CheckCheck className="h-3.5 w-3.5" />
                  Mark all read
                </button>
              )}
            </div>

            <div className="max-h-[22rem] overflow-y-auto">
              {notes.length === 0 ? (
                <p className="px-4 py-8 text-center text-sm text-slate-500">
                  Nothing needs attention.
                </p>
              ) : (
                notes.map((n) => {
                  const style = SEVERITY_STYLE[n.severity];
                  const isUnread = !dismissed.includes(n.id);
                  const Icon = n.icon;
                  return (
                    <button
                      key={n.id}
                      onClick={() => openNote(n)}
                      className={`flex w-full gap-3 border-b border-slate-800/40 px-4 py-3 text-left transition-colors hover:bg-slate-800/40 ${
                        isUnread ? "" : "opacity-55"
                      }`}
                    >
                      <span
                        className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border bg-slate-950/60 ${style.ring} ${style.text}`}
                      >
                        <Icon className="h-3.5 w-3.5" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-2">
                          <span className="truncate text-sm font-medium text-slate-200">
                            {n.title}
                          </span>
                          {isUnread && (
                            <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${style.dot}`} />
                          )}
                        </span>
                        <span className="mt-0.5 block text-[11px] leading-snug text-slate-500">
                          {n.detail}
                        </span>
                      </span>
                    </button>
                  );
                })
              )}
            </div>

            <div className="border-t border-slate-800/70 px-4 py-2 text-[10px] text-slate-600">
              Derived live from the last pipeline run — no separate alert store.
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
