import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { AnimatePresence, motion } from "motion/react";
import {
  Activity,
  ChevronRight,
  Cpu,
  Database,
  RefreshCw,
  Search,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useFleetSummary, useHealthCheck } from "../../lib/hooks";
import { NotificationCenter } from "./NotificationCenter";
import { engineLink } from "../../lib/engine-param";

const PAGE_TITLES: Record<string, { section: string; title: string }> = {
  "/": { section: "Fleet", title: "Executive Overview" },
  "/upload": { section: "Fleet", title: "Upload & Run" },
  "/health": { section: "Anomaly & Health", title: "Health Monitoring" },
  "/anomaly": { section: "Anomaly & Health", title: "Anomaly Detection" },
  "/digital-twin": { section: "Anomaly & Health", title: "Digital Twin" },
  "/ai-reasoning": { section: "Anomaly & Health", title: "AI Reasoning" },
  "/rul": { section: "Predictive Maintenance", title: "RUL Prediction" },
  "/failure-risk": { section: "Predictive Maintenance", title: "Failure Risk" },
  "/maintenance": { section: "Decision Layer", title: "Maintenance Supervisor" },
  "/confidence": { section: "Decision Layer", title: "Confidence & Trust" },
  "/reports": { section: "System", title: "Reports" },
  "/settings": { section: "System", title: "Settings" },
};

export function Topbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const health = useHealthCheck();
  const { summary, latest } = useFleetSummary();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const crumb = PAGE_TITLES[location.pathname] ?? { section: "Fleet", title: "Dashboard" };
  const online = !health.error && !!health.data;

  // Engine jump: filters the loaded fleet rather than a hardcoded list.
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return latest
      .filter((r) => String(r.engine_id).includes(q) || String(r.health_state ?? "").toLowerCase().includes(q))
      .slice(0, 6);
  }, [query, latest]);

  return (
    <header className="flex h-16 items-center justify-between gap-4 border-b border-slate-800/50 bg-slate-950/30 px-6 backdrop-blur-xl">
      <div className="flex min-w-0 items-center gap-3">
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.25 }}
          className="flex min-w-0 items-center gap-2"
        >
          <span className="truncate text-xs text-slate-500">{crumb.section}</span>
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-slate-700" />
          <span className="truncate text-sm font-semibold text-slate-200">{crumb.title}</span>
        </motion.div>
      </div>

      <div className="relative hidden md:block md:w-72">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
        <input
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Jump to engine…"
          className="w-full rounded-lg border border-slate-700/50 bg-slate-800/40 py-1.5 pl-9 pr-3 text-sm text-slate-200 outline-none transition-colors placeholder:text-slate-600 focus:border-cyan-500/50"
        />
        <AnimatePresence>
          {open && matches.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              className="absolute z-50 mt-1 w-full overflow-hidden rounded-lg border border-slate-700/60 bg-slate-900 shadow-xl"
            >
              {matches.map((r) => (
                <button
                  key={String(r.engine_id)}
                  onMouseDown={() => {
                    // Carry the engine so the destination opens on it, and so
                    // every other screen keeps it as you move between them.
                    navigate(engineLink("/health", r.engine_id));
                    setQuery("");
                  }}
                  className="flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors hover:bg-slate-800"
                >
                  <span className="text-slate-200">Engine E{String(r.engine_id)}</span>
                  <span className="text-xs text-slate-500">
                    {r.health_state ?? "—"} · {(r.health_index ?? 0).toFixed(0)}
                  </span>
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="flex items-center gap-2">
        {summary && (
          <div className="hidden items-center gap-3 rounded-lg border border-slate-700/40 bg-slate-800/30 px-3 py-1.5 lg:flex">
            <span className="flex items-center gap-1.5 text-xs text-slate-400">
              <Cpu className="h-3.5 w-3.5 text-cyan-400" />
              {summary.engines} engines
            </span>
            <span className="h-3 w-px bg-slate-700" />
            <span className="flex items-center gap-1.5 text-xs text-slate-400">
              <Database className="h-3.5 w-3.5 text-slate-500" />
              {summary.rows.toLocaleString()} rows
            </span>
            {summary.subsets.length > 0 && (
              <>
                <span className="h-3 w-px bg-slate-700" />
                <span className="text-xs text-slate-400">{summary.subsets.join(", ")}</span>
              </>
            )}
          </div>
        )}

        <div
          className={`flex items-center gap-2 rounded-full border px-3 py-1.5 ${
            online
              ? "border-emerald-500/20 bg-emerald-500/10"
              : "border-red-500/20 bg-red-500/10"
          }`}
        >
          <span className="relative flex h-2 w-2">
            {online && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            )}
            <span
              className={`relative inline-flex h-2 w-2 rounded-full ${
                online ? "bg-emerald-400" : "bg-red-500"
              }`}
            />
          </span>
          <span
            className={`text-xs font-medium ${online ? "text-emerald-400" : "text-red-400"}`}
          >
            {online ? "Live" : "Offline"}
          </span>
        </div>

        <button
          onClick={() => qc.invalidateQueries()}
          title="Refresh all panels"
          className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-slate-200"
        >
          <RefreshCw className="h-[18px] w-[18px]" />
        </button>

        <NotificationCenter />

        {summary && (
          <div className="hidden items-center gap-2 rounded-lg border border-slate-700/30 bg-slate-800/30 px-3 py-1.5 xl:flex">
            <Activity className="h-4 w-4 text-cyan-400" />
            <div className="text-xs">
              <div className="font-semibold text-slate-200">
                {summary.avgHealth.toFixed(1)} health
              </div>
              <div className="text-slate-500">fleet mean</div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
