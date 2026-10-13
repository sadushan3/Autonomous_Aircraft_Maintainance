import { useState } from "react";
import { NavLink } from "react-router";
import { AnimatePresence, motion } from "motion/react";
import {
  Activity,
  AlertTriangle,
  Brain,
  ChevronLeft,
  Clock,
  FileText,
  LayoutDashboard,
  Network,
  Radar,
  Settings,
  Shield,
  Upload,
  Wrench,
} from "lucide-react";
import { useFleetSummary, useHealthCheck, useMaintenanceQueue } from "../../lib/hooks";

type Badge = "critical" | "anomalies" | "queue" | null;

/**
 * Navigation grouped by the platform's actual architecture rather than as one
 * flat list. The twelve pages are not peers: four belong to the Anomaly &
 * Health component, two to Predictive Maintenance, two to the decision layer.
 * Grouping them that way makes the system's shape legible from the sidebar.
 */
const SECTIONS: Array<{
  title: string;
  hint?: string;
  items: Array<{ path: string; icon: typeof Activity; label: string; badge?: Badge }>;
}> = [
  {
    title: "Fleet",
    items: [
      { path: "/", icon: LayoutDashboard, label: "Overview" },
      { path: "/upload", icon: Upload, label: "Upload & Run" },
    ],
  },
  {
    title: "Anomaly & Health",
    hint: "Digital-twin residual analysis",
    items: [
      { path: "/health", icon: Activity, label: "Health Monitoring", badge: "critical" },
      { path: "/anomaly", icon: Radar, label: "Anomaly Detection", badge: "anomalies" },
      { path: "/digital-twin", icon: Network, label: "Digital Twin" },
      { path: "/ai-reasoning", icon: Brain, label: "AI Reasoning" },
    ],
  },
  {
    title: "Predictive Maintenance",
    hint: "Remaining life & failure risk",
    items: [
      { path: "/rul", icon: Clock, label: "RUL Prediction" },
      { path: "/failure-risk", icon: AlertTriangle, label: "Failure Risk" },
    ],
  },
  {
    title: "Decision Layer",
    hint: "Fused maintenance decisions",
    items: [
      { path: "/maintenance", icon: Wrench, label: "Maintenance Supervisor", badge: "queue" },
      { path: "/confidence", icon: Shield, label: "Confidence & Trust" },
    ],
  },
  {
    title: "System",
    items: [
      { path: "/reports", icon: FileText, label: "Reports" },
      { path: "/settings", icon: Settings, label: "Settings" },
    ],
  },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  // Shares React Query's cache with the pages, so this adds no extra fetching.
  const { summary } = useFleetSummary();
  const queue = useMaintenanceQueue();
  const health = useHealthCheck();
  const online = !health.error && !!health.data;

  const badgeValue = (b: Badge): number | null => {
    if (b === "critical") return summary?.criticalCount ?? null;
    if (b === "anomalies") return summary?.anomalyCount ?? null;
    if (b === "queue") return queue.data?.rows ?? null;
    return null;
  };

  return (
    <motion.aside
      animate={{ width: collapsed ? 76 : 256 }}
      transition={{ duration: 0.25, ease: "easeInOut" }}
      className="flex flex-col border-r border-slate-800/50 bg-slate-950/50 backdrop-blur-xl"
    >
      <div className="flex items-center gap-3 border-b border-slate-800/50 p-5">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600">
          <Network className="h-6 w-6 text-white" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: "auto" }}
              exit={{ opacity: 0, width: 0 }}
              className="overflow-hidden whitespace-nowrap"
            >
              <h1 className="text-lg font-bold tracking-tight">AeroTwin OS</h1>
              <p className="text-xs text-slate-400">Digital Twin Platform</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <nav className="flex-1 space-y-5 overflow-y-auto p-3">
        {SECTIONS.map((section, si) => (
          <div key={section.title}>
            <AnimatePresence>
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="mb-1.5 px-3"
                >
                  <div className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">
                    {section.title}
                  </div>
                  {section.hint && (
                    <div className="text-[10px] text-slate-700">{section.hint}</div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            <div className="space-y-0.5">
              {section.items.map((item, ii) => {
                const count = badgeValue(item.badge ?? null);
                return (
                  <motion.div
                    key={item.path}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: si * 0.04 + ii * 0.02 }}
                  >
                    <NavLink
                      to={item.path}
                      end={item.path === "/"}
                      title={collapsed ? item.label : undefined}
                      className={({ isActive }) =>
                        `group relative flex items-center gap-3 rounded-lg px-3 py-2.5 transition-all ${
                          isActive
                            ? "bg-cyan-500/10 text-cyan-400"
                            : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
                        }`
                      }
                    >
                      {({ isActive }) => (
                        <>
                          {isActive && (
                            <motion.span
                              layoutId="nav-active"
                              className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2 rounded-r bg-cyan-400"
                              transition={{ type: "spring", stiffness: 400, damping: 30 }}
                            />
                          )}
                          <item.icon className="h-[18px] w-[18px] shrink-0" />
                          {!collapsed && (
                            <>
                              <span className="flex-1 truncate text-sm font-medium">
                                {item.label}
                              </span>
                              {count !== null && count > 0 && (
                                <span
                                  className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
                                    item.badge === "critical"
                                      ? "bg-red-500/15 text-red-300"
                                      : item.badge === "queue"
                                      ? "bg-amber-500/15 text-amber-300"
                                      : "bg-cyan-500/15 text-cyan-300"
                                  }`}
                                >
                                  {count > 999 ? "999+" : count}
                                </span>
                              )}
                            </>
                          )}
                        </>
                      )}
                    </NavLink>
                  </motion.div>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-slate-800/50 p-3">
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="mb-2 flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-500 transition-colors hover:bg-slate-800/50 hover:text-slate-300"
        >
          <motion.span animate={{ rotate: collapsed ? 180 : 0 }} transition={{ duration: 0.25 }}>
            <ChevronLeft className="h-4 w-4" />
          </motion.span>
          {!collapsed && <span>Collapse</span>}
        </button>

        <div className="flex items-center gap-2 px-3 text-xs">
          <span className="relative flex h-2 w-2 shrink-0">
            {online && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            )}
            <span
              className={`relative inline-flex h-2 w-2 rounded-full ${
                online ? "bg-emerald-400" : "bg-red-500"
              }`}
            />
          </span>
          {!collapsed && (
            <span className={online ? "text-emerald-400/80" : "text-red-400"}>
              {online ? "Backend online" : "Backend offline"}
            </span>
          )}
        </div>
      </div>
    </motion.aside>
  );
}
