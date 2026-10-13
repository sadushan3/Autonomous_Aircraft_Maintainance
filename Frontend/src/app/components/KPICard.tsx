import { LucideIcon } from "lucide-react";
import { ReactNode } from "react";

interface KPICardProps {
  title: string;
  value: ReactNode;
  icon?: LucideIcon;
  trend?: {
    value: string;
    direction: "up" | "down" | "neutral";
  };
  status?: "success" | "warning" | "danger" | "info" | "neutral";
  subtitle?: string;
  children?: ReactNode;
}

export function KPICard({ title, value, icon: Icon, trend, status = "neutral", subtitle, children }: KPICardProps) {
  const statusColors = {
    success: "from-green-500/10 to-emerald-500/5 border-green-500/20",
    warning: "from-amber-500/10 to-yellow-500/5 border-amber-500/20",
    danger: "from-red-500/10 to-rose-500/5 border-red-500/20",
    info: "from-cyan-500/10 to-blue-500/5 border-cyan-500/20",
    neutral: "from-slate-800/50 to-slate-800/30 border-slate-700/30",
  };

  const statusTextColors = {
    success: "text-green-400",
    warning: "text-amber-400",
    danger: "text-red-400",
    info: "text-cyan-400",
    neutral: "text-slate-300",
  };

  return (
    <div
      className={`bg-gradient-to-br ${statusColors[status]} backdrop-blur-xl border rounded-xl p-5 hover:scale-[1.02] transition-transform`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="text-sm text-slate-400 font-medium">{title}</div>
        {Icon && (
          <div className={`p-2 rounded-lg bg-slate-900/50 ${statusTextColors[status]}`}>
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className={`text-3xl font-bold mb-1 ${statusTextColors[status]}`}>{value}</div>

      {subtitle && <div className="text-xs text-slate-500 mb-2">{subtitle}</div>}

      {trend && (
        <div className="flex items-center gap-1 text-xs">
          <span
            className={
              trend.direction === "up"
                ? "text-green-400"
                : trend.direction === "down"
                ? "text-red-400"
                : "text-slate-400"
            }
          >
            {trend.direction === "up" ? "↑" : trend.direction === "down" ? "↓" : "→"} {trend.value}
          </span>
        </div>
      )}

      {children}
    </div>
  );
}
