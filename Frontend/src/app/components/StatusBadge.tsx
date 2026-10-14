interface StatusBadgeProps {
  status: "healthy" | "warning" | "critical" | "normal" | "offline" | "online" | "active" | "inactive";
  label?: string;
  size?: "sm" | "md" | "lg";
}

export function StatusBadge({ status, label, size = "md" }: StatusBadgeProps) {
  const statusConfig = {
    healthy: { bg: "bg-green-500/10", border: "border-green-500/30", text: "text-green-400", label: "Healthy" },
    warning: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400", label: "Warning" },
    critical: { bg: "bg-red-500/10", border: "border-red-500/30", text: "text-red-400", label: "Critical" },
    normal: { bg: "bg-slate-500/10", border: "border-slate-500/30", text: "text-slate-400", label: "Normal" },
    offline: { bg: "bg-slate-500/10", border: "border-slate-500/30", text: "text-slate-400", label: "Offline" },
    online: { bg: "bg-green-500/10", border: "border-green-500/30", text: "text-green-400", label: "Online" },
    active: { bg: "bg-cyan-500/10", border: "border-cyan-500/30", text: "text-cyan-400", label: "Active" },
    inactive: { bg: "bg-slate-500/10", border: "border-slate-500/30", text: "text-slate-400", label: "Inactive" },
  };

  const sizeConfig = {
    sm: "px-2 py-0.5 text-xs",
    md: "px-3 py-1 text-sm",
    lg: "px-4 py-1.5 text-base",
  };

  const config = statusConfig[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 ${config.bg} ${config.border} ${config.text} border rounded-full ${sizeConfig[size]} font-medium`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${config.text.replace("text-", "bg-")} animate-pulse`} />
      {label || config.label}
    </span>
  );
}
