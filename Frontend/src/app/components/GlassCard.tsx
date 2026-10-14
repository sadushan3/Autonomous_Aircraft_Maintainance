import { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  variant?: "default" | "primary" | "success" | "warning" | "danger";
}

export function GlassCard({ children, className = "", variant = "default" }: GlassCardProps) {
  const variants = {
    default: "from-slate-900/50 to-slate-900/30 border-slate-800/50",
    primary: "from-cyan-900/30 to-blue-900/20 border-cyan-500/30",
    success: "from-green-900/30 to-emerald-900/20 border-green-500/30",
    warning: "from-amber-900/30 to-yellow-900/20 border-amber-500/30",
    danger: "from-red-900/30 to-rose-900/20 border-red-500/30",
  };

  return (
    <div className={`bg-gradient-to-br ${variants[variant]} backdrop-blur-xl border rounded-xl p-6 ${className}`}>
      {children}
    </div>
  );
}
