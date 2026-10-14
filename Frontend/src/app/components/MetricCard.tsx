import { ReactNode } from "react";

interface MetricCardProps {
  title: string;
  /**
   * Optional qualifier under the title. Used to state the basis a figure is
   * computed on (population, window, units) so a headline number is never
   * ambiguous. Omitted usages render exactly as before.
   */
  subtitle?: string;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
}

export function MetricCard({
  title,
  subtitle,
  children,
  className = "",
  actions,
}: MetricCardProps) {
  return (
    <div
      className={`bg-gradient-to-br from-slate-900/50 to-slate-900/30 backdrop-blur-xl border border-slate-800/50 rounded-xl p-6 ${className}`}
    >
      <div className="flex items-start justify-between mb-4 gap-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-200">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        </div>
        {actions}
      </div>
      <div>{children}</div>
    </div>
  );
}
