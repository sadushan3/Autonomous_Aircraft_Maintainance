interface ProgressBarProps {
  value: number;
  max?: number;
  color?: "cyan" | "green" | "amber" | "red" | "purple";
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

export function ProgressBar({ value, max = 100, color = "cyan", size = "md", showLabel = false }: ProgressBarProps) {
  const percentage = Math.min((value / max) * 100, 100);

  const colorConfig = {
    cyan: "bg-cyan-500",
    green: "bg-green-500",
    amber: "bg-amber-500",
    red: "bg-red-500",
    purple: "bg-purple-500",
  };

  const sizeConfig = {
    sm: "h-1",
    md: "h-2",
    lg: "h-3",
  };

  return (
    <div className="w-full">
      <div className={`w-full bg-slate-800/50 rounded-full overflow-hidden ${sizeConfig[size]}`}>
        <div
          className={`h-full ${colorConfig[color]} transition-all duration-500 rounded-full`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <div className="mt-1 text-xs text-slate-400 text-right">
          {value} / {max}
        </div>
      )}
    </div>
  );
}
