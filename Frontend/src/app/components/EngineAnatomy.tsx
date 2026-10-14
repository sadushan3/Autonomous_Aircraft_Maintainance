import { useState } from "react";
import { motion } from "motion/react";
import type { SubsystemAttribution } from "../../lib/types";

/**
 * Gas-path layout, in the order air actually flows through a turbofan.
 *
 * `x`/`w` are fractions of the diagram width so the whole thing scales with
 * its container, and `h` is a fraction of height — the core sections are drawn
 * taller than the bypass duct to read as an engine cross-section rather than a
 * bar chart.
 */
const LAYOUT: Array<{ id: string; label: string; x: number; w: number; y: number; h: number }> = [
  { id: "fan", label: "Fan", x: 0.02, w: 0.12, y: 0.12, h: 0.76 },
  { id: "bypass", label: "Bypass Duct", x: 0.15, w: 0.7, y: 0.06, h: 0.16 },
  { id: "lpc", label: "LPC", x: 0.15, w: 0.14, y: 0.28, h: 0.44 },
  { id: "hpc", label: "HPC", x: 0.3, w: 0.14, y: 0.31, h: 0.38 },
  { id: "burner", label: "Combustor", x: 0.45, w: 0.13, y: 0.29, h: 0.42 },
  { id: "hpt", label: "HPT", x: 0.59, w: 0.12, y: 0.31, h: 0.38 },
  { id: "lpt", label: "LPT", x: 0.72, w: 0.13, y: 0.28, h: 0.44 },
  { id: "bleed", label: "Bleed Air", x: 0.15, w: 0.7, y: 0.78, h: 0.14 },
];

const STATUS_FILL: Record<string, string> = {
  degraded: "#ef4444",
  elevated: "#f97316",
  nominal: "#1e293b",
};
const STATUS_STROKE: Record<string, string> = {
  degraded: "#fca5a5",
  elevated: "#fdba74",
  nominal: "#334155",
};

/**
 * Engine cross-section shaded by each subsystem's contribution to the anomaly.
 *
 * This renders the reasoning layer's `subsystem_attribution` — a contribution
 * share, member sensors, mean residual z and elevated/nominal status per
 * physical section — which is a more interpretable answer to "why" than a flat
 * ranked sensor list, because it says *where* in the engine the deviation sits.
 */
export function EngineAnatomy({
  attribution,
  height = 300,
}: {
  attribution: SubsystemAttribution[];
  height?: number;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const width = 760;

  const byId = new Map(attribution.map((a) => [a.subsystem, a]));
  const maxContribution = Math.max(...attribution.map((a) => a.contribution), 0.0001);
  const active = hovered ? byId.get(hovered) : null;

  if (!attribution.length) {
    return (
      <div className="py-12 text-center text-sm text-slate-500">
        No subsystem attribution for this alert.
      </div>
    );
  }

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ maxHeight: height }}>
        <defs>
          <filter id="sub-glow">
            <feGaussianBlur stdDeviation="5" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <marker id="flow" markerWidth="8" markerHeight="8" refX="7" refY="2.5" orient="auto">
            <path d="M0,0 L0,5 L8,2.5 z" fill="#334155" />
          </marker>
        </defs>

        {/* Airflow direction, front to back. */}
        <line
          x1={width * 0.02}
          y1={height * 0.5}
          x2={width * 0.95}
          y2={height * 0.5}
          stroke="#1e293b"
          strokeWidth={1}
          strokeDasharray="4 6"
          markerEnd="url(#flow)"
        />

        {LAYOUT.map((sec, i) => {
          const a = byId.get(sec.id);
          const share = a ? a.contribution / maxContribution : 0;
          const status = a?.status ?? "nominal";
          const isHot = status !== "nominal";
          const dim = hovered !== null && hovered !== sec.id;

          return (
            <motion.g
              key={sec.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: dim ? 0.35 : 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.06 }}
              onMouseEnter={() => setHovered(sec.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "pointer" }}
            >
              {isHot && (
                <motion.rect
                  x={width * sec.x - 3}
                  y={height * sec.y - 3}
                  width={width * sec.w + 6}
                  height={height * sec.h + 6}
                  rx={8}
                  fill="none"
                  stroke={STATUS_STROKE[status]}
                  strokeWidth={1.5}
                  animate={{ opacity: [0.6, 0.05, 0.6] }}
                  transition={{ duration: 2.2, repeat: Infinity }}
                />
              )}
              <rect
                x={width * sec.x}
                y={height * sec.y}
                width={width * sec.w}
                height={height * sec.h}
                rx={6}
                fill={STATUS_FILL[status]}
                fillOpacity={status === "nominal" ? 0.55 : 0.2 + share * 0.65}
                stroke={STATUS_STROKE[status]}
                strokeWidth={isHot ? 2 : 1}
                filter={status === "degraded" ? "url(#sub-glow)" : undefined}
              />
              <text
                x={width * (sec.x + sec.w / 2)}
                y={height * (sec.y + sec.h / 2) - 4}
                textAnchor="middle"
                fontSize={12}
                fontWeight={600}
                fill={isHot ? "#fed7aa" : "#94a3b8"}
                style={{ pointerEvents: "none" }}
              >
                {sec.label}
              </text>
              {a && (
                <text
                  x={width * (sec.x + sec.w / 2)}
                  y={height * (sec.y + sec.h / 2) + 12}
                  textAnchor="middle"
                  fontSize={11}
                  fill={isHot ? "#fdba74" : "#64748b"}
                  style={{ pointerEvents: "none" }}
                >
                  {(a.contribution * 100).toFixed(1)}%
                </text>
              )}
            </motion.g>
          );
        })}
      </svg>

      <div className="mt-2 min-h-[3.5rem] rounded-lg border border-slate-800/60 bg-slate-900/40 p-3">
        {active ? (
          <motion.div
            key={active.subsystem}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold text-slate-200">{active.subsystem_label}</span>
              <span
                className={`rounded border px-2 py-0.5 text-xs ${
                  active.status === "degraded"
                    ? "border-red-500/30 bg-red-500/10 text-red-300"
                    : active.status === "elevated"
                    ? "border-orange-500/30 bg-orange-500/10 text-orange-300"
                    : "border-slate-600/30 bg-slate-700/20 text-slate-400"
                }`}
              >
                {active.status}
              </span>
              <span className="text-xs text-slate-500">
                {(active.contribution * 100).toFixed(1)}% of deviation · mean residual z{" "}
                <span className="font-mono">{active.mean_residual_z >= 0 ? "+" : ""}
                {active.mean_residual_z.toFixed(3)}</span>
              </span>
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {active.sensors.map((s) => (
                <span
                  key={s}
                  className="rounded border border-slate-700/50 bg-slate-800/50 px-1.5 py-0.5 font-mono text-[11px] text-slate-400"
                >
                  {s}
                </span>
              ))}
            </div>
          </motion.div>
        ) : (
          <p className="text-sm text-slate-500">
            Hover a section for its contribution share, member sensors and mean residual.
          </p>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-red-500" /> degraded
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-orange-500" /> elevated
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm border border-slate-600 bg-slate-800" /> nominal
        </span>
        <span className="ml-auto">Shading scales with contribution share · airflow runs left to right.</span>
      </div>
    </div>
  );
}
