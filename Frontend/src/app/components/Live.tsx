/**
 * Shared presentation primitives for live backend data.
 *
 * Every page that reads the API needs the same three states -- loading, error,
 * and "the pipeline has not been run yet" -- and the third is not an error: a
 * fresh install genuinely has no decisions until /integration/run-full has run
 * once, and saying so is far more useful than an empty chart.
 */
import { motion } from "motion/react";
import { AlertTriangle, Loader2, Radio, ServerCrash } from "lucide-react";
import type { ReactNode } from "react";
import { GlassCard } from "./GlassCard";

/** Pulsing dot indicating the panel is polling the backend. */
export function LiveBadge({ label = "LIVE" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold tracking-wider text-emerald-400">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
      </span>
      {label}
    </span>
  );
}

export function Loading({ label = "Loading live data…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 py-16 text-slate-400">
      <Loader2 className="h-5 w-5 animate-spin text-cyan-400" />
      <span className="text-sm">{label}</span>
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  // A 404 here means the pipeline simply hasn't produced outputs yet, which is
  // an expected first-run state rather than a fault.
  const notRunYet = /no supervisor decisions|not found|404/i.test(msg);
  return (
    <GlassCard variant={notRunYet ? "warning" : "danger"}>
      <div className="flex items-start gap-4">
        <div
          className={`rounded-lg border p-3 ${
            notRunYet
              ? "border-amber-500/30 bg-amber-500/10"
              : "border-red-500/30 bg-red-500/10"
          }`}
        >
          {notRunYet ? (
            <AlertTriangle className="h-6 w-6 text-amber-400" />
          ) : (
            <ServerCrash className="h-6 w-6 text-red-400" />
          )}
        </div>
        <div>
          <h3 className="mb-1 font-semibold text-slate-200">
            {notRunYet ? "No pipeline results yet" : "Backend unavailable"}
          </h3>
          <p className="text-sm text-slate-400">{msg}</p>
          <p className="mt-2 text-sm text-slate-500">
            {notRunYet
              ? "Upload a C-MAPSS file on the Upload page and run the full pipeline to populate this view."
              : "Confirm the API is running on http://127.0.0.1:8000."}
          </p>
        </div>
      </div>
    </GlassCard>
  );
}

export function Empty({ label = "No data for this selection." }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-14 text-sm text-slate-500">
      <Radio className="h-4 w-4" />
      {label}
    </div>
  );
}

/** Fade/slide-in wrapper; `index` staggers grids without per-item config. */
export function Reveal({
  children,
  index = 0,
  className = "",
}: {
  children: ReactNode;
  index?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: Math.min(index * 0.05, 0.4), ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="flex items-start justify-between gap-4"
    >
      <div>
        <h1 className="mb-2 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-3xl font-bold text-transparent">
          {title}
        </h1>
        {subtitle && <p className="text-slate-400">{subtitle}</p>}
      </div>
      {right}
    </motion.div>
  );
}

/** Number that eases to its value instead of snapping on each poll. */
export function AnimatedNumber({
  value,
  decimals = 0,
  suffix = "",
}: {
  value: number;
  decimals?: number;
  suffix?: string;
}) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0.4, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      {value.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}
      {suffix}
    </motion.span>
  );
}
