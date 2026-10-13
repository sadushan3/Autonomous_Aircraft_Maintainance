import { AlertTriangle, Crosshair, Info } from "lucide-react";
import { motion } from "motion/react";
import { COMPONENT_STATUS_COLOR } from "../../lib/chartColors";
import type { ComponentSource, SupervisorComponent } from "../../lib/types";

/**
 * Digital-twin condition per engine section, as a grid of selectable cards.
 *
 * What each card shows is decided by what the run can actually support:
 *
 * * `health_pct` is the section's mean |robust residual z| mapped onto 0-100.
 *   It is twin *agreement*, not a manufacturer health index — 100 means the
 *   twin predicted every informative sensor in that section correctly at this
 *   cycle.
 * * Sections whose only sensors are the constant (1/5/18/19) or quantised
 *   (10/16) C-MAPSS channels get `health_pct: null`. The card says so instead
 *   of showing a number that would describe the dataset rather than the engine.
 * * When the processed residual table cannot be aligned to these decisions at
 *   all, every card still carries its section name, its sensors, its inspection
 *   focus and whether the reasoning layer blamed it — only the measured values
 *   are withheld.
 *
 * The reference design for this panel showed a vibration reading per section.
 * C-MAPSS has no vibration channel, so the third figure here is the section's
 * mean residual deviation, which is real and is what the health figure is
 * actually computed from.
 */
export function SupervisorComponentMap({
  components,
  source,
  selectedId,
  onSelect,
}: {
  components: SupervisorComponent[];
  source: ComponentSource;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-3">
      {!source.available && (
        <div className="flex items-start gap-3 rounded-lg border border-amber-500/25 bg-amber-500/5 p-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          <div className="text-xs leading-relaxed text-slate-300">
            <span className="font-semibold text-amber-300">
              Twin residuals unavailable for this engine.
            </span>{" "}
            {source.unavailable_reason}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
        {components.map((component, index) => (
          <ComponentCard
            key={component.id}
            component={component}
            index={index}
            selected={component.id === selectedId}
            onSelect={onSelect}
          />
        ))}
      </div>

      <p className="flex items-start gap-2 text-[11px] leading-relaxed text-slate-500">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>
          Health is {source.health_scale.metric}, scaled so |z| ={" "}
          {source.health_scale.full_scale_z} reads 0%. Elevated at |z| ≥{" "}
          {source.health_scale.elevated_z}, degraded at |z| ≥{" "}
          {source.health_scale.degraded_z}. Sensors{" "}
          {source.health_scale.excluded_sensors
            .map((s) => s.replace("sensor_", ""))
            .sort((a, b) => Number(a) - Number(b))
            .join(", ")}{" "}
          are excluded — {source.health_scale.excluded_reason}.
        </span>
      </p>
    </div>
  );
}

function ComponentCard({
  component,
  index,
  selected,
  onSelect,
}: {
  component: SupervisorComponent;
  index: number;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const color = COMPONENT_STATUS_COLOR[component.status] ?? COMPONENT_STATUS_COLOR.no_signal;
  const hasHealth = component.health_pct !== null;
  // Two informative readings keep every card the same height whatever its
  // sensor count; the rest are on the detail panel below the grid.
  const readings = component.readings.filter((r) => r.informative).slice(0, 2);

  return (
    <motion.button
      type="button"
      onClick={() => onSelect(component.id)}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index * 0.04, 0.3) }}
      className={`rounded-xl border p-3 text-left transition-colors ${
        selected
          ? "border-cyan-400/60 bg-cyan-500/10"
          : "border-slate-800/60 bg-slate-900/40 hover:border-slate-700"
      }`}
      aria-pressed={selected}
    >
      <div className="mb-1 flex items-start justify-between gap-2">
        <span className="font-semibold text-slate-100">{component.short_label}</span>
        {hasHealth ? (
          <span
            className="rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold"
            style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}
          >
            {component.health_pct!.toFixed(0)}%
          </span>
        ) : (
          <span className="rounded border border-slate-700/60 bg-slate-800/50 px-1.5 py-0.5 font-mono text-[11px] text-slate-500">
            n/a
          </span>
        )}
      </div>

      <div className="mb-2 truncate text-[11px] text-slate-500">{component.name}</div>

      {hasHealth && (
        <div className="mb-2 h-1 overflow-hidden rounded-full bg-slate-800">
          <motion.div
            className="h-full rounded-full"
            style={{ background: color }}
            initial={{ width: 0 }}
            animate={{ width: `${component.health_pct}%` }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          />
        </div>
      )}

      <div className="space-y-0.5 font-mono text-[11px] text-slate-400">
        {readings.length > 0 ? (
          readings.map((reading) => (
            <div key={reading.sensor} className="flex justify-between gap-2">
              <span className="text-slate-500">{reading.symbol}</span>
              <span>
                {reading.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                {reading.unit !== "—" && <span className="text-slate-600"> {reading.unit}</span>}
              </span>
            </div>
          ))
        ) : (
          <div className="text-slate-600">no reading</div>
        )}
        <div className="flex justify-between gap-2">
          <span className="text-slate-500">|z|</span>
          <span>{component.mean_abs_z !== null ? component.mean_abs_z.toFixed(2) : "—"}</span>
        </div>
      </div>

      {component.is_root_cause && (
        <div className="mt-2 flex items-center gap-1 text-[10px] font-semibold tracking-wide text-rose-300">
          <Crosshair className="h-3 w-3" />
          ROOT CAUSE
        </div>
      )}
    </motion.button>
  );
}

/** Full detail for whichever section is selected, below the grid. */
export function SupervisorComponentDetail({
  component,
}: {
  component: SupervisorComponent | null;
}) {
  if (!component) {
    return (
      <p className="py-6 text-center text-sm text-slate-500">
        Select a section to inspect its sensors and inspection focus.
      </p>
    );
  }

  const color = COMPONENT_STATUS_COLOR[component.status] ?? COMPONENT_STATUS_COLOR.no_signal;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold text-slate-100">{component.subsystem_label}</span>
        <span
          className="rounded border px-2 py-0.5 text-xs"
          style={{ background: `${color}1a`, color, borderColor: `${color}55` }}
        >
          {component.status.replace("_", " ")}
        </span>
        {component.contribution_pct !== null && (
          <span className="text-xs text-slate-500">
            {component.contribution_pct.toFixed(1)}% of the engine's total residual deviation
          </span>
        )}
      </div>

      {component.note && <p className="text-xs leading-relaxed text-slate-500">{component.note}</p>}

      {component.readings.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-left text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="pb-1.5 pr-3 font-medium">Sensor</th>
                <th className="pb-1.5 pr-3 font-medium">Measures</th>
                <th className="pb-1.5 pr-3 text-right font-medium">Value</th>
                <th className="pb-1.5 text-right font-medium">Residual z</th>
              </tr>
            </thead>
            <tbody className="font-mono text-slate-300">
              {component.readings.map((reading) => (
                <tr key={reading.sensor} className="border-t border-slate-800/60">
                  <td className="py-1.5 pr-3">
                    {reading.symbol}
                    {!reading.informative && (
                      <span
                        className="ml-1.5 rounded bg-slate-800 px-1 text-[9px] text-slate-500"
                        title="Constant or quantised in C-MAPSS; excluded from the health figure."
                      >
                        excl
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 pr-3 font-sans text-slate-500">{reading.description}</td>
                  <td className="py-1.5 pr-3 text-right">
                    {reading.value.toLocaleString(undefined, { maximumFractionDigits: 3 })}
                    {reading.unit !== "—" && (
                      <span className="text-slate-600"> {reading.unit}</span>
                    )}
                  </td>
                  <td className="py-1.5 text-right">
                    {reading.residual_z !== null ? reading.residual_z.toFixed(3) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {component.inspection_focus && (
        <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-3 text-xs">
          <div className="mb-1 font-semibold text-slate-300">Inspection focus</div>
          <p className="text-slate-400">{component.inspection_focus}</p>
          {component.suggested_action && (
            <p className="mt-1.5 text-slate-500">{component.suggested_action}</p>
          )}
        </div>
      )}
    </div>
  );
}
