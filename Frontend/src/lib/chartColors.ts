/**
 * Validated chart colours for the dark (#0f172a) chart surface.
 *
 * These are not taste. Every set here was checked with the dataviz validator
 * (`validate_palette.js --mode dark --surface "#0f172a"`) against the six
 * checks — lightness band, chroma floor, CVD separation, normal-vision
 * separation, and contrast — and the results are recorded below. Re-run it
 * before introducing any new categorical set; do not eyeball ΔE.
 *
 * The pairing this replaces put amber `#f59e0b` directly beside orange
 * `#f97316` in the decision bars and the health donut. Measured, that pair is
 * ΔE 9.6 for normal vision (the floor is 15) and 6.2 under deuteranopia — two
 * categories rendered as effectively one colour. Never reintroduce it.
 */

/**
 * The five-step maintenance-decision ladder, least to most severe.
 *
 * Ordinal, so lightness varies deliberately as a second channel alongside hue:
 * the validator's "lightness band" check is expected to fail for that reason
 * and only that reason. Everything else passes — worst adjacent pair is
 * `#22d3ee`↔`#10b981` at ΔE 15.6 normal vision, `#e11d48`↔`#f97316` at ΔE 13.3
 * under deuteranopia, all five ≥ 3:1 against the surface.
 */
export const DECISION_COLOR: Record<string, string> = {
  continue_operation: "#10b981",
  monitor_closely: "#22d3ee",
  schedule_inspection: "#facc15",
  schedule_maintenance: "#f97316",
  immediate_maintenance: "#e11d48",
};

/** Operational priority, drawn from the same ordinal ramp. */
export const PRIORITY_COLOR: Record<string, string> = {
  Low: "#10b981",
  Medium: "#22d3ee",
  High: "#f97316",
  Critical: "#e11d48",
};

/**
 * Subsystem condition from the twin residual.
 *
 * Status colours are reserved: they never double as "series 4", and they always
 * ship with a text label rather than carrying meaning by colour alone.
 * `no_signal` is deliberately neutral — it is an absence of evidence, not a
 * fourth severity.
 */
export const COMPONENT_STATUS_COLOR: Record<string, string> = {
  nominal: "#10b981",
  elevated: "#facc15",
  degraded: "#e11d48",
  no_signal: "#475569",
};

/**
 * The three failure-risk horizons, as a sequential cyan ramp.
 *
 * Sequential rather than categorical because the horizons are ordered
 * (10 → 30 → 50 cycles), so lightness carries the ordering: on a dark surface
 * the longest horizon is the lightest and sits topmost, which is also where its
 * line falls. Validated at ΔE 20.8 normal vision and 19.1 under protanopia for
 * the worst adjacent pair, all three ≥ 3:1 on the surface. Lightness-band and
 * chroma-floor failures are inherent to a sequential ramp and expected here.
 *
 * Stroke width and dash pattern are applied as a second channel so the series
 * are separable without colour at all.
 */
export const RISK_HORIZON_COLOR: Record<string, string> = {
  risk_10: "#0369a1",
  risk_30: "#06b6d4",
  risk_50: "#a5f3fc",
};

/**
 * Single hue for magnitude-only marks — counts, histograms, meters.
 *
 * Hue carries no information on those, so varying it would imply a distinction
 * that does not exist.
 */
export const MAGNITUDE_HUE = "#22d3ee";

/** Recessive grid, axis and reference-line inks. */
export const CHART_INK = {
  grid: "#1e293b",
  axis: "#64748b",
  reference: "#64748b",
  surface: "#0f172a",
} as const;

/** Shared Recharts tooltip chrome, so every panel reads as one system. */
export const TOOLTIP_STYLE = {
  background: "#0f172a",
  border: "1px solid #1e293b",
  borderRadius: 8,
  fontSize: 12,
} as const;
