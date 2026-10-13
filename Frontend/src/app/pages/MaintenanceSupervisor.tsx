import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Clock,
  FileBadge,
  Gauge,
  Play,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Volume2,
  Zap,
} from "lucide-react";
import { motion } from "motion/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GlassCard } from "../components/GlassCard";
import { MetricCard } from "../components/MetricCard";
import {
  AnimatedNumber,
  Empty,
  ErrorState,
  LiveBadge,
  Loading,
  PageHeader,
  Reveal,
} from "../components/Live";
import { BaselinePanel } from "../components/BaselinePanel";
import {
  SupervisorComponentDetail,
  SupervisorComponentMap,
} from "../components/SupervisorComponentMap";
import {
  useSupervisorAgentView,
  useSupervisorEngines,
  useSupervisorFeedbackHistory,
  useSupervisorFusionSummary,
  useSupervisorRunFusion,
  useSupervisorSubmitFeedback,
  useSupervisorWhatIf,
} from "../../lib/hooks";
import {
  CHART_INK,
  DECISION_COLOR,
  MAGNITUDE_HUE,
  PRIORITY_COLOR,
  RISK_HORIZON_COLOR,
  TOOLTIP_STYLE,
} from "../../lib/chartColors";
import type {
  SupervisorAgentView as AgentView,
  SupervisorRecommendation,
  SupervisorRiskBlock,
  SupervisorRulBlock,
} from "../../lib/types";

const pretty = (value?: string | null) =>
  value ? value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) : "—";

const pct = (value?: number | null, decimals = 1) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(decimals)}%`;

/**
 * Autonomous Maintenance Supervisor Agent.
 *
 * The decision layer of the system: it consumes the RUL, Failure-Risk and
 * Anomaly & Health agents' outputs for one engine, shows the fused decision
 * those three produced, and exposes the evidence behind it — which safety rule
 * fired, which engine section the twin residual implicates, what the smallest
 * operational change that would lower the tier is, and what an engineer decided
 * about it.
 *
 * Everything on this screen is read from the fused decision record. Where a
 * figure cannot be supported by the run — an unaligned residual table, a
 * subsystem with no informative sensor, a projection past the observed history
 * — the panel says so rather than filling the space.
 */
export function MaintenanceSupervisor() {
  const engines = useSupervisorEngines();
  const [selectedEngine, setSelectedEngine] = useState<number | string | null>(null);
  const [cyclesAhead, setCyclesAhead] = useState(0);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);

  // The catalogue is sorted most-urgent-first, so the default selection is the
  // engine that actually needs a decision rather than engine 1.
  const engineId = selectedEngine ?? engines.data?.engines[0]?.engine_id ?? null;

  const view = useSupervisorAgentView(engineId ?? undefined);
  const whatIf = useSupervisorWhatIf(engineId ?? undefined, cyclesAhead);
  const summary = useSupervisorFusionSummary();
  const fusion = useSupervisorRunFusion();
  const history = useSupervisorFeedbackHistory(engineId ?? undefined);

  const data = view.data;

  // Reset the simulator and the section selection when the operator changes
  // unit: a slider position and a highlighted section from the previous engine
  // are meaningless against a different trajectory.
  useEffect(() => {
    setCyclesAhead(0);
    setSelectedComponent(null);
  }, [engineId]);

  if (engines.isLoading && !data) return <Loading label="Loading supervisor decisions…" />;
  if (engines.error) return <ErrorState error={engines.error} />;
  if (view.error && !data) return <ErrorState error={view.error} />;
  if (!engines.data?.engines.length) return <Empty label="No engines have fused decisions yet." />;
  if (!data) return <Loading label="Loading agent view…" />;

  // With the simulator engaged the four headline cards report the projected
  // state, so the operator is never reading a mix of current and simulated
  // numbers across the same row.
  const projecting = cyclesAhead > 0 && Boolean(whatIf.data);
  const risk: SupervisorRiskBlock = projecting ? whatIf.data!.failure_risk : data.failure_risk;
  const rul: SupervisorRulBlock = projecting ? whatIf.data!.rul : data.rul;
  const recommendation: SupervisorRecommendation = projecting
    ? whatIf.data!.recommendation
    : data.recommendation;
  const shownCycle = projecting ? whatIf.data!.target_cycle : data.cycle;

  const activeComponent =
    data.components.find((c) => c.id === selectedComponent) ??
    data.components.find((c) => c.id === data.anomaly.focus_component?.id) ??
    data.components[0] ??
    null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Autonomous Maintenance Supervisor Agent"
        subtitle="Decision fusion over the RUL, Failure-Risk and Anomaly agents, with digital-twin attribution and safety-policy guardrails"
        right={<LiveBadge />}
      />

      <Reveal index={0}>
        <ControlBar
          data={data}
          engines={engines.data.engines}
          engineId={engineId}
          onEngineChange={setSelectedEngine}
          fusion={fusion}
        />
      </Reveal>

      <Reveal index={1}>
        <AgentStrip data={data} />
      </Reveal>

      <Reveal index={2}>
        <WhatIfSimulator
          data={data}
          cyclesAhead={cyclesAhead}
          onChange={setCyclesAhead}
          basis={whatIf.data?.basis ?? null}
          basisNote={whatIf.data?.basis_note ?? null}
          isFetching={whatIf.isFetching}
        />
      </Reveal>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Reveal index={3}>
          <RiskCard risk={risk} projecting={projecting} />
        </Reveal>
        <Reveal index={4}>
          <RulCard rul={rul} projecting={projecting} />
        </Reveal>
        <Reveal index={5}>
          <AnomalyCard data={data} projecting={projecting} />
        </Reveal>
        <Reveal index={6}>
          <ActionCard recommendation={recommendation} projecting={projecting} cycle={shownCycle} />
        </Reveal>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Reveal index={7} className="xl:col-span-2">
          <MetricCard
            title={`Digital Twin Component Attribution — ${data.label}`}
            subtitle={`Per-section twin residual at cycle ${data.cycle}. Click a section for its sensors and inspection focus.`}
          >
            <SupervisorComponentMap
              components={data.components}
              source={data.component_source}
              selectedId={activeComponent?.id ?? null}
              onSelect={setSelectedComponent}
            />
            <div className="mt-4 border-t border-slate-800/60 pt-4">
              <SupervisorComponentDetail component={activeComponent} />
            </div>
          </MetricCard>
        </Reveal>

        <div className="space-y-6">
          <Reveal index={8}>
            <XaiPanel data={data} />
          </Reveal>
          <Reveal index={9}>
            <FeedbackPanel data={data} history={history.data} />
          </Reveal>
        </div>
      </div>

      <Reveal index={10}>
        <CounterfactualPanel data={data} />
      </Reveal>

      <Reveal index={11}>
        <RiskTrajectoryPanel data={data} cyclesAhead={cyclesAhead} />
      </Reveal>

      {fusion.data && (
        <Reveal index={12}>
          <FusionRunPanel run={fusion.data} elapsedMs={fusion.elapsedMs} />
        </Reveal>
      )}

      <Reveal index={13}>
        <FleetPanel summary={summary.data} />
      </Reveal>

      <Reveal index={14}>
        <BaselinePanel />
      </Reveal>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Header controls
// ---------------------------------------------------------------------------

function ControlBar({
  data,
  engines,
  engineId,
  onEngineChange,
  fusion,
}: {
  data: AgentView;
  engines: Array<{ engine_id: number | string; label: string; final_decision: string }>;
  engineId: number | string | null;
  onEngineChange: (id: number | string) => void;
  fusion: ReturnType<typeof useSupervisorRunFusion>;
}) {
  const [speaking, setSpeaking] = useState(false);
  const canSpeak = typeof window !== "undefined" && "speechSynthesis" in window;

  /** Read the decision justification aloud, for a hands-busy walkaround. */
  const speak = () => {
    if (!canSpeak) return;
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(
      `${data.label}. ${data.xai.narrative} Recommended action: ${
        data.recommendation.recommended_action ?? data.recommendation.final_decision
      }`,
    );
    utterance.rate = 0.95;
    utterance.onend = () => setSpeaking(false);
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    setSpeaking(true);
  };

  // Stop any in-flight narration when the panel goes away, so it does not keep
  // talking about an engine that is no longer on screen.
  useEffect(() => {
    return () => {
      if (canSpeak) window.speechSynthesis.cancel();
    };
  }, [canSpeak]);

  const downloadCertificate = async () => {
    const res = await fetch(
      `/api/integration/supervisor/certificate?engine_id=${data.engine_id}&cycle=${data.cycle}`,
    );
    const payload = await res.json();
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${payload.certificate_id ?? "decision-certificate"}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <GlassCard className="flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-2">
        <BrainCircuit className="h-5 w-5 text-cyan-400" />
        <div>
          <div className="text-xs text-slate-500">Decision policy</div>
          <div className="text-sm font-semibold text-slate-200">
            {data.policy.champion_label}
          </div>
        </div>
      </div>

      {/* The reference design labelled this slot "RL PPO Agent (Autonomous)".
          The champion is whatever the manifest names, and no candidate head is
          trained by reinforcement learning, so the regime is stated instead. */}
      <span
        className="rounded border border-slate-700/60 bg-slate-800/40 px-2 py-1 text-[11px] text-slate-400"
        title={data.policy.target_note}
      >
        {data.policy.training_regime}
      </span>

      <span
        className={`rounded border px-2 py-1 text-[11px] ${
          data.policy.continuous_learning_active
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
            : "border-slate-700/60 bg-slate-800/40 text-slate-500"
        }`}
      >
        {data.policy.continuous_learning_active
          ? `Policy adapted from ${data.policy.feedback_entries} verdict${
              data.policy.feedback_entries === 1 ? "" : "s"
            }`
          : "No engineer feedback logged yet"}
      </span>

      <div className="ml-auto flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2 text-xs text-slate-400">
          Unit
          <select
            value={String(engineId ?? "")}
            onChange={(event) => onEngineChange(event.target.value)}
            className="rounded-lg border border-slate-700/60 bg-slate-900/70 px-2 py-1.5 text-sm text-slate-200 outline-none focus:border-cyan-500/60"
          >
            {engines.map((engine) => (
              <option key={String(engine.engine_id)} value={String(engine.engine_id)}>
                {engine.label} · {pretty(engine.final_decision)}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={speak}
          disabled={!canSpeak}
          className="flex items-center gap-1.5 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-300 transition-colors hover:border-slate-600 disabled:opacity-40"
          title={canSpeak ? "Read the decision justification aloud" : "Speech synthesis unavailable"}
        >
          <Volume2 className="h-3.5 w-3.5" />
          {speaking ? "Stop briefing" : "Audio briefing"}
        </button>

        <button
          type="button"
          onClick={downloadCertificate}
          className="flex items-center gap-1.5 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-300 transition-colors hover:border-slate-600"
        >
          <FileBadge className="h-3.5 w-3.5" />
          Decision certificate
        </button>

        <button
          type="button"
          onClick={() => fusion.mutate({ engineId: data.engine_id, cycle: data.cycle })}
          disabled={fusion.isPending}
          className="flex items-center gap-1.5 rounded-lg bg-cyan-500/90 px-3 py-1.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 disabled:opacity-50"
        >
          <Play className="h-3.5 w-3.5" />
          {fusion.isPending ? "Fusing…" : "Run decision fusion"}
        </button>
      </div>
    </GlassCard>
  );
}

// ---------------------------------------------------------------------------
// Upstream agent status
// ---------------------------------------------------------------------------

/**
 * Which upstream agents actually contributed to this decision.
 *
 * "Active" means that agent's own columns carry values on this fused record —
 * not that a process answered a health ping. An agent whose columns are all
 * null did not reach the fusion step for this engine, and the strip says so.
 */
function AgentStrip({ data }: { data: AgentView }) {
  const agents = [data.agents.rul, data.agents.failure_risk, data.agents.anomaly];
  const writtenAt = data.source.decisions_written_at;

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-slate-800/60 bg-slate-900/40 px-4 py-3">
      <div className="flex items-center gap-2">
        <Activity className="h-4 w-4 text-emerald-400" />
        <span className="text-xs text-slate-500">Fused decision record</span>
        <span className="font-mono text-xs text-slate-300">{data.label}</span>
      </div>
      <span className="font-mono text-xs text-slate-500">
        {data.fd_subset} · cycle {data.cycle} of {data.cycle_range.max} ·{" "}
        {data.cycle_range.observed} scored cycles
      </span>
      <div className="ml-auto flex flex-wrap items-center gap-4">
        {agents.map((agent) => (
          <span key={agent.key} className="flex items-center gap-1.5 text-[11px]">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                agent.status === "active" ? "bg-emerald-400" : "bg-slate-600"
              }`}
            />
            <span className="text-slate-400">{agent.name}</span>
            <span
              className={
                agent.status === "active"
                  ? "font-semibold text-emerald-300"
                  : "font-semibold text-slate-500"
              }
              title={`Columns present: ${agent.fields_present.join(", ") || "none"}`}
            >
              {agent.status === "active" ? "ACTIVE" : "NO SIGNAL"}
            </span>
          </span>
        ))}
      </div>
      {writtenAt && (
        <span className="w-full text-[11px] text-slate-600">
          Decisions written {new Date(writtenAt).toLocaleString()} — this screen reads that run, it
          does not re-score.
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// What-if simulator
// ---------------------------------------------------------------------------

/**
 * Step the engine forward and watch the decision move.
 *
 * Within the engine's observed history the projection is not a forecast at all
 * — C-MAPSS units run to failure, so those cycles were actually scored, and the
 * simulator replays what the agents really said. Past the last observed cycle
 * each signal is extended along the slope of its own recent history, and the
 * badge changes to say so.
 */
function WhatIfSimulator({
  data,
  cyclesAhead,
  onChange,
  basis,
  basisNote,
  isFetching,
}: {
  data: AgentView;
  cyclesAhead: number;
  onChange: (value: number) => void;
  basis: string | null;
  basisNote: string | null;
  isFetching: boolean;
}) {
  const observedAhead = Math.max(0, data.cycle_range.max - data.cycle);
  // Always leave room past the observed history so the extrapolated regime is
  // reachable, and always cover the critical crossing when the run recorded one.
  const max = Math.max(observedAhead + 20, data.critical_offset ?? 0, 20);
  const ticks = [0, Math.round(max * 0.25), Math.round(max * 0.5), Math.round(max * 0.75), max];

  return (
    <MetricCard
      title="Prescriptive What-If Scenario Simulator"
      subtitle="Step the engine forward to watch the fused decision and risk trajectory move."
      actions={
        <div className="flex items-center gap-2">
          {basis && (
            <span
              className={`rounded border px-2 py-0.5 text-[11px] ${
                basis === "recorded"
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                  : "border-amber-500/30 bg-amber-500/10 text-amber-300"
              }`}
              title={basisNote ?? undefined}
            >
              {basis === "recorded" ? "Recorded & scored" : "Extrapolated"}
            </span>
          )}
          <span className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 font-mono text-[11px] text-cyan-300">
            Cycle {data.cycle + cyclesAhead} (+{cyclesAhead})
          </span>
        </div>
      }
    >
      <div className="flex items-center gap-3">
        <SlidersHorizontal className="h-4 w-4 shrink-0 text-slate-500" />
        <input
          type="range"
          min={0}
          max={max}
          value={cyclesAhead}
          onChange={(event) => onChange(Number(event.target.value))}
          className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-800 accent-cyan-400"
          aria-label="Cycles to simulate forward"
        />
        {isFetching && <span className="shrink-0 text-[11px] text-slate-500">…</span>}
      </div>

      <div className="mt-2 flex justify-between font-mono text-[11px] text-slate-500">
        {ticks.map((tick) => (
          <span key={tick} className={tick === 0 ? "text-cyan-400" : undefined}>
            {tick === 0 ? `Current (${data.cycle})` : `+${tick}`}
            {tick > observedAhead && tick > 0 && <span className="text-amber-500/70"> ext</span>}
          </span>
        ))}
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
        {basisNote ??
          `Cycles up to +${observedAhead} were observed and scored in the last run; beyond that the state is extrapolated along each signal's own slope.`}
        {data.critical_offset !== null && (
          <>
            {" "}
            This engine's first recorded critical decision is{" "}
            <span className="font-semibold text-rose-300">+{data.critical_offset} cycles</span>{" "}
            away.
          </>
        )}
      </p>
    </MetricCard>
  );
}

// ---------------------------------------------------------------------------
// Headline cards
// ---------------------------------------------------------------------------

function ProjectedTag({ projecting }: { projecting: boolean }) {
  if (!projecting) return null;
  return (
    <span className="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-amber-300">
      SIMULATED
    </span>
  );
}

function RiskCard({ risk, projecting }: { risk: SupervisorRiskBlock; projecting: boolean }) {
  const headline = risk.headline;
  const value = headline.value ?? 0;
  const exceeded = headline.threshold !== null && value >= headline.threshold;
  const others = risk.horizons.filter((h) => h.key !== headline.key);

  return (
    <div className="h-full rounded-xl border border-slate-800/50 bg-gradient-to-br from-slate-900/50 to-slate-900/30 p-5 backdrop-blur-xl">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="text-sm font-medium text-slate-400">
          Failure Risk
          {/* The upstream model's horizons are cycles, not hours. */}
          <span className="ml-1 text-slate-600">({headline.label})</span>
        </div>
        <div className="flex items-center gap-1.5">
          <ProjectedTag projecting={projecting} />
          <ShieldAlert
            className={`h-4 w-4 ${exceeded ? "text-rose-400" : "text-slate-500"}`}
          />
        </div>
      </div>

      <div
        className={`mb-1 text-4xl font-bold ${exceeded ? "text-rose-400" : "text-slate-100"}`}
      >
        <AnimatedNumber value={value * 100} decimals={1} suffix="%" />
      </div>

      <div className="mb-3 flex items-center gap-2 text-xs">
        <span className="text-slate-500">state</span>
        <span className="font-semibold text-slate-300">{pretty(risk.risk_state)}</span>
        {headline.threshold !== null && (
          <span className="text-slate-600">· alert ≥ {pct(headline.threshold, 0)}</span>
        )}
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
        <motion.div
          className="h-full rounded-full"
          style={{ background: exceeded ? "#e11d48" : MAGNITUDE_HUE }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, value * 100)}%` }}
          transition={{ duration: 0.6 }}
        />
      </div>

      <div className="mt-3 flex justify-between font-mono text-[11px] text-slate-500">
        {others.map((horizon) => (
          <span key={horizon.key}>
            {horizon.label}: <span className="text-slate-300">{pct(horizon.value)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function RulCard({ rul, projecting }: { rul: SupervisorRulBlock; projecting: boolean }) {
  const guaranteed = rul.interval.source === "inductive_conformal";
  return (
    <div className="h-full rounded-xl border border-slate-800/50 bg-gradient-to-br from-slate-900/50 to-slate-900/30 p-5 backdrop-blur-xl">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="text-sm font-medium text-slate-400">Remaining Useful Life</div>
        <div className="flex items-center gap-1.5">
          <ProjectedTag projecting={projecting} />
          <Clock className="h-4 w-4 text-slate-500" />
        </div>
      </div>

      <div className="mb-1 flex items-baseline gap-2">
        <span className="text-4xl font-bold text-cyan-300">
          <AnimatedNumber value={rul.predicted_rul ?? 0} decimals={1} />
        </span>
        <span className="text-sm text-slate-500">cycles</span>
      </div>

      <div className="mb-2 text-xs text-slate-500">
        lifecycle <span className="text-slate-300">{pretty(rul.lifecycle_state)}</span>
        {rul.model_used && <span className="text-slate-600"> · {rul.model_used}</span>}
      </div>

      {rul.interval.lower !== null && rul.interval.upper !== null ? (
        <>
          <div className="font-mono text-xs text-slate-400">
            [{rul.interval.lower.toFixed(1)} – {rul.interval.upper.toFixed(1)}]
          </div>
          {/* The interval's provenance is stated because only the conformal one
              carries a distribution-free coverage guarantee. */}
          <div
            className={`mt-1.5 text-[11px] ${
              guaranteed ? "text-emerald-400" : "text-slate-500"
            }`}
            title={rul.interval.label}
          >
            {guaranteed
              ? `Conformal, target coverage ${pct(rul.interval.target_coverage ?? 0.95, 0)}`
              : "Agent interval — no conformal calibration on disk"}
          </div>
        </>
      ) : (
        <div className="text-[11px] text-slate-500">{rul.interval.label}</div>
      )}
    </div>
  );
}

function AnomalyCard({ data, projecting }: { data: AgentView; projecting: boolean }) {
  const anomaly = data.anomaly;
  const severe = ["high", "critical"].includes((anomaly.severity ?? "").toLowerCase());
  const focus = anomaly.focus_component;

  return (
    <div className="h-full rounded-xl border border-slate-800/50 bg-gradient-to-br from-slate-900/50 to-slate-900/30 p-5 backdrop-blur-xl">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="text-sm font-medium text-slate-400">Explainable Anomaly</div>
        <div className="flex items-center gap-1.5">
          {/* Anomaly is never projected: the what-if replays the fused record,
              and past the observed history there is no anomaly score to show. */}
          {projecting && (
            <span className="rounded border border-slate-700/60 bg-slate-800/40 px-1.5 py-0.5 text-[10px] text-slate-500">
              AT CYCLE {data.cycle}
            </span>
          )}
          <Sparkles className={`h-4 w-4 ${severe ? "text-rose-400" : "text-slate-500"}`} />
        </div>
      </div>

      <div className={`mb-1 text-4xl font-bold ${severe ? "text-rose-400" : "text-slate-100"}`}>
        <AnimatedNumber value={anomaly.anomaly_score ?? 0} decimals={3} />
      </div>

      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded border border-slate-700/60 bg-slate-800/40 px-1.5 py-0.5 text-slate-300">
          {pretty(anomaly.severity)}
        </span>
        <span className="text-slate-500">
          health <span className="text-slate-300">{anomaly.health_index?.toFixed(1) ?? "—"}</span>
        </span>
        <span className="text-slate-500">{pretty(anomaly.health_state)}</span>
      </div>

      {focus ? (
        <div className="text-[11px] leading-relaxed text-slate-500">
          Focus:{" "}
          <span className="font-semibold text-slate-300">
            {focus.name} ({focus.short_label})
          </span>
          {focus.top_sensor_label && (
            <>
              {" "}
              · driven by <span className="font-mono">{focus.top_sensor_label}</span>
            </>
          )}
          <div className="text-slate-600">localised by {focus.basis}</div>
        </div>
      ) : (
        <div className="text-[11px] text-slate-500">
          No section could be localised for this cycle.
        </div>
      )}
    </div>
  );
}

function ActionCard({
  recommendation,
  projecting,
  cycle,
}: {
  recommendation: SupervisorRecommendation;
  projecting: boolean;
  cycle: number;
}) {
  const color = DECISION_COLOR[recommendation.final_decision] ?? "#64748b";
  const priorityColor = PRIORITY_COLOR[recommendation.priority ?? ""] ?? "#64748b";

  return (
    <div
      className="h-full rounded-xl border p-5 backdrop-blur-xl"
      style={{ borderColor: `${color}44`, background: `linear-gradient(to bottom right, ${color}14, ${color}05)` }}
    >
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-medium text-slate-400">
          <Zap className="h-4 w-4" style={{ color }} />
          Recommended Action
        </div>
        <div className="flex items-center gap-1.5">
          <ProjectedTag projecting={projecting} />
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ background: `${priorityColor}22`, color: priorityColor }}
          >
            {(recommendation.priority ?? "—").toUpperCase()}
          </span>
        </div>
      </div>

      <div className="mb-1 text-xl font-bold leading-tight" style={{ color }}>
        {recommendation.final_decision_label ?? pretty(recommendation.final_decision)}
      </div>

      <div className="mb-2 font-mono text-[11px] text-slate-500">
        Action code #{recommendation.action_code} · cycle {cycle} ·{" "}
        {pretty(recommendation.urgency)}
      </div>

      <p className="mb-3 text-xs leading-relaxed text-slate-400">
        {recommendation.recommended_action}
      </p>

      {recommendation.confidence !== null ? (
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-slate-500">Fusion confidence</span>
          <span className="font-mono font-semibold text-slate-200">
            {pct(recommendation.confidence)}
            <span className="ml-1 text-slate-500">({recommendation.confidence_level})</span>
          </span>
        </div>
      ) : (
        <div className="text-[11px] text-slate-500">{recommendation.basis}</div>
      )}

      {recommendation.requires_human_review && (
        <div className="mt-2 rounded border border-amber-500/25 bg-amber-500/5 px-2 py-1 text-[11px] text-amber-300">
          Human review required — {recommendation.review_reason}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Explainability
// ---------------------------------------------------------------------------

/**
 * Why this decision, in three registers: the narrative, how much each upstream
 * agent pushed for it, and which safety rules fired.
 *
 * The contribution meters use one hue on purpose. They are three shares of a
 * single quantity, and the agent is named on each row, so a second hue would
 * imply a distinction that is not there.
 */
function XaiPanel({ data }: { data: AgentView }) {
  const contributions = [
    { key: "rul", label: "RUL Prognostics", value: data.xai.upstream_contributions.rul },
    { key: "risk", label: "Failure Risk", value: data.xai.upstream_contributions.failure_risk },
    { key: "anomaly", label: "Anomaly & Health", value: data.xai.upstream_contributions.anomaly },
  ];

  return (
    <MetricCard
      title="Explainable Decision Justification"
      subtitle="Narrative, upstream pressure share, and the safety rules as evaluated."
    >
      <p className="mb-4 text-sm italic leading-relaxed text-slate-300">
        “{data.xai.narrative}”
      </p>

      <div className="mb-4 space-y-2">
        <div className="text-xs font-semibold text-slate-400">Upstream decision pressure</div>
        {contributions.map((item) => (
          <div key={item.key}>
            <div className="mb-0.5 flex justify-between text-[11px]">
              <span className="text-slate-400">{item.label}</span>
              <span className="font-mono text-slate-300">{item.value.toFixed(1)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
              <motion.div
                className="h-full rounded-full"
                style={{ background: MAGNITUDE_HUE }}
                initial={{ width: 0 }}
                animate={{ width: `${item.value}%` }}
                transition={{ duration: 0.6 }}
              />
            </div>
          </div>
        ))}
        <p className="text-[11px] text-slate-600">
          Normalised shares of RUL urgency (1 − RUL/150), 30-cycle risk and anomaly score — the
          same decomposition the Supervisor's own explainer uses.
        </p>
      </div>

      <div className="space-y-1.5">
        <div className="text-xs font-semibold text-slate-400">Safety policy guardrails</div>
        {data.xai.policy_drivers.map((driver) => (
          <div
            key={driver.rule}
            className={`flex flex-wrap items-center justify-between gap-2 rounded border px-2 py-1.5 text-[11px] ${
              driver.fired
                ? "border-rose-500/25 bg-rose-500/5"
                : "border-slate-800/60 bg-slate-900/30"
            }`}
          >
            <span className={driver.fired ? "font-semibold text-rose-300" : "text-slate-400"}>
              {driver.rule}
            </span>
            <span className="font-mono text-slate-500">
              {driver.expression} · now{" "}
              <span className="text-slate-300">
                {typeof driver.value === "number" ? driver.value.toFixed(3) : driver.value ?? "—"}
              </span>
            </span>
            <span
              className={`rounded px-1.5 py-0.5 ${
                driver.fired
                  ? "bg-rose-500/15 text-rose-300"
                  : "bg-slate-800/60 text-slate-500"
              }`}
            >
              {driver.fired ? `forces ${pretty(driver.forces)}` : "clear"}
            </span>
          </div>
        ))}
      </div>

      {data.recommendation.override_applied && (
        <div className="mt-3 rounded border border-amber-500/25 bg-amber-500/5 p-2 text-[11px] text-amber-200">
          Safety policy overrode the model:{" "}
          <span className="font-semibold">
            {pretty(data.recommendation.raw_model_decision)}
          </span>{" "}
          → <span className="font-semibold">{pretty(data.recommendation.final_decision)}</span> (
          {data.recommendation.override_reason})
        </div>
      )}
    </MetricCard>
  );
}

// ---------------------------------------------------------------------------
// Human feedback
// ---------------------------------------------------------------------------

/**
 * The engineer's verdict on the agent's decision.
 *
 * This is the entry point to the Supervisor's continuous-learning loop: the
 * component always had a feedback store and an adaptive-policy pass, and until
 * now nothing in the product called either, so the loop had never run on real
 * input. Submitting logs the verdict and re-derives the policy. It does not
 * retrain a model — retraining stays a deliberate offline action.
 */
function FeedbackPanel({
  data,
  history,
}: {
  data: AgentView;
  history: ReturnType<typeof useSupervisorFeedbackHistory>["data"];
}) {
  const submit = useSupervisorSubmitFeedback();
  const [decision, setDecision] = useState(data.recommendation.final_decision);
  const [comments, setComments] = useState("");

  // Track the agent's decision when the operator changes engine or cycle, so
  // the default is always "accept what the agent said" rather than a stale
  // choice carried over from another unit.
  useEffect(() => {
    setDecision(data.recommendation.final_decision);
    setComments("");
  }, [data.engine_id, data.cycle, data.recommendation.final_decision]);

  const overriding = decision !== data.recommendation.final_decision;

  return (
    <MetricCard
      title="Engineer Feedback & Policy Adaptation"
      subtitle="Log an accept or override; the adaptive-policy pass re-runs over the full history."
    >
      <div className="space-y-3">
        <div>
          <label className="mb-1 block text-xs text-slate-400">Engineer decision</label>
          <select
            value={decision}
            onChange={(event) => setDecision(event.target.value)}
            className="w-full rounded-lg border border-slate-700/60 bg-slate-900/70 px-3 py-2 text-sm text-slate-200 outline-none focus:border-cyan-500/60"
          >
            {data.policy.decision_classes.map((option) => (
              <option key={option} value={option}>
                {option === data.recommendation.final_decision ? "Accept: " : "Override to: "}
                {pretty(option)}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs text-slate-400">Notes</label>
          <textarea
            value={comments}
            onChange={(event) => setComments(event.target.value)}
            rows={2}
            placeholder={`Rationale for ${data.label} at cycle ${data.cycle}…`}
            className="w-full resize-none rounded-lg border border-slate-700/60 bg-slate-900/70 px-3 py-2 text-sm text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-500/60"
          />
        </div>

        <button
          type="button"
          disabled={submit.isPending}
          onClick={() =>
            submit.mutate({
              engine_id: Number(data.engine_id),
              cycle: data.cycle,
              engineer_decision: decision,
              comments,
            })
          }
          className={`w-full rounded-lg px-3 py-2 text-sm font-semibold transition-colors disabled:opacity-50 ${
            overriding
              ? "bg-amber-500/90 text-slate-950 hover:bg-amber-400"
              : "bg-cyan-500/90 text-slate-950 hover:bg-cyan-400"
          }`}
        >
          {submit.isPending
            ? "Recording…"
            : overriding
            ? "Submit override & adapt policy"
            : "Confirm decision & adapt policy"}
        </button>

        {submit.data && (
          <div className="rounded border border-emerald-500/25 bg-emerald-500/5 p-2 text-[11px] text-emerald-200">
            Logged {submit.data.agreed_with_agent ? "agreement" : "override"} —{" "}
            {submit.data.total_feedback_entries} verdict
            {submit.data.total_feedback_entries === 1 ? "" : "s"} on file. {submit.data.note}
          </div>
        )}
        {submit.error && (
          <div className="rounded border border-rose-500/25 bg-rose-500/5 p-2 text-[11px] text-rose-200">
            {(submit.error as Error).message}
          </div>
        )}

        {history && history.total > 0 && (
          <div className="border-t border-slate-800/60 pt-3">
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-slate-400">
                {history.total} verdict{history.total === 1 ? "" : "s"} for this engine
              </span>
              {history.agreement_pct !== null && (
                <span className="text-slate-500">{history.agreement_pct}% agreement</span>
              )}
            </div>
            <div className="space-y-1">
              {history.records.slice(0, 3).map((record) => (
                <div
                  key={`${record.sample_id}-${record.timestamp}`}
                  className="flex items-center justify-between gap-2 text-[11px]"
                >
                  <span className="text-slate-500">{record.engineer_id}</span>
                  <span className={record.agreed ? "text-emerald-300" : "text-amber-300"}>
                    {record.agreed ? "accepted" : `→ ${pretty(record.engineer_decision)}`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </MetricCard>
  );
}

// ---------------------------------------------------------------------------
// Counterfactual
// ---------------------------------------------------------------------------

/**
 * The smallest operational change that would lower the maintenance tier.
 *
 * The two halves of this are shown separately because only one is derived from
 * the system's own evidence. The binding constraints and the required margin
 * come from inverting the safety policy's published thresholds — exact, no
 * modelling assumption. The derate that would deliver that margin rests on an
 * assumed linear sensitivity, since C-MAPSS has no controllable pressure input
 * from which one could be fitted. The panel labels which is which.
 */
function CounterfactualPanel({ data }: { data: AgentView }) {
  const counterfactual = data.counterfactual;

  return (
    <MetricCard
      title="Actionable Counterfactual — Operational Optimisation"
      subtitle={
        counterfactual.target_decision
          ? `What would move ${pretty(counterfactual.current_decision)} down to ${pretty(
              counterfactual.target_decision,
            )}`
          : "No lower tier is reachable from this state"
      }
      actions={
        <span
          className={`rounded border px-2 py-0.5 text-[11px] ${
            counterfactual.feasible
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
              : "border-slate-700/60 bg-slate-800/40 text-slate-400"
          }`}
        >
          {counterfactual.feasible ? "Feasible" : pretty(counterfactual.blocked_reason ?? "n/a")}
        </span>
      }
    >
      <p className="mb-4 text-sm leading-relaxed text-slate-300">
        {counterfactual.actionable_statement}
      </p>

      {counterfactual.binding_constraints.length > 0 && (
        <div className="mb-4">
          <div className="mb-1.5 text-xs font-semibold text-slate-400">
            Binding constraints — exact, read off the safety policy's own thresholds
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  <th className="pb-1.5 pr-3 font-medium">Constraint</th>
                  <th className="pb-1.5 pr-3 font-medium">Must satisfy</th>
                  <th className="pb-1.5 pr-3 text-right font-medium">Now</th>
                  <th className="pb-1.5 text-right font-medium">Shortfall</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                {counterfactual.binding_constraints.map((constraint) => (
                  <tr key={constraint.name} className="border-t border-slate-800/60">
                    <td className="py-1.5 pr-3">{constraint.name}</td>
                    <td className="py-1.5 pr-3 font-mono text-slate-400">
                      {constraint.expression}
                    </td>
                    <td className="py-1.5 pr-3 text-right font-mono">
                      {constraint.current_value?.toFixed(3) ?? "—"}
                    </td>
                    <td className="py-1.5 text-right font-mono text-amber-300">
                      {constraint.shortfall.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {counterfactual.feasible && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="HPC pressure" value={`−${counterfactual.hpc_pressure_reduction_pct}%`} />
          <Stat label="Thrust derate" value={`−${counterfactual.cruise_thrust_reduction_pct}%`} />
          <Stat
            label="RUL"
            value={`${counterfactual.original_rul?.toFixed(1) ?? "—"} → ${
              counterfactual.projected_rul?.toFixed(1) ?? "—"
            }`}
          />
          <Stat
            label="30-cycle risk"
            value={`${pct(counterfactual.original_risk_30, 0)} → ${pct(
              counterfactual.projected_risk_30,
              0,
            )}`}
          />
        </div>
      )}

      {counterfactual.lever_is_assumption && counterfactual.feasible && (
        <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
          <span className="font-semibold text-amber-400/90">Assumption:</span> the derate figures
          translate the exact required margin above through a linear sensitivity of{" "}
          {counterfactual.assumptions.rul_cycles_per_pct_pressure} RUL cycles and{" "}
          {counterfactual.assumptions.risk_30_per_pct_pressure} of 30-cycle risk per 1% pressure
          reduction. Those constants are not fitted — C-MAPSS records no controllable pressure
          input and no counterfactual trajectories, so no derate response can be estimated from
          it. Treat the lever as indicative and the constraints as authoritative.
        </p>
      )}
    </MetricCard>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-2.5">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="font-mono text-sm font-semibold text-slate-200">{value}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

/**
 * Multi-horizon failure-risk evolution across the engine's whole life.
 *
 * One y-axis, probability only. The three horizons share a sequential cyan ramp
 * because they are ordered rather than categorical, and each also carries its
 * own dash pattern so the series stay separable without colour. The alert
 * threshold is a recessive dashed reference line, and the simulator's position
 * is marked so the slider and the chart read as one control.
 */
function RiskTrajectoryPanel({ data, cyclesAhead }: { data: AgentView; cyclesAhead: number }) {
  const points = useMemo(
    () => data.trajectory.filter((p) => p.risk_30 !== null),
    [data.trajectory],
  );
  const threshold = points.at(-1)?.threshold_30 ?? null;

  if (!points.length) return null;

  return (
    <MetricCard
      title="Multi-Horizon Failure Risk Trajectory"
      subtitle={`${data.label} · ${points.length} scored cycles · probability of failure within 10, 30 and 50 cycles`}
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={points} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_INK.grid} vertical={false} />
          <XAxis
            dataKey="cycle"
            stroke={CHART_INK.axis}
            tick={{ fontSize: 11 }}
            label={{ value: "cycle", position: "insideBottomRight", fill: CHART_INK.axis, fontSize: 11 }}
          />
          <YAxis
            stroke={CHART_INK.axis}
            tick={{ fontSize: 11 }}
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            width={48}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(value: number, name: string) => [pct(value), name]}
            labelFormatter={(label) => `Cycle ${label}`}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {threshold !== null && (
            <ReferenceLine
              y={threshold}
              stroke={CHART_INK.reference}
              strokeDasharray="6 4"
              label={{
                value: `alert ${Math.round(threshold * 100)}%`,
                fill: CHART_INK.axis,
                fontSize: 10,
                position: "insideTopLeft",
              }}
            />
          )}
          <ReferenceLine
            x={data.cycle + cyclesAhead}
            stroke="#22d3ee"
            strokeDasharray="4 4"
            label={{ value: "now", fill: "#22d3ee", fontSize: 10, position: "top" }}
          />
          <Line
            type="monotone"
            dataKey="risk_50"
            name="50 cycles"
            stroke={RISK_HORIZON_COLOR.risk_50}
            strokeWidth={2}
            dot={false}
            animationDuration={800}
          />
          <Line
            type="monotone"
            dataKey="risk_30"
            name="30 cycles"
            stroke={RISK_HORIZON_COLOR.risk_30}
            strokeWidth={2}
            strokeDasharray="6 3"
            dot={false}
            animationDuration={800}
          />
          <Line
            type="monotone"
            dataKey="risk_10"
            name="10 cycles"
            stroke={RISK_HORIZON_COLOR.risk_10}
            strokeWidth={2}
            strokeDasharray="2 3"
            dot={false}
            animationDuration={800}
          />
        </LineChart>
      </ResponsiveContainer>
    </MetricCard>
  );
}

// ---------------------------------------------------------------------------
// Live fusion result
// ---------------------------------------------------------------------------

function FusionRunPanel({
  run,
  elapsedMs,
}: {
  run: NonNullable<ReturnType<typeof useSupervisorRunFusion>["data"]>;
  elapsedMs: number | null;
}) {
  return (
    <GlassCard variant={run.matches_stored ? "primary" : "warning"}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="mb-1 flex items-center gap-2 font-semibold text-slate-200">
            <Gauge className="h-4 w-4 text-cyan-400" />
            Decision fusion executed live
          </h3>
          <p className="text-sm text-slate-300">
            Re-fused {run.rows_fused.toLocaleString()} cycles in {run.latency_ms} ms server-side
            {elapsedMs !== null && ` (${elapsedMs} ms round trip)`}. Live result{" "}
            <span className="font-semibold text-cyan-300">
              {pretty(run.live.final_decision)}
            </span>{" "}
            {run.matches_stored ? "matches" : "differs from"} the stored{" "}
            <span className="font-semibold">{pretty(run.stored.final_decision)}</span>.
          </p>
          <p className="mt-1.5 text-xs text-slate-500">{run.note}</p>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-500">Fusion confidence</div>
          <div className="font-mono text-lg font-semibold text-slate-200">
            {pct(run.live.confidence)}
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

// ---------------------------------------------------------------------------
// Fleet context
// ---------------------------------------------------------------------------

function FleetPanel({
  summary,
}: {
  summary: ReturnType<typeof useSupervisorFusionSummary>["data"];
}) {
  if (!summary) return null;

  const decisions = Object.entries(summary.decision_counts)
    .map(([key, value]) => ({
      name: pretty(key),
      key,
      value,
      color: DECISION_COLOR[key] ?? "#64748b",
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <MetricCard
        title="Fleet Decision Distribution"
        subtitle={`${summary.engines} engines at their latest cycle, from ${summary.rows_total.toLocaleString()} fused rows`}
        className="lg:col-span-2"
      >
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={decisions} layout="vertical" margin={{ left: 60, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_INK.grid} horizontal={false} />
            <XAxis type="number" stroke={CHART_INK.axis} tick={{ fontSize: 11 }} />
            <YAxis
              dataKey="name"
              type="category"
              stroke={CHART_INK.axis}
              width={150}
              tick={{ fontSize: 11 }}
            />
            <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#1e293b55" }} />
            <Bar dataKey="value" name="engines" radius={[0, 4, 4, 0]} animationDuration={800}>
              {decisions.map((entry) => (
                <Cell key={entry.key} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </MetricCard>

      <MetricCard title="Fusion Provenance" subtitle="What the last run's decisions are made of">
        <div className="space-y-3 text-sm">
          <Row
            label="Model / final agreement"
            value={`${summary.model_final_agreement.toFixed(1)}%`}
          />
          <Row label="Safety-policy overrides" value={String(summary.override_count)} />
          <Row
            label="Mean confidence"
            value={summary.mean_confidence !== null ? pct(summary.mean_confidence) : "—"}
          />
          {Object.entries(summary.override_reasons).map(([reason, count]) => (
            <div key={reason} className="flex justify-between text-[11px] text-slate-500">
              <span>{reason}</span>
              <span className="font-mono">{count}</span>
            </div>
          ))}
          <p className="border-t border-slate-800/60 pt-3 text-[11px] leading-relaxed text-slate-500">
            The champion's training target is a deterministic rule over RUL, risk and anomaly
            severity, so high model/final agreement is expected by construction — it shows the
            model reproducing a known policy, not independent predictive skill.
          </p>
        </div>
      </MetricCard>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono font-semibold text-slate-200">{value}</span>
    </div>
  );
}
