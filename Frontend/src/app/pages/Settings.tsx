import { Activity, Database, Server, SlidersHorizontal } from "lucide-react";
import { motion } from "motion/react";
import { GlassCard } from "../components/GlassCard";
import { KPICard } from "../components/KPICard";
import { MetricCard } from "../components/MetricCard";
import { AnimatedNumber, ErrorState, LiveBadge, Loading, PageHeader, Reveal } from "../components/Live";
import {
  useArtifactStatus,
  useFeatureRouting,
  useFeedbackSummary,
  useFeedbackThresholds,
  useHealthCheck,
} from "../../lib/hooks";
import {
  REFRESH_CHOICES,
  START_PAGES,
  resetPreferences,
  setPreferences,
  usePreferences,
} from "../../lib/preferences";

function JsonPanel({ title, data }: { title: string; data: unknown }) {
  return (
    <MetricCard title={title}>
      <pre className="max-h-80 overflow-auto text-xs leading-relaxed text-slate-400">
        {data ? JSON.stringify(data, null, 2) : "—"}
      </pre>
    </MetricCard>
  );
}

export function Settings() {
  const prefs = usePreferences();
  const health = useHealthCheck();
  const thresholds = useFeedbackThresholds();
  const feedback = useFeedbackSummary();
  const routing = useFeatureRouting();
  const artifacts = useArtifactStatus();

  if (health.isLoading) return <Loading label="Checking backend…" />;

  const online = !health.error && !!health.data;
  const fb = feedback.data as Record<string, unknown> | undefined;
  const total = Number(fb?.total ?? 0);

  return (
    <div className="space-y-6">
      <PageHeader
        title="System Settings"
        subtitle="Display preferences, backend status, adaptive thresholds and the operator feedback loop"
        right={online ? <LiveBadge label="ONLINE" /> : undefined}
      />

      {health.error && <ErrorState error={health.error} />}

      {/*
        Display and pacing preferences only. Nothing on this screen can change a
        model, a threshold or what the pipeline computes -- the refresh interval
        governs how often the browser re-reads results that already exist. The
        distinction matters because the dashboard is reachable by anyone.
      */}
      <Reveal>
        <MetricCard
          title="Preferences"
          subtitle="Stored in this browser only · applied immediately"
          actions={
            <button
              onClick={resetPreferences}
              className="rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-xs text-slate-300 transition-colors hover:bg-slate-700/60"
            >
              Reset to defaults
            </button>
          }
        >
          <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
            <label className="block">
              <span className="text-xs text-slate-400">Auto-refresh</span>
              <select
                value={prefs.refreshSeconds}
                onChange={(e) => setPreferences({ refreshSeconds: Number(e.target.value) })}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500"
              >
                {REFRESH_CHOICES.map((sec) => (
                  <option key={sec} value={sec}>
                    {sec === 0 ? "Off — refresh manually" : `Every ${sec}s`}
                  </option>
                ))}
              </select>
              <span className="mt-1 block text-[11px] leading-snug text-slate-500">
                How often live panels re-read the backend. Turning it off leaves the
                topbar refresh button as the only trigger.
              </span>
            </label>

            <label className="block">
              <span className="text-xs text-slate-400">Start page</span>
              <select
                value={prefs.startPage}
                onChange={(e) => setPreferences({ startPage: e.target.value })}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500"
              >
                {START_PAGES.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
              <span className="mt-1 block text-[11px] leading-snug text-slate-500">
                Opened on a cold start. The sidebar still navigates normally.
              </span>
            </label>

            <div>
              <span className="text-xs text-slate-400">Motion</span>
              <button
                onClick={() => setPreferences({ reduceMotion: !prefs.reduceMotion })}
                role="switch"
                aria-checked={prefs.reduceMotion}
                className={`mt-1 flex w-full items-center justify-between rounded-lg border px-3 py-2 text-sm transition-colors ${
                  prefs.reduceMotion
                    ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
                    : "border-slate-700 bg-slate-950/60 text-slate-300"
                }`}
              >
                <span>{prefs.reduceMotion ? "Reduced" : "Full animation"}</span>
                <span
                  className={`flex h-5 w-9 items-center rounded-full p-0.5 transition-colors ${
                    prefs.reduceMotion ? "bg-cyan-500/60" : "bg-slate-700"
                  }`}
                >
                  <span
                    className={`h-4 w-4 rounded-full bg-white transition-transform ${
                      prefs.reduceMotion ? "translate-x-4" : ""
                    }`}
                  />
                </span>
              </button>
              <span className="mt-1 block text-[11px] leading-snug text-slate-500">
                Suppresses entrance and chart animation. Your OS "reduce motion"
                setting is honoured either way.
              </span>
            </div>
          </div>
        </MetricCard>
      </Reveal>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Reveal index={0}>
          <KPICard
            title="Backend"
            value={<span className="text-2xl">{online ? "Online" : "Offline"}</span>}
            icon={Server}
            status={online ? "success" : "danger"}
            subtitle="127.0.0.1:8000"
          />
        </Reveal>
        <Reveal index={1}>
          <KPICard title="Feedback Records" value={<AnimatedNumber value={total} />} icon={Activity} status="info" />
        </Reveal>
        <Reveal index={2}>
          <KPICard
            title="Routed Components"
            value={<AnimatedNumber value={Object.keys(routing.data?.routes ?? {}).length} />}
            icon={Database}
            status="neutral"
          />
        </Reveal>
        <Reveal index={3}>
          <KPICard
            title="Required Columns"
            value={<AnimatedNumber value={routing.data?.required_upload_columns.length ?? 0} />}
            icon={SlidersHorizontal}
            status="neutral"
          />
        </Reveal>
      </div>

      <Reveal index={4}>
        <GlassCard variant="primary">
          <h3 className="mb-2 font-semibold text-slate-200">Retraining is offline-only</h3>
          <p className="text-sm text-slate-300">
            Every API path — including uploads and the full pipeline — runs strictly in inference mode. Uploading a
            file loads the persisted twins, scalers, detectors and calibrations; it never refits them and never
            rewrites the sensor or feature policy. Retraining is a deliberate action run from the pipeline scripts on
            the server.
          </p>
        </GlassCard>
      </Reveal>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Reveal index={5}>
          <JsonPanel title="Adaptive Thresholds" data={thresholds.data} />
        </Reveal>
        <Reveal index={6}>
          <JsonPanel title="Feedback Summary" data={feedback.data} />
        </Reveal>
      </div>

      <Reveal index={7}>
        <MetricCard title="Feature Routing">
          {routing.data ? (
            <div className="space-y-3">
              {Object.entries(routing.data.routes).map(([name, r], i) => (
                <motion.div
                  key={name}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className="rounded-lg border border-slate-800/60 bg-slate-900/40 p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-medium text-slate-200">{name.replace(/_/g, " ")}</span>
                    <span className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-xs text-cyan-300">
                      {r.features.length} features
                    </span>
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    identity <code className="rounded bg-slate-800/60 px-1">{r.id_column}</code> · {r.scaling}
                  </div>
                </motion.div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">Routing contract unavailable.</p>
          )}
        </MetricCard>
      </Reveal>

      {artifacts.data && (
        <Reveal index={8}>
          <JsonPanel title="Predictive Maintenance Artifacts" data={artifacts.data} />
        </Reveal>
      )}
    </div>
  );
}
