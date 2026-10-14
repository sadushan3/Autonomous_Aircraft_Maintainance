import { Activity, GaugeCircle, ShieldQuestion, UserCheck } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { DataTable } from "../components/DataTable";
import { GlassCard } from "../components/GlassCard";
import { KPICard } from "../components/KPICard";
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
import { useFleetSummary } from "../../lib/hooks";

export function Confidence() {
  const { summary, latest, isLoading, error } = useFleetSummary();

  if (isLoading) return <Loading label="Loading confidence & uncertainty…" />;
  if (error) return <ErrorState error={error} />;
  if (!summary) return <Empty />;

  const conf = latest
    .map((r) => r.confidence_score)
    .filter((n): n is number => n !== null);

  // Histogram over the observed range rather than [0,1]: the confidence ceiling
  // is capped by the fixed conformal width penalty, so a [0,1] axis would leave
  // most of the chart empty and hide the actual spread.
  const bins = 12;
  const lo = Math.min(...conf, 0);
  const hi = Math.max(...conf, 1);
  const step = (hi - lo) / bins || 1;
  const hist = Array.from({ length: bins }, (_, i) => ({
    range: `${(lo + i * step).toFixed(2)}`,
    count: conf.filter((c) => c >= lo + i * step && c < lo + (i + 1) * step).length,
    mid: lo + (i + 0.5) * step,
  }));

  const levelCounts: Record<string, number> = {};
  for (const r of latest) {
    const k = String(r.confidence_level ?? "unknown");
    levelCounts[k] = (levelCounts[k] ?? 0) + 1;
  }

  const scatter = latest
    .filter((r) => r.confidence_score !== null && r.entropy_score !== null)
    .map((r) => ({
      conf: Number((r.confidence_score! * 100).toFixed(2)),
      entropy: Number(r.entropy_score!.toFixed(4)),
      z: 60,
    }));

  const reviewPct = latest.length ? (summary.reviewCount / latest.length) * 100 : 0;
  const allLow = (levelCounts["low"] ?? 0) === latest.length && latest.length > 0;
  const confMax = conf.length ? Math.max(...conf) : 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Confidence & Uncertainty"
        subtitle="Multi-factor decision confidence, prediction entropy and inter-agent conflict"
        right={<LiveBadge />}
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Reveal index={0}>
          <KPICard
            title="Mean Confidence"
            value={<AnimatedNumber value={summary.avgConfidence * 100} decimals={1} suffix="%" />}
            icon={GaugeCircle}
            status={summary.avgConfidence > 0.7 ? "success" : "warning"}
          />
        </Reveal>
        <Reveal index={1}>
          <KPICard title="Distinct Values" value={<AnimatedNumber value={new Set(conf).size} />} icon={Activity} status="info" />
        </Reveal>
        <Reveal index={2}>
          <KPICard
            title="Requires Review"
            value={<AnimatedNumber value={reviewPct} decimals={1} suffix="%" />}
            icon={UserCheck}
            status={reviewPct > 50 ? "danger" : "success"}
          />
        </Reveal>
        <Reveal index={3}>
          <KPICard
            title="Peak Confidence"
            value={<AnimatedNumber value={confMax * 100} decimals={1} suffix="%" />}
            icon={ShieldQuestion}
            status="neutral"
          />
        </Reveal>
      </div>

      {allLow && (
        <Reveal index={4}>
          <GlassCard variant="warning">
            <h3 className="mb-1 font-semibold text-amber-300">Every decision is graded “low” confidence</h3>
            <p className="text-sm text-slate-300">
              Confidence is computed as{" "}
              <code className="rounded bg-slate-800/60 px-1 text-xs">max_prob − uncertainty_penalty − drift − collapse</code>.
              The RUL conformal interval has a fixed width, so its penalty is pinned at 0.30 and confidence cannot
              exceed <span className="font-semibold text-amber-300">{(confMax * 100).toFixed(1)}%</span> — below the
              0.80 “medium” threshold. The grade is therefore unreachable by construction, not a model failure.
            </p>
          </GlassCard>
        </Reveal>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Reveal index={5}>
          <MetricCard title="Confidence Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={hist}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="range" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} animationDuration={900}>
                  {hist.map((d) => (
                    <Cell key={d.range} fill={d.mid > 0.7 ? "#10b981" : d.mid > 0.45 ? "#f59e0b" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </MetricCard>
        </Reveal>

        <Reveal index={6}>
          <MetricCard title="Confidence vs Prediction Entropy">
            {scatter.length ? (
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" dataKey="conf" name="Confidence %" stroke="#64748b" />
                  <YAxis type="number" dataKey="entropy" name="Entropy" stroke="#64748b" />
                  <ZAxis type="number" dataKey="z" range={[40, 80]} />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                  />
                  <Scatter data={scatter} fill="#a855f7" fillOpacity={0.7} animationDuration={900} />
                </ScatterChart>
              </ResponsiveContainer>
            ) : (
              <Empty />
            )}
            <p className="mt-2 text-xs text-slate-500">
              Higher entropy should track lower confidence — an inverse relationship is the healthy pattern.
            </p>
          </MetricCard>
        </Reveal>
      </div>

      <Reveal index={7}>
        <MetricCard title="Lowest Confidence Decisions">
          <DataTable
            columns={[
              { header: "Engine", accessor: "id" },
              { header: "Decision", accessor: "decision" },
              { header: "Confidence", accessor: "conf" },
              { header: "Level", accessor: "level" },
              { header: "Entropy", accessor: "entropy" },
              { header: "Conflict", accessor: "conflict" },
              { header: "Review Reason", accessor: "reason" },
            ]}
            data={[...latest]
              .sort((a, b) => (a.confidence_score ?? 1) - (b.confidence_score ?? 1))
              .slice(0, 15)
              .map((r) => ({
                id: `E${r.engine_id}`,
                decision: String(r.final_decision ?? "—").replace(/_/g, " "),
                conf: r.confidence_score !== null ? `${(r.confidence_score * 100).toFixed(2)}%` : "—",
                level: r.confidence_level ?? "—",
                entropy: r.entropy_score !== null ? r.entropy_score.toFixed(5) : "—",
                conflict: r.conflict_score !== null ? r.conflict_score.toFixed(4) : "—",
                reason: r.review_reason ?? "—",
              }))}
          />
        </MetricCard>
      </Reveal>
    </div>
  );
}
