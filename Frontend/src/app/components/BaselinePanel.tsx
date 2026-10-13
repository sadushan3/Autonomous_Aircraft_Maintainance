import { motion } from "motion/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MetricCard } from "./MetricCard";
import { GlassCard } from "./GlassCard";
import { DataTable } from "./DataTable";
import { Empty, Loading } from "./Live";
import { useBaselines } from "../../lib/hooks";

const CHAMPION = /champion/i;
const RULE = /rule-based/i;

/**
 * Champion vs baseline benchmark.
 *
 * Renders the honest reading rather than the flattering one: the champion and
 * the rule-based baseline both score 1.0 because the training target *is* a
 * deterministic rule over RUL, risk and anomaly severity, so a model that
 * reproduces the rule perfectly is reproducing a lookup, not demonstrating
 * predictive skill.
 */
export function BaselinePanel() {
  const q = useBaselines();

  if (q.isLoading) return <Loading label="Running baseline benchmark…" />;
  if (q.error || !q.data) return <Empty label="Baseline comparison unavailable." />;

  const rows = [...(q.data.comparison_table ?? [])].sort((a, b) => b.macro_f1 - a.macro_f1);
  const champion = rows.find((r) => CHAMPION.test(r.model));
  const rule = rows.find((r) => RULE.test(r.model));
  const tautological =
    champion && rule && Math.abs(champion.macro_f1 - rule.macro_f1) < 0.005 && rule.macro_f1 > 0.99;

  const f1Chart = rows.map((r) => ({
    name: r.model.replace(" Baseline", "").replace(" Supervisor Agent", ""),
    f1: Number(r.macro_f1.toFixed(4)),
    isChampion: CHAMPION.test(r.model),
  }));

  const costChart = rows.map((r) => ({
    name: r.model.replace(" Baseline", "").replace(" Supervisor Agent", ""),
    cost: r.cost_penalty,
    missed: Number((r.missed_critical_rate * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6">
      {tautological && (
        <GlassCard variant="warning">
          <h3 className="mb-1 font-semibold text-amber-300">
            Read these scores with care
          </h3>
          <p className="text-sm text-slate-300">
            The champion scores <span className="font-mono text-amber-300">{champion!.macro_f1.toFixed(4)}</span> macro-F1 —
            and so does the plain <span className="font-semibold">Rule-Based Baseline</span>. That is not a coincidence:
            the training target is a deterministic rule over RUL, risk and anomaly severity, so a perfect score means
            the model reproduced a rule that already exists in closed form.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            The comparison is still useful — it shows how much each single signal (RUL, risk, anomaly) explains on its
            own — but a 1.0 here is not evidence of maintenance-decision skill.
          </p>
        </GlassCard>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <MetricCard title="Macro-F1 by Model">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={f1Chart} layout="vertical" margin={{ left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" domain={[0, 1]} stroke="#64748b" />
              <YAxis dataKey="name" type="category" stroke="#64748b" width={120} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
              <Bar dataKey="f1" radius={[0, 6, 6, 0]} animationDuration={900}>
                {f1Chart.map((d, i) => (
                  <Cell key={i} fill={d.isChampion ? "#22d3ee" : "#475569"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </MetricCard>

        <MetricCard title="Missed-Critical Rate (%) — lower is safer">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={costChart} layout="vertical" margin={{ left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" stroke="#64748b" />
              <YAxis dataKey="name" type="category" stroke="#64748b" width={120} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
              <Bar dataKey="missed" radius={[0, 6, 6, 0]} animationDuration={900}>
                {costChart.map((d, i) => (
                  <Cell key={i} fill={d.missed > 50 ? "#ef4444" : d.missed > 10 ? "#f59e0b" : "#10b981"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-2 text-xs text-slate-500">
            A model can look accurate while missing every critical case — the Majority-Vote baseline does exactly that.
          </p>
        </MetricCard>
      </div>

      <MetricCard title={`Benchmark Detail — ${(q.data.total_samples ?? 0).toLocaleString()} samples`}>
        <DataTable
          columns={[
            { header: "Model", accessor: "model" },
            { header: "Accuracy", accessor: "acc" },
            { header: "Macro F1", accessor: "f1" },
            { header: "False Critical", accessor: "fc" },
            { header: "Missed Critical", accessor: "mc" },
            { header: "Cost Penalty", accessor: "cost" },
          ]}
          data={rows.map((r) => ({
            model: r.model,
            acc: r.accuracy.toFixed(4),
            f1: r.macro_f1.toFixed(4),
            fc: `${(r.false_critical_rate * 100).toFixed(1)}%`,
            mc: `${(r.missed_critical_rate * 100).toFixed(1)}%`,
            cost: r.cost_penalty.toLocaleString(),
          }))}
        />
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-3 text-xs text-slate-500"
        >
          Benchmarked on the encoded (pre-scaling) dataset, which matches the representation the champion is served at
          inference. This set overlaps the champion's training data, so its scores are in-sample and optimistic; the
          baselines are unaffected by that.
        </motion.p>
      </MetricCard>
    </div>
  );
}
