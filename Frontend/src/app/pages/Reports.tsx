import { useMemo, useState } from "react";
import { Download, FileText, Filter, Table2 } from "lucide-react";
import { toast } from "sonner";
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
import { useDashboard, useFleetSummary } from "../../lib/hooks";
import type { DecisionRow } from "../../lib/types";

const EXPORT_COLUMNS = [
  "engine_id", "cycle", "fd_subset", "health_index", "health_state",
  "predicted_rul", "lower_bound", "upper_bound", "lifecycle_state",
  "risk_10", "risk_30", "risk_50", "risk_state",
  "anomaly_score", "anomaly_severity", "anomaly_type", "root_cause",
  "final_decision", "priority", "maintenance_urgency",
  "confidence_score", "confidence_level", "requires_human_review",
] as const;

function toCsv(rows: DecisionRow[]): string {
  const head = EXPORT_COLUMNS.join(",");
  const body = rows.map((r) =>
    EXPORT_COLUMNS.map((c) => {
      const v = r[c];
      if (v === null || v === undefined) return "";
      const s = String(v);
      // Root-cause strings contain commas; quote anything that could break the row.
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    }).join(","),
  );
  return [head, ...body].join("\n");
}

export function Reports() {
  const { summary, latest, isLoading, error } = useFleetSummary();
  const full = useDashboard(20000);
  const [decision, setDecision] = useState<string>("all");
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    let rows = latest;
    if (decision !== "all") rows = rows.filter((r) => r.final_decision === decision);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter(
        (r) =>
          String(r.engine_id).includes(q) ||
          String(r.root_cause ?? "").toLowerCase().includes(q) ||
          String(r.health_state ?? "").toLowerCase().includes(q),
      );
    }
    return rows;
  }, [latest, decision, search]);

  if (isLoading) return <Loading label="Loading reports…" />;
  if (error) return <ErrorState error={error} />;
  if (!summary) return <Empty />;

  const download = (rows: DecisionRow[], name: string) => {
    const blob = new Blob([toCsv(rows)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Export ready", { description: `${rows.length.toLocaleString()} rows → ${name}` });
  };

  const decisions = ["all", ...Object.keys(summary.decisionCounts)];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Reports & Export"
        subtitle="Query the fused decision set and export it as CSV"
        right={<LiveBadge />}
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Reveal index={0}>
          <KPICard title="Total Scored Rows" value={<AnimatedNumber value={summary.rows} />} icon={Table2} status="info" />
        </Reveal>
        <Reveal index={1}>
          <KPICard title="Engines" value={<AnimatedNumber value={summary.engines} />} icon={FileText} status="neutral" />
        </Reveal>
        <Reveal index={2}>
          <KPICard title="Matching Filter" value={<AnimatedNumber value={filtered.length} />} icon={Filter} status="success" />
        </Reveal>
        <Reveal index={3}>
          <KPICard
            title="Subsets"
            value={<span className="text-xl">{summary.subsets.join(", ") || "—"}</span>}
            icon={Download}
            status="neutral"
          />
        </Reveal>
      </div>

      <Reveal index={4}>
        <GlassCard>
          <div className="flex flex-wrap items-end gap-4">
            <div className="min-w-52 flex-1">
              <label className="mb-1 block text-sm text-slate-400">Search engine, cause or state</label>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="e.g. 227, turbine, Critical"
                className="w-full rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 outline-none transition-colors focus:border-cyan-500/60"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm text-slate-400">Decision</label>
              <select
                value={decision}
                onChange={(e) => setDecision(e.target.value)}
                className="rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 outline-none focus:border-cyan-500/60"
              >
                {decisions.map((d) => (
                  <option key={d} value={d}>
                    {d === "all" ? "All decisions" : d.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={() => download(filtered, "aerotwin_latest_per_engine.csv")}
              className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white transition-transform hover:scale-[1.03]"
            >
              <Download className="h-4 w-4" /> Export filtered ({filtered.length})
            </button>
            <button
              onClick={() => download(full.data?.decisions ?? [], "aerotwin_full_decisions.csv")}
              disabled={!full.data}
              className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-cyan-500/50 disabled:opacity-40"
            >
              <Download className="h-4 w-4" />
              Export all rows ({(full.data?.returned ?? 0).toLocaleString()})
            </button>
          </div>
          {full.data?.truncated && (
            <p className="mt-3 text-xs text-amber-400/80">
              The full export is capped at {full.data.returned.toLocaleString()} of{" "}
              {full.data.total.toLocaleString()} rows — the endpoint is paged to keep responses transferable.
            </p>
          )}
        </GlassCard>
      </Reveal>

      <Reveal index={5}>
        <MetricCard title="Fleet Report — latest cycle per engine">
          {filtered.length ? (
            <DataTable
              columns={[
                { header: "Engine", accessor: "id" },
                { header: "Cycle", accessor: "cycle" },
                { header: "Health", accessor: "health" },
                { header: "State", accessor: "state" },
                { header: "RUL", accessor: "rul" },
                { header: "Risk @30", accessor: "risk" },
                { header: "Severity", accessor: "sev" },
                { header: "Decision", accessor: "decision" },
                { header: "Root Cause", accessor: "cause" },
              ]}
              data={filtered.slice(0, 60).map((r) => ({
                id: `E${r.engine_id}`,
                cycle: r.cycle,
                health: (r.health_index ?? 0).toFixed(1),
                state: r.health_state ?? "—",
                rul: r.predicted_rul !== null ? r.predicted_rul.toFixed(1) : "—",
                risk: `${((r.risk_30 ?? 0) * 100).toFixed(1)}%`,
                sev: r.anomaly_severity ?? "none",
                decision: String(r.final_decision ?? "—").replace(/_/g, " "),
                cause: r.root_cause ?? "—",
              }))}
            />
          ) : (
            <Empty label="No rows match the current filter." />
          )}
        </MetricCard>
      </Reveal>
    </div>
  );
}
