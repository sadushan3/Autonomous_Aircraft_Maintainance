import { useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  CheckCircle2,
  FileUp,
  Loader2,
  Play,
  RotateCcw,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";
import { useQueryClient } from "@tanstack/react-query";
import { GlassCard } from "../components/GlassCard";
import { MetricCard } from "../components/MetricCard";
import { LiveBadge, PageHeader, Reveal } from "../components/Live";
import { api, ApiError } from "../../lib/api";
import { useFeatureRouting } from "../../lib/hooks";
import type { RunFullResponse } from "../../lib/types";

const STAGES = [
  { key: "anomaly_health", label: "Anomaly & Health", note: "preprocess → context → twin → health → anomaly → reasoning" },
  { key: "predictive_maintenance", label: "RUL & Failure Risk", note: "windowed scoring per engine cycle" },
  { key: "maintenance_supervisor", label: "Maintenance Supervisor", note: "three-way merge → fused decisions" },
];

export function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [stride, setStride] = useState(20);
  const [uncertainty, setUncertainty] = useState(true);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<RunFullResponse | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const qc = useQueryClient();
  const routing = useFeatureRouting();

  const pick = (f: File | null) => {
    setFile(f);
    setResult(null);
    setFailed(null);
  };

  const run = async () => {
    if (!file) return;
    setRunning(true);
    setResult(null);
    setFailed(null);
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    try {
      // Uploads the file directly to /integration/run-full in one call, so no
      // Windows path has to survive being pasted into a query string.
      const res = await api.runFull(file, { stride, include_uncertainty: uncertainty });
      setResult(res);
      toast.success("Pipeline complete", {
        description: `All stages finished for ${file.name}.`,
      });
      // Every downstream panel reads from the exchange layer this just rewrote.
      qc.invalidateQueries();
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
      setFailed(msg);
      toast.error("Pipeline failed", { description: msg });
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setRunning(false);
    }
  };

  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  const eta = stride === 1 ? "~60 min" : stride <= 5 ? "~15 min" : stride <= 20 ? "~4 min" : "~1 min";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Data Upload & Pipeline"
        subtitle="Upload a raw C-MAPSS file and run all three components end to end"
        right={running ? <LiveBadge label="RUNNING" /> : undefined}
      />

      <Reveal index={0}>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f) pick(f);
          }}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer rounded-xl border-2 border-dashed p-12 text-center transition-all ${
            dragging
              ? "scale-[1.01] border-cyan-400 bg-cyan-500/10"
              : file
              ? "border-emerald-500/40 bg-emerald-500/5"
              : "border-slate-700 bg-slate-900/30 hover:border-cyan-500/50 hover:bg-slate-900/50"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".txt,.csv"
            className="hidden"
            onChange={(e) => pick(e.target.files?.[0] ?? null)}
          />
          <motion.div animate={{ y: dragging ? -6 : 0 }} transition={{ type: "spring", stiffness: 300 }}>
            <FileUp className={`mx-auto mb-4 h-12 w-12 ${file ? "text-emerald-400" : "text-cyan-400"}`} />
          </motion.div>
          {file ? (
            <>
              <p className="font-semibold text-emerald-300">{file.name}</p>
              <p className="mt-1 text-sm text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB · ready to run</p>
            </>
          ) : (
            <>
              <p className="font-semibold text-slate-200">Drop a C-MAPSS file here, or click to browse</p>
              <p className="mt-1 text-sm text-slate-500">
                Accepts raw <code className="rounded bg-slate-800/60 px-1">.txt</code> (26 whitespace-delimited
                columns) or a prepared <code className="rounded bg-slate-800/60 px-1">.csv</code>. Name it with its
                subset — e.g. <code className="rounded bg-slate-800/60 px-1">test_FD002.txt</code> — so per-subset
                models are selected correctly.
              </p>
            </>
          )}
        </div>
      </Reveal>

      <Reveal index={1}>
        <MetricCard title="Run Configuration">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <div>
              <label className="mb-2 block text-sm text-slate-400">
                Stride — score every Nth cycle
              </label>
              <input
                type="range"
                min={1}
                max={50}
                value={stride}
                onChange={(e) => setStride(Number(e.target.value))}
                disabled={running}
                className="w-full accent-cyan-400"
              />
              <div className="mt-1 flex justify-between text-xs">
                <span className="font-semibold text-cyan-300">stride = {stride}</span>
                <span className="text-slate-500">est. {eta}</span>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm text-slate-400">Monte-Carlo uncertainty</label>
              <button
                onClick={() => setUncertainty((v) => !v)}
                disabled={running}
                className={`w-full rounded-lg border px-4 py-2 text-sm transition-all ${
                  uncertainty
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                    : "border-slate-700 bg-slate-800/40 text-slate-400"
                }`}
              >
                {uncertainty ? "Enabled — 20 MC passes/point" : "Disabled — faster, no spread"}
              </button>
              <p className="mt-1 text-xs text-slate-500">
                Adds real uncertainty_10/30/50; roughly 35× slower.
              </p>
            </div>

            <div className="flex items-end gap-3">
              <button
                onClick={run}
                disabled={!file || running}
                className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2.5 font-semibold text-white transition-all hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {running ? `Running ${mmss}` : "Run Full Pipeline"}
              </button>
              <button
                onClick={() => pick(null)}
                disabled={running}
                className="rounded-lg border border-slate-700 px-3 py-2.5 text-slate-400 transition-colors hover:text-slate-200 disabled:opacity-40"
                title="Clear"
              >
                <RotateCcw className="h-4 w-4" />
              </button>
            </div>
          </div>
        </MetricCard>
      </Reveal>

      <AnimatePresence>
        {(running || result || failed) && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
          >
            <MetricCard title="Pipeline Stages">
              <div className="space-y-3">
                {STAGES.map((s, i) => {
                  const st = result?.stages?.[s.key]?.status;
                  const done = st === "success";
                  const bad = failed !== null || (st !== undefined && st !== "success");
                  return (
                    <motion.div
                      key={s.key}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="flex items-center gap-4 rounded-lg border border-slate-800/60 bg-slate-900/40 p-4"
                    >
                      {done ? (
                        <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-400" />
                      ) : bad ? (
                        <XCircle className="h-5 w-5 shrink-0 text-red-400" />
                      ) : running ? (
                        <Loader2 className="h-5 w-5 shrink-0 animate-spin text-cyan-400" />
                      ) : (
                        <div className="h-5 w-5 shrink-0 rounded-full border-2 border-slate-700" />
                      )}
                      <div className="flex-1">
                        <div className="font-medium text-slate-200">{s.label}</div>
                        <div className="text-xs text-slate-500">{s.note}</div>
                      </div>
                      <span
                        className={`text-xs font-semibold ${
                          done ? "text-emerald-400" : bad ? "text-red-400" : running ? "text-cyan-400" : "text-slate-600"
                        }`}
                      >
                        {done ? "SUCCESS" : bad ? "FAILED" : running ? "RUNNING" : "PENDING"}
                      </span>
                    </motion.div>
                  );
                })}
              </div>

              {failed && (
                <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
                  {failed}
                </div>
              )}
              {result && (
                <div className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-300">
                  Pipeline status: <span className="font-semibold">{result.status}</span> — every dashboard has been
                  refreshed with the new results.
                </div>
              )}
            </MetricCard>
          </motion.div>
        )}
      </AnimatePresence>

      {routing.data && (
        <Reveal index={2}>
          <GlassCard>
            <h3 className="mb-3 font-semibold text-slate-200">Expected Schema</h3>
            <p className="mb-3 text-sm text-slate-400">
              {routing.data.required_upload_columns.length} required columns; extra columns are ignored, never an
              error. Carried when present: {routing.data.carried_when_present.join(", ")}.
            </p>
            <div className="flex flex-wrap gap-1.5">
              {routing.data.required_upload_columns.map((c, i) => (
                <motion.span
                  key={c}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: Math.min(i * 0.015, 0.5) }}
                  className="rounded border border-slate-700/50 bg-slate-800/50 px-2 py-0.5 font-mono text-xs text-slate-400"
                >
                  {c}
                </motion.span>
              ))}
            </div>
          </GlassCard>
        </Reveal>
      )}
    </div>
  );
}
