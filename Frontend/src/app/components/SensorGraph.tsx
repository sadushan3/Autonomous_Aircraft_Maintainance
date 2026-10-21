import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";

interface Edge {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

const SUBSYSTEM_LABEL: Record<string, string> = {
  fan: "Fan",
  lpc: "LP Compressor",
  hpc: "HP Compressor",
  burner: "Combustor",
  hpt: "HP Turbine",
  lpt: "LP Turbine",
  bypass: "Bypass Duct",
  bleed: "Bleed Air",
  general: "General",
};


export function SensorGraph({
  nodes,
  edges,
  faultySubsystem,
  height = 420,
}: {
  nodes: string[];
  edges: Edge[];
  faultySubsystem?: string | null;
  height?: number;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);


  const positions = useMemo(() => {
    const n = nodes.length;
    if (!n) return {} as Record<string, { x: number; y: number }>;
    const cx = width / 2;
    const cy = height / 2;
    const r = Math.min(width, height) * 0.36;

    const pos: Record<string, { x: number; y: number }> = {};
    nodes.forEach((id, i) => {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      pos[id] = { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
    });

    const linked = edges.filter((e) => pos[e.source] && pos[e.target]);
    for (let iter = 0; iter < 260; iter++) {
      const force: Record<string, { x: number; y: number }> = {};
      for (const id of nodes) force[id] = { x: 0, y: 0 };


      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = pos[nodes[i]];
          const b = pos[nodes[j]];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) d2 = 1;
          const f = 9000 / d2;
          const d = Math.sqrt(d2);
          force[nodes[i]].x += (dx / d) * f;
          force[nodes[i]].y += (dy / d) * f;
          force[nodes[j]].x -= (dx / d) * f;
          force[nodes[j]].y -= (dy / d) * f;
        }
      }


      for (const e of linked) {
        const a = pos[e.source];
        const b = pos[e.target];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const f = (d - 130) * 0.02 * (e.weight || 0.5);
        force[e.source].x += (dx / d) * f;
        force[e.source].y += (dy / d) * f;
        force[e.target].x -= (dx / d) * f;
        force[e.target].y -= (dy / d) * f;
      }

      const damp = 0.12;
      for (const id of nodes) {
        pos[id].x += Math.max(-12, Math.min(12, force[id].x)) * damp;
        pos[id].y += Math.max(-12, Math.min(12, force[id].y)) * damp;

        pos[id].x = Math.max(60, Math.min(width - 60, pos[id].x));
        pos[id].y = Math.max(40, Math.min(height - 40, pos[id].y));
      }
    }
    return pos;
  }, [nodes, edges, width, height]);


  const downstream = useMemo(() => {
    if (!faultySubsystem) return new Set<string>();
    const seen = new Set<string>([faultySubsystem]);
    const queue = [faultySubsystem];
    while (queue.length) {
      const cur = queue.shift()!;
      for (const e of edges) {
        if (e.source === cur && !seen.has(e.target)) {
          seen.add(e.target);
          queue.push(e.target);
        }
      }
    }
    return seen;
  }, [faultySubsystem, edges]);

  const active = hovered ?? faultySubsystem ?? null;
  const isLit = (id: string) => (active ? downstream.has(id) || id === active : false);
  const edgeLit = (e: Edge) => (active ? downstream.has(e.source) && downstream.has(e.target) : false);

  if (!nodes.length) {
    return <div className="py-14 text-center text-sm text-slate-500">No dependency graph available.</div>;
  }

  return (
    <div ref={wrapRef} className="w-full">
      <svg width={width} height={height} className="overflow-visible">
        <defs>
          <marker id="arw" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" fill="#334155" />
          </marker>
          <marker id="arw-lit" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" fill="#ef4444" />
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {edges.map((e, i) => {
          const a = positions[e.source];
          const b = positions[e.target];
          if (!a || !b) return null;
          const lit = edgeLit(e);
          return (
            <motion.line
              key={`${e.source}-${e.target}-${i}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={lit ? "#ef4444" : "#334155"}
              strokeWidth={lit ? 2.2 : 1 + (e.weight || 0.5)}
              strokeOpacity={active && !lit ? 0.18 : 0.75}
              markerEnd={lit ? "url(#arw-lit)" : "url(#arw)"}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 0.7, delay: Math.min(i * 0.02, 0.5) }}
            />
          );
        })}

        {nodes.map((id, i) => {
          const p = positions[id];
          if (!p) return null;
          const lit = isLit(id);
          const isSource = id === (hovered ?? faultySubsystem);
          return (
            <motion.g
              key={id}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: Math.min(i * 0.03, 0.6) }}
              onMouseEnter={() => setHovered(id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: "pointer" }}
            >
              {isSource && (
                <motion.circle
                  cx={p.x}
                  cy={p.y}
                  r={26}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth={1.5}
                  animate={{ r: [26, 38, 26], opacity: [0.7, 0, 0.7] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}
              <circle
                cx={p.x}
                cy={p.y}
                r={isSource ? 20 : 15}
                fill={isSource ? "#ef4444" : lit ? "#f97316" : "#0f172a"}
                stroke={isSource ? "#fca5a5" : lit ? "#fdba74" : "#22d3ee"}
                strokeWidth={2}
                opacity={active && !lit ? 0.3 : 1}
                filter={isSource ? "url(#glow)" : undefined}
              />
              <text
                x={p.x}
                y={p.y + (isSource ? 36 : 30)}
                textAnchor="middle"
                fontSize={11}
                fill={lit ? "#fed7aa" : "#94a3b8"}
                opacity={active && !lit ? 0.35 : 1}
                style={{ pointerEvents: "none" }}
              >
                {SUBSYSTEM_LABEL[id] ?? id}
              </text>
            </motion.g>
          );
        })}
      </svg>

      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" /> fault origin
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-orange-500" /> downstream (can be affected)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full border border-cyan-400 bg-slate-900" /> unaffected
        </span>
        <span className="ml-auto">Hover any node to trace its propagation path.</span>
      </div>
    </div>
  );
}
