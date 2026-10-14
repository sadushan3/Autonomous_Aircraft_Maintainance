# AeroTwin OS — Frontend

React + Vite + TypeScript dashboard for the Explainable Health Monitoring &
Anomaly Reasoning platform. Every page reads **live data from the FastAPI
backend** — there is no mock data left in the app.

## Running

Start the backend first (from `../Backend`):

```bash
python -m uvicorn main:app --reload --port 8000
```

Then:

```bash
npm install
npm run dev          # http://localhost:5173
```

Other scripts: `npm run build`, `npm run preview`, `npm run typecheck`.

## How it connects

The client calls **relative `/api/*` paths only**. Vite proxies `/api` to
`http://127.0.0.1:8000` (see `vite.config.ts`), which means:

- no CORS configuration is needed in dev, and
- the same production bundle works behind any reverse proxy that maps `/api`
  to the backend — no rebuild, no baked-in host.

The proxy has its timeouts disabled because a full pipeline run at `stride=1`
can take around an hour.

| Layer | File |
|---|---|
| Typed endpoint client | `src/lib/api.ts` |
| Response types (from real payloads) | `src/lib/types.ts` |
| React Query hooks + fleet rollup | `src/lib/hooks.ts` |
| Loading / error / empty / animation primitives | `src/app/components/Live.tsx` |

Live panels poll every 15s (`LIVE_MS`). `refetchOnWindowFocus` is off — the
dashboard endpoint reads a 34k-row CSV per call, so focus-refetching on top of
polling would double backend load for no benefit.

## Pages → backend

| Page | Endpoints |
|---|---|
| Overview | `/integration/dashboard`, `/integration/fleet-modes`, `/integration/maintenance-queue` |
| Health Monitoring | `/integration/dashboard` |
| RUL Prediction | `/integration/dashboard`, `/integration/leaderboard` |
| Failure Risk | `/integration/dashboard` |
| Digital Twin | `/integration/fleet-modes`, `/anomaly-health/context-status`, `/integration/feature-routing` |
| AI Reasoning | `/anomaly-health/reasoning/sensor-graph`, `/anomaly-health/reasoning/{engine_id}` |
| Confidence | `/integration/dashboard` |
| Maintenance Supervisor | `/integration/dashboard`, `/integration/maintenance-queue` |
| Upload | `/integration/run-full` (direct multipart upload), `/integration/feature-routing` |
| Reports | `/integration/dashboard` (+ client-side CSV export) |
| Settings | `/health`, `/anomaly-health/feedback/*`, `/integration/feature-routing`, `/predictive-maintenance/artifacts/status` |

## Notes on the data

- **First run shows an empty state, not an error.** Until
  `/integration/run-full` has been executed once there are no decisions to
  read; the UI says so explicitly and points at the Upload page.
- **Fleet figures use the latest cycle per engine.** Averaging every cycle of
  every engine would weight long trajectories more heavily and understate
  current fleet health.
- **The dashboard endpoint is paged.** `total` is always shown alongside what
  was fetched, so a figure never silently describes a subset.
- Panels that depend on a run-time option (Monte-Carlo uncertainty) detect when
  it was disabled and say so rather than rendering a flat zero chart.

## Setup changes made to the Figma export

- `react` / `react-dom` were only optional `peerDependencies`, so `npm install`
  never installed them and the app could not build — moved to `dependencies`.
- Added `typescript`, `@types/react`, `@types/react-dom`, `@types/node` and a
  `tsconfig.json` (the export had none).
- Added `@tanstack/react-query` for fetching, caching and polling.
- Added the `/api` dev proxy.



python -m uvicorn main:app --reload --port 8000

npm install
npm run dev
