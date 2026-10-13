# Mixed-condition test datasets

Two 30-engine C-MAPSS files built to exercise all four agents in one run, each
with a known label for every engine.

| File | Upload this | Engines | Rows | Scoring points |
|---|---|---|---|---|
| `mixed_conditions_FD001.txt` | yes | 30 | 3,138 | ~1,668 |
| `mixed_conditions_FD002.txt` | yes | 30 | 3,932 | ~1,862 |
| `mixed_conditions_FD001_ground_truth.csv` | answer key | | | |
| `mixed_conditions_FD002_ground_truth.csv` | answer key | | | |
| `../../scripts/make_test_dataset.py` | regenerates either (`--subset`) | | | |

Both finish in minutes rather than the hours a full subset takes.

**Which to use.** FD001 flies a single operating condition — the simplest
end-to-end check. FD002 flies six, so it additionally exercises the
operating-mode clustering, the regime-drift (PSI) check, mode-aware scaling, and
the 70-cycle window, and it uses the FD002 model artefacts your pipeline has
been running. Running both also confirms the subset routing picks the right
models rather than silently defaulting.

## How to run

Upload the `.txt` on the Upload page, or:

```bash
python scripts/run_pipeline.py --input data/synthetic/mixed_conditions_FD002.txt
```

Two notes on the filenames, both deliberate:

- They contain **`FD001`/`FD002`** because the subset is read from the filename,
  and the loader quietly falls back to FD001 models when it cannot find one.
  Renaming to something without a subset scores the data against the wrong
  models — with the wrong window size too.
- They do **not** contain `test`. That word makes the loader join a companion
  `RUL_FD00X.txt`, and the ones in `data/raw` hold values for a different set of
  engines. RUL is instead derived as `max(cycle) - cycle`, which is wrong for
  these truncated trajectories — but that column only feeds
  `health_index_target` and RUL-correlated evaluation metrics, never scoring.
  **Use the key file as truth for these datasets**, not the pipeline's derived RUL.

## What is in them

Every row is a real `train_FD00X` row, so the sensor physics are genuine — the
twin, the health index and the RUL regression all assume the real sensor
manifold, and invented numbers would score as noise. What is synthetic is the
composition: which engine appears, where its trajectory is cut, and the faults
injected into four of them. Because training engines run to failure, cutting a
trajectory of length `L` at cycle `c` leaves an engine whose true remaining life
is exactly `L - c`.

| Units | Category | True RUL at final cycle | Expect |
|---|---|---|---|
| 1–8 | `healthy` | 110–155 | Healthy, no anomaly, `continue_operation` |
| 9–15 | `degrading` | 60–95 | Degrading, `monitor_closely` |
| 16–21 | `warning` | 30–55 | Warning, inspection/maintenance scheduled |
| 22–26 | `critical` | 5–20 | Critical, anomalous, `immediate_maintenance` |
| 27–30 | `anomaly_injected` | 100–125 | **Healthy life, but flagged anomalous** |

Every engine comfortably clears its subset's RUL window (FD001 ≥ 80 cycles
against a 50-cycle window; FD002 ≥ 112 against 70), so none are skipped by the
batch.

## Units 27–30 — the sharpest test in the file

They have 100+ cycles of genuine life left, so an agent that only tracks
remaining life will call them healthy, and it will be right to. Each carries a
deliberate sensor fault over its final stretch, after ~150 clean cycles of
baseline. Only a detector genuinely modelling sensor behaviour flags them, and
the four signatures differ by design so root-cause attribution has something to
separate.

Measured against each engine's own pre-fault baseline, in within-regime sigma:

| Unit | Fault | Sensors | FD001 | FD002 |
|---|---|---|---|---|
| 27 | `hpt_drift` | 3, 4, 11 | +2.1–2.2 sd, spread 2–3× | +2.0–2.4 sd, spread 2.3–3.1× |
| 28 | `sensor_bias` | 2, 17 | +3.5–3.9 sd, spread ~1× | +3.9–4.1 sd, spread ~1× |
| 29 | `transient_spikes` | 9, 14 | mean ~0, spread 9–10× | mean ~0, spread 7–8× |
| 30 | `instability` | 7, 12, 21 | mean ~0, spread 3.5–5× | mean ~0.5, spread 2.4–2.6× |

A healthy engine over the same relative window moves ±0.18 sd at 0.83–1.01×
spread, so the separation is unambiguous.

### Why the scale is per regime, not fleet-wide

FD001 flies one condition, so a sensor's fleet-wide spread *is* its degradation
spread. FD002 flies six, and the regime dominates: measured on `train_FD002`,
fleet-wide standard deviation runs from 5× the within-regime figure (sensor_14)
to **1514×** it (sensor_6). Sizing an injection against the fleet-wide number
would push sensor_6 about 6000 within-regime sigma off its own operating point —
not a fault, a physically impossible reading any detector trips on trivially.
So each row is scaled by the sigma of the sensor inside the regime that row is
flying, which is the scale the detectors actually judge against.

Sensors 1, 5, 10, 16, 18 and 19 are never injected: they are constant or
quantised within a regime, and preprocessing drops them, so a fault there would
be silently discarded.

## What to check after a run

1. **Ordering.** Mean predicted RUL should fall monotonically across
   healthy → degrading → warning → critical. The single best signal that the RUL
   agent and the fusion are behaving.
2. **Anomaly separation.** Units 27–30 should be flagged anomalous while their
   predicted RUL stays high. If they are flagged only because RUL is low, the
   anomaly agent is not contributing independent evidence.
3. **False alarms.** Units 1–8 should stay unflagged. Anything there is a false
   positive on genuinely healthy data.
4. **Root cause.** The four injected engines should not all attribute to the
   same cause — the signatures differ by design.
5. **Risk horizons.** Units 22–26 (true RUL 5–20) should trip the 10- and
   30-cycle thresholds. This is exactly where the stale-threshold bug showed up:
   with the thresholds that were sitting in the exchange layer, none of them fired.
6. **FD002 only — regimes.** The operating-mode panel should report 6 regimes
   and no drift; all six conditions are present in the file.
