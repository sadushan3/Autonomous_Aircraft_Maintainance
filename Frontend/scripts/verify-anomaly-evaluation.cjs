// Read-only regression checks for the evaluation report adapter.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'src/lib/anomalyEvaluation.ts'), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
}).outputText;
const context = { exports: {} };
vm.runInNewContext(compiled, context);
const { readAnomalyHoldout, criticalEngineDetection } = context.exports;
const wrap = (report) => ({ available: true, metrics: { holdout_anomaly_evaluation: report } });

const saved = JSON.parse(fs.readFileSync(
  path.join(root, '../Backend/metrics/holdout_anomaly_evaluation.json'), 'utf8',
));
const actual = readAnomalyHoldout(wrap(saved));
assert.ok(actual.test, 'Saved report must populate the detection panel');
assert.equal(actual.test.auc_healthy_vs_critical, saved.metrics.test.auc_healthy_vs_critical);
assert.equal(actual.perSubset, saved.metrics.test_per_subset);
assert.equal(criticalEngineDetection(actual.test),
  saved.metrics.test.condition_proxy_metrics.engine_events.detection_rate_of_reaching);

const legacy = { test: { auc_healthy_vs_critical: 0.9, detection_rate_of_reaching: 0 },
  train: { rows: 10 }, test_per_subset: { FD001: { auc: 0.9, false_alarm: 0, detection: 0.5 } } };
assert.equal(readAnomalyHoldout(wrap(legacy)).test, legacy.test);
assert.equal(readAnomalyHoldout(wrap(legacy)).train, legacy.train);
assert.equal(criticalEngineDetection(legacy.test), 0, 'Zero must remain a valid result');
assert.equal(criticalEngineDetection({ detection_rate: 0.22 }), undefined,
  'Do not substitute an all-engine rate for a critical-engine rate');
assert.equal(readAnomalyHoldout().test, null);
assert.equal(readAnomalyHoldout(wrap({ error: 'invalid JSON' })).test, null);
assert.equal(actual.test.severity_mean_rul, saved.metrics.test.severity_mean_rul,
  'Saved severity evidence must reach the chart unchanged');
for (const band of ['none', 'low', 'medium', 'high', 'critical']) {
  assert.ok(Number.isFinite(actual.test.severity_mean_rul[band]), `Missing ${band} mean RUL`);
}
assert.equal(Object.values(actual.test.severity_row_counts).reduce((a, b) => a + b, 0),
  actual.test.rows, 'Band counts must cover the evaluated rows');
assert.equal(readAnomalyHoldout(wrap(legacy)).test.severity_mean_rul, undefined,
  'Missing severity evidence must remain unavailable');
console.log('Anomaly evaluation adapter checks passed (saved v2, legacy, zero, missing/error, denominators).');
