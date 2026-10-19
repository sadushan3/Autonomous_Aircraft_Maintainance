const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const root = path.resolve(__dirname, '..');
const keys = ['isolation_forest', 'mahalanobis', 'lstm_autoencoder', 'residual_rule'];
const scoreKeys = ['isolation_forest_score', 'mahalanobis_score', 'lstm_ae_score', 'residual_rule_score'];
const alert = { engine_id: 7, cycle: 110, severity: 'critical', anomaly_score: .9456, threshold_used: .9085,
  isolation_forest_score: .9876, mahalanobis_score: .8765, lstm_ae_score: .9654, residual_rule_score: .9543,
  model_agreement: .8123, detector_agreement: .9234, uncertainty_score: .1877, reliability_score: .9234,
  anomaly_persistence_cycles: 5, early_warning_score: .7345 };
const state = { data: { fleet: { total_anomalies: 50, fault_anomalies: 45, operational_anomalies: 5 },
  engines: [alert], alerts: [alert] } };
const evaluation = { test: { severity_mean_rul: { none: 170, low: 80, critical: 12 },
  severity_row_counts: { none: 40, low: 8, critical: 2 } },
  acceptance: { gate_behaviour: { total_flagged: 50, fault_anomalies: 45, operational_anomalies: 5,
    operational_suppression_rate: .1 }, rul_alignment: { fleet_mean_rul: 120, fault_mean_rul: 20 } } };
const detectorConfig = { data: { detectors: keys.map((key) => ({ key, weight: .25, enabled: true, trained: true,
  principle: key })), fusion: { active_detectors: 4, calibrated_threshold: .9085,
    normalisation: 'Persisted healthy-reference calibration and equal weights.' } } };
const hooks = { useAnomalyDashboard: () => state, useAnomalyEvaluation: () => evaluation,
  useDetectorConfig: () => detectorConfig };
const passthrough = ({ children }) => React.createElement('div', null, children);
const live = { AnimatedNumber: ({ value, decimals = 0 }) => React.createElement('span', null, value.toFixed(decimals)),
  Reveal: passthrough, LiveBadge: () => null,
  PageHeader: ({ title, subtitle }) => React.createElement('header', null, title, subtitle),
  Empty: ({ label }) => React.createElement('span', null, label), Loading: () => null, ErrorState: () => null };
const chart = new Proxy({}, { get: (_, name) => ({ children, dataKey, data }) =>
  React.createElement('div', { 'data-chart': name, 'data-key': dataKey, 'data-values': JSON.stringify(data) }, children) });
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), { fileName: file, compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2020,
  } }).outputText;
  const module = { exports: {} };
  cache.set(file, module.exports);
  const localRequire = (name) => {
    if (name.endsWith('/lib/hooks')) return hooks;
    if (name.endsWith('/lib/engine-param')) return { useEngineParam: () => ['999', () => {}] };
    if (name.endsWith('/components/Live')) return live;
    if (name.endsWith('/components/FeedbackControl')) return { FeedbackControl: () => null };
    if (name.endsWith('/components/AnomalyValidationPanels')) return { DetectionPerformancePanel: () => null };
    if (name.endsWith('/components/DataTable')) return { DataTable: ({ data }) => React.createElement('span', null, JSON.stringify(data)) };
    if (name === 'recharts') return chart;
    if (name === 'motion/react') return { motion: new Proxy({}, { get: () => passthrough }) };
    if (name.startsWith('.')) {
      const resolved = path.resolve(path.dirname(file), name);
      const target = ['.tsx', '.ts', '.js'].map((ext) => resolved + ext).find(fs.existsSync);
      if (target) return load(target);
    }
    return require(name);
  };
  vm.runInNewContext(compiled, { module, exports: module.exports, require: localRequire }, { filename: file });
  return module.exports;
}
const { AnomalyDetection } = load(path.join(root, 'src/app/pages/AnomalyDetection.tsx'));
const render = () => renderToStaticMarkup(React.createElement(AnomalyDetection));
let html = render();
assert.ok(html.includes('4 active detector models'));
assert.equal(html.split('Active · trained model').length - 1, 4);
for (const key of scoreKeys) assert.ok(html.includes(alert[key].toFixed(4)), key);
for (const value of [.8123, .9234, .1877, .7345]) assert.ok(html.includes(value.toFixed(3)) || html.includes(value.toFixed(2)));
assert.ok(html.includes('10.0% of raw flags'));
assert.ok(!html.includes('One flag in five'));
assert.ok(html.includes('Engine E7'));
assert.ok(!html.includes('Engine E999'));
assert.ok(!html.includes('saved evaluation report does not include mean RUL'));
for (const key of scoreKeys) alert[key] = null;
html = render();
assert.ok(html.includes('No detector scores are available in the current alerts.'));
assert.ok(!html.includes('0.0000'), 'Missing detector outputs must not be displayed as zero scores');
alert.isolation_forest_score = 0;
html = render();
assert.ok(html.includes('0.0000'), 'A real zero score must be preserved');
state.data.alerts = [];
html = render();
assert.ok(html.includes('0.9085'), 'The saved cutoff must remain visible without alerts');
console.log('Anomaly UI checks passed: four active models, score cards, severity chart, measured gating, selection and missing/zero handling.');
