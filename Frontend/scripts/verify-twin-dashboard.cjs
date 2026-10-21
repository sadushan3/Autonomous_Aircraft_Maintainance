const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const root = path.resolve(__dirname, '..');
const training = JSON.parse(fs.readFileSync(path.join(root, '../Backend/metrics/digital_twin.json'), 'utf8'));
const recipe = JSON.parse(fs.readFileSync(path.join(root, '../Backend/models/startup_profile_recipe.json'), 'utf8'));
assert.equal(training.startup_profile.recipe, recipe.name);
assert.equal(training.startup_profile.active, true);
const current = training.startup_profile.official_full_trajectory_metrics;
const forecast = { available: true, active: true, ...current };
const state = { data: { available: true, forecast, summary: { subsets: [], unit_id_offset_step: 0 },
  engines: [], available_sensors: [], patterns: {}, models: [{ name: 'old', rmse: 99.12345, weight: 1 }],
  validation: { centring: { mean_absolute_bias: null },
    spread: { exceedance_fraction: null, per_sensor_p99_abs_z: {} },
    growth: { fraction_rising: null, engines_checked: null, engines_with_rising_residual: null } } } };
const hooks = { useTwinMetrics: () => state, useTwinTrace: () => ({ data: {
  engine_id: 1, sensor: 'sensor_11', cycles_total: 1, last_cycle: 1, healthy_until_cycle: 0,
  units: 'normalized sensor units', points: [
  { cycle: 1, expected: 0, actual: 1, forecast: .9, band_low: -.1, band_high: .1, residual: 1, anomaly: 0 }
] } }), useFleetModes: () => ({ data: null }) };
const passthrough = ({ children }) => React.createElement('div', null, children);
const live = { AnimatedNumber: ({ value, decimals = 0 }) => React.createElement('span', null, value.toFixed(decimals)),
  Reveal: passthrough, PageHeader: ({ title, subtitle, right }) => React.createElement('header', null, title, subtitle, right),
  Empty: () => React.createElement('span', null, 'Unavailable'),
  Loading: () => null, ErrorState: () => null };
const chart = new Proxy({}, { get: (_, name) => ({ children, dataKey }) =>
  React.createElement('div', { 'data-chart': name, 'data-key': dataKey }, children) });
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file);
  const source = fs.readFileSync(file, 'utf8');
  const compiled = ts.transpileModule(source, { fileName: file, compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2020,
  } }).outputText;
  const module = { exports: {} };
  cache.set(file, module.exports);
  const localRequire = (name) => {
    if (name.endsWith('/lib/hooks')) return hooks;
    if (name.endsWith('/components/Live')) return live;
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
const { DigitalTwin } = load(path.join(root, 'src/app/pages/DigitalTwin.tsx'));
const render = () => renderToStaticMarkup(React.createElement(DigitalTwin));
const html = render();
for (const [title, value] of [['Forecast R²', current.r2], ['Forecast RMSE', current.rmse], ['Forecast MAE', current.mae]]) {
  assert.ok(html.includes(title), title);
  assert.ok(html.includes(value.toFixed(5)), `${title} must use the active model score`);
}
assert.ok(html.includes('Forecast performance by subset'));
for (const subset of Object.keys(current.per_subset)) assert.ok(html.includes(subset), subset);
assert.ok(html.includes('data-key="forecast"'), 'The chart must expose a distinct forecast series');
assert.ok(!html.includes('99.12345'), 'Old member RMSE must not populate forecast cards');
state.data.forecast = { available: true, active: true, r2: .71234, rmse: 0, mae: 0 };
assert.ok(render().includes('0.71234'), 'New API scores must reach the cards without a hardcoded value');
state.data.forecast = { available: false, active: false, message: 'Active model has no matching evaluation.' };
const missing = render();
assert.ok(missing.includes('Forecast metrics unavailable'));
assert.ok(missing.includes('Active model has no matching evaluation.'));
assert.ok(!missing.includes(current.r2.toFixed(5)), 'Unavailable metrics must not reuse the previous score');
assert.ok(!missing.includes('99.12345'), 'Unavailable metrics must not fall back to old member error');
console.log('Digital twin UI checks passed: active scores, subset table, forecast series, updates and missing metrics.');
