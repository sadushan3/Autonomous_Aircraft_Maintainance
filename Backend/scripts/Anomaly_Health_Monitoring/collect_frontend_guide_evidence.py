from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/frontend_guide'
BASE = 'http://127.0.0.1:8000/anomaly-health'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'api': {}, 'files': {}}
    routes = ['dashboard', 'evaluation/metrics', 'twin/metrics', 'context-status',
              'detector-config', 'feedback/summary', 'feedback/thresholds', 'reasoning/sensor-graph',
              '/integration/fleet-modes']
    for route in routes:
        try:
            url = 'http://127.0.0.1:8000' + route if route.startswith('/') else f'{BASE}/{route}'
            with urlopen(url, timeout=60) as response:
                evidence['api'][route.lstrip('/')] = json.load(response)
            print(route, 'OK', flush=True)
        except Exception as exc:
            evidence['api'][route] = {'capture_error': str(exc)}
            print(route, str(exc), flush=True)
    dash = evidence['api'].get('dashboard', {})
    engines = dash.get('engines', [])
    if engines:
        focus = min(engines, key=lambda e: e.get('health_index', 100))
        unit = focus['engine_id']
        evidence['focus_engine'] = unit
        for route in [f'engine/{unit}/timeline?max_points=5000', f'reasoning/{unit}']:
            try:
                with urlopen(f'{BASE}/{route}', timeout=60) as response:
                    evidence['api'][route] = json.load(response)
            except Exception as exc:
                evidence['api'][route] = {'capture_error': str(exc)}
    twin = evidence['api'].get('twin/metrics', {})
    rows = twin.get('engines', [])
    p99 = twin.get('validation', {}).get('spread', {}).get('per_sensor_p99_abs_z', {})
    ceiling = twin.get('validation', {}).get('spread', {}).get('ceiling', 3.5)
    good = [s for s in twin.get('available_sensors', []) if p99.get(s, 0) <= ceiling]
    if rows:
        ranked = sorted(rows, key=lambda e: e['max_abs_residual'], reverse=True)
        focus = next((e for e in ranked if e['dominant_sensor_at_worst'] in good), ranked[0])
        sensor = focus['dominant_sensor_at_worst'] if focus['dominant_sensor_at_worst'] in good else (good or ['sensor_11'])[0]
        route = f"twin/trace?engine_id={focus['unit']}&sensor={sensor}&max_points=2000"
        try:
            with urlopen(f'{BASE}/{route}', timeout=60) as response:
                evidence['api']['twin/trace'] = json.load(response)
        except Exception as exc:
            evidence['api']['twin/trace'] = {'capture_error': str(exc)}
    paths = [ROOT.parent / 'Frontend/src/app/pages' / f'{p}.tsx' for p in
             ['HealthMonitoring', 'AnomalyDetection', 'DigitalTwin', 'AIReasoning']]
    paths += [ROOT.parent / 'Frontend/src/app/components' / f'{p}.tsx' for p in
              ['AnomalyValidationPanels', 'FeedbackControl', 'EngineAnatomy', 'SensorGraph', 'Sidebar']]
    paths += [ROOT / p for p in ['app/routers/Anomaly_Health_Monitoring/routes.py',
              'app/services/Anomaly_Health_Monitoring/dashboard/dashboard_service.py',
              'app/services/Anomaly_Health_Monitoring/digital_twin/ensemble_twin.py',
              'app/services/Anomaly_Health_Monitoring/digital_twin/twin_comparator.py',
              'app/services/Anomaly_Health_Monitoring/health_monitoring/health_index_calculator.py',
              'app/services/Anomaly_Health_Monitoring/health_monitoring/health_score_engine.py',
              'app/services/Anomaly_Health_Monitoring/health_monitoring/health_trend_tracker.py',
              'app/services/Anomaly_Health_Monitoring/health_monitoring/health_state_classifier.py',
              'app/services/Anomaly_Health_Monitoring/health_monitoring/health_alert_engine.py',
              'app/services/Anomaly_Health_Monitoring/uncertainty/confidence_estimator.py',
              'app/services/Anomaly_Health_Monitoring/uncertainty/model_agreement.py',
              'app/services/Anomaly_Health_Monitoring/anomaly_detection/early_warning_score.py',
              'app/services/Anomaly_Health_Monitoring/evaluation/evaluate_health.py',
              'app/services/Anomaly_Health_Monitoring/context_modeling/context_drift.py',
              'models/health_state_thresholds.json', 'models/severity_bands.json',
              'models/anomaly_fusion_meta.json', 'models/twin_ensemble_weights.json',
              'models/health_index_calibration.json', 'metrics/forecast_target_080.json']]
    for path in paths:
        if path.is_file():
            item = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if path.suffix == '.json':
                item['data'] = json.loads(path.read_text(encoding='utf-8-sig'))
            evidence['files'][str(path.relative_to(ROOT.parent)).replace('\\', '/')] = item
    (OUT / 'frontend_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
