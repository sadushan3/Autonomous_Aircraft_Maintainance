from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import OperatingModeDetector
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.trend_forecast import engine_batches
from app.utils.Anomaly_Health_Monitoring.training_split import training_scope
from scripts.Anomaly_Health_Monitoring.evaluate_linear_baseline import prepare
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT, SOURCE, digest, score, write


def main(run):
    config = Config()
    config.mlops.enable_mlflow = config.mlops.log_to_file = False
    lock = json.loads((run / 'model_lock.json').read_text())
    assert digest(run / 'candidate_profile.pkl') == lock['candidate_sha256']
    bundle = joblib.load(run / 'candidate_profile.pkl')
    assert bundle['feature_schema'] == 'causal_trend_history_v1'
    predictor = object.__new__(EnsembleTwin)
    predictor.config, predictor.startup_profile = config, bundle
    predictor.target_names = bundle['target_names']
    prepared = joblib.load(SOURCE / 'prepared.joblib')['scaled']
    selection = training_scope(prepared, config, ('selection',), required=True)
    del prepared
    gc.collect()
    population = bundle['population'].predict(EnsembleTwin._profile_context(selection, bundle))
    started = time.perf_counter()
    features, ready = EnsembleTwin._profile_features(selection, population, bundle)
    expected = population.copy()
    expected[ready] += EnsembleTwin._predict_profile_correction(bundle['correction'], features, row_mask=ready)
    direct_seconds = time.perf_counter() - started
    global_feature_bytes = features.nbytes
    del features
    gc.collect()
    started = time.perf_counter()
    actual, streamed_ready = predictor._profile_prediction(selection, population)
    streamed_seconds = time.perf_counter() - started
    np.testing.assert_array_equal(ready, streamed_ready)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    maximum_difference = float(np.max(np.abs(actual - expected)))
    largest_batch = max(len(rows) for rows in engine_batches(selection))
    print('SELECTION STREAM MATCH', int(ready.sum()), maximum_difference, flush=True)
    del actual, expected, selection, population
    gc.collect()

    context = OperatingModeDetector(config).load()
    scaler = PerModeScaler(config).load_scalers()
    raw = CMAPSSReader(config).read_subsets(split='test')
    official = prepare(raw, config, context, scaler)
    del raw
    gc.collect()
    started = time.perf_counter()
    prediction, ready = predictor._profile_prediction(official, np.zeros((len(official), 21)))
    official_seconds = time.perf_counter() - started
    actual = official[bundle['target_names']].to_numpy(float)
    measured = score(official, actual, prediction, ready)
    recorded = json.loads((run / 'official_full_comparison.json').read_text())['candidate']
    for key in ('r2', 'rmse', 'mae'):
        assert abs(measured[key] - recorded[key]) < 1e-10, (key, measured[key], recorded[key])
    assert (measured['rows'], measured['engines']) == (recorded['rows'], recorded['engines'])
    for subset, subset_score in recorded['per_subset'].items():
        for key in ('r2', 'rmse', 'mae'):
            assert abs(measured['per_subset'][subset][key] - subset_score[key]) < 1e-10
    result = {'candidate_sha256': lock['candidate_sha256'], 'parameters_changed': False,
        'selection_ready_rows': int(streamed_ready.sum()),
        'maximum_absolute_prediction_difference': maximum_difference,
        'selection_direct_seconds': direct_seconds, 'selection_streamed_seconds': streamed_seconds,
        'global_selection_feature_bytes': global_feature_bytes,
        'largest_selection_batch_feature_bytes': largest_batch * 1141 * 8,
        'official_input_rows': len(official), 'official_seconds_without_context_twins': official_seconds,
        'official_metrics': measured, 'official_metrics_match_locked_audit': True,
        'timing_note': 'Single runs; selection timing compares the feature/forecast paths after loading the same model. Official timing excludes the unchanged four context twins. Hardware load and cache warmup affect timing.'}
    write(run / 'history_streaming_verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'official_metrics'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    with threadpool_limits(limits=4):
        main(parser.parse_args().run_dir.resolve())
