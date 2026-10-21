from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.Anomaly_Health_Monitoring.search_profile_target import ROOT, RUN, SOURCE
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import write
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope


def main():
    config = Config()
    config.mlops.log_to_file = config.mlops.enable_mlflow = False
    bundle = joblib.load(config.paths.models / "startup_profile_twin.pkl")
    source = joblib.load(SOURCE / "prepared.joblib")["scaled"]
    frame = training_scope(source, config, ("selection",), required=True)
    population = bundle["population"].predict(EnsembleTwin._profile_context(frame, bundle))
    x, ready = EnsembleTwin._profile_features(frame, population, bundle)
    keep = ready & frame.RUL.ge(125).to_numpy()
    healthy = frame.loc[keep].copy()
    targets = bundle["target_names"]
    actual = healthy[targets].to_numpy(float)
    prediction = population[keep] + EnsembleTwin._predict_profile_correction(bundle["correction"], x[keep])
    residual = pd.DataFrame(actual - prediction, columns=targets, index=healthy.index)
    residual["unit"], residual["cycle"], residual["mode"] = healthy.unit, healthy.cycle, healthy.operating_mode
    residual = residual.sort_values(["unit", "mode", "cycle"])
    groups = residual.groupby(["unit", "mode"])
    differences = groups[targets].diff()
    consecutive = groups.cycle.diff().eq(1)
    differences = differences.loc[consecutive]
    difference_units = residual.loc[consecutive, "unit"]
    units = sorted(healthy.unit.unique())
    sums = healthy.groupby("unit")[targets].sum().reindex(units).to_numpy()
    squares = healthy[targets].pow(2).groupby(healthy.unit).sum().reindex(units).to_numpy()
    counts = healthy.groupby("unit").size().reindex(units).to_numpy()
    diff_squares = differences.pow(2).groupby(difference_units).sum().reindex(units, fill_value=0).to_numpy()
    diff_counts = differences.groupby(difference_units).size().reindex(units, fill_value=0).to_numpy()
    informative = actual.var(axis=0) > 1e-8

    def estimate(weights):
        n = weights @ counts
        variance = weights @ squares / n[:,None] - (weights @ sums / n[:,None])**2
        noise = weights @ diff_squares / (2 * (weights @ diff_counts)[:,None])
        proxy = 1. - noise / np.maximum(variance, 1e-30)
        return proxy, proxy[:,informative].mean(axis=1)

    proxy, point = estimate(np.ones((1, len(units))))
    rng = np.random.default_rng(20261016)
    weights = np.array([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(1000)])
    _, replicates = estimate(weights)
    result = {"population": "shared development selection engines only", "healthy_rows":len(healthy),
        "healthy_engines":len(units), "same_mode_adjacent_pairs":len(differences),
        "current_metric": "uniform-average R2 of the 17 informative unsmoothed sensor targets",
        "estimated_noise_limited_r2": float(point[0]),
        "engine_bootstrap_95_percent_interval": np.quantile(replicates,[.025,.975]).tolist(),
        "bootstrap_replicates":1000,
        "assumptions": ["The latent healthy signal changes little between adjacent observations in the same mode.",
            "Measurement errors are independent across these adjacent cycles.",
            "The frozen baseline accounts for predictable context changes before differencing."],
        "limitations": "This is an approximate noise diagnostic, not a strict theoretical upper bound. Temporal correlation or rapid latent dynamics can bias it. Bootstrap intervals do not include assumption error or historical benchmark reuse.",
        "target_r2":.8, "targets_smoothed_or_removed":False,
        "per_sensor":{name:{"informative":bool(informative[i]), "estimated_noise_limited_r2":float(proxy[0,i])}
                      for i,name in enumerate(targets)}}
    write(RUN / "noise_bootstrap.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
