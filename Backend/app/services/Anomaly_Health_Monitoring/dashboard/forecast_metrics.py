import math


def active_forecast_metrics(training, validation, recipe):
    startup = training.get("startup_profile") or validation.get("startup_profile") or {}
    name = recipe.get("name")
    matches = bool(name and startup.get("recipe") == name)
    metrics = startup.get("official_full_trajectory_metrics") or {}
    values = [metrics.get(key) for key in ("r2", "rmse", "mae")]
    available = bool(startup.get("active") and matches and
                     all(isinstance(value, (int, float)) and math.isfinite(value) for value in values))
    if not available:
        return {"available": False, "active": bool(startup.get("active") and matches),
                "recipe": name, "message": "No official forecast metrics are available for the active model."}
    return {"available": True, "active": True, "recipe": name,
            "r2": metrics["r2"], "rmse": metrics["rmse"], "mae": metrics["mae"],
            "rows": metrics.get("rows"), "engines": metrics.get("engines"),
            "informative_targets": metrics.get("n_informative_targets"),
            "target_channels": training.get("n_sensors_modelled"),
            "population": "official_full_trajectory_forecasts",
            "evaluation_status": "retrospective_reused_benchmark",
            "per_subset": {name: {key: score.get(key) for key in ("r2", "rmse", "mae")}
                           for name, score in metrics.get("per_subset", {}).items()}}
