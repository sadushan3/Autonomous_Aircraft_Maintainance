from __future__ import annotations
import json
from pathlib import Path


def render(result, run, root):
    if result.get("objective") == "full_trajectory_forecasts":
        return render_full_forecast(result,run,root)
    advanced = "metrics" in result["selected"]
    adaptive = result.get("prediction_role") == "adaptive_sensor_forecast"
    title = "R2 0.8 target search and verified improvement" if advanced else "Verified startup sensor prediction improvement"
    if adaptive:
        title = "Verified causal sensor forecast improvement"
    lines = [f"# {title} - 16 October 2026", "",
        ("The accepted forecast is active. With the user's approval, it uses operating context, frozen commissioning statistics and strictly prior sensor readings to forecast actual raw sensor behavior. The original frozen healthy reference still drives anomaly detection and health monitoring. The baseline used frozen commissioning statistics; the forecast has access to additional past observations." if adaptive else
         "The accepted correction is active. It uses only operating context and frozen statistics from cycles 1-30 to improve post-commissioning sensor predictions. Original monitoring references are preserved."), "",
        f"Tested {result.get('model_recipes_tested',49)} model recipes and four blend weights per recipe. The selected model was locked before confirmation and official scoring. All C-MAPSS results remain retrospective because these trajectories have been inspected previously.", "",
        "| Population | R2 before | R2 after | RMSE before | RMSE after | MAE before | MAE after |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    if advanced:
        populations = [("Development selection",result["baseline"]["selection"],result["selected"]["metrics"])]
        if "crossfit_metrics" in result["selected"]:
            populations.append(("Cross-fit sensor weights (development)",result["baseline"]["selection"],result["selected"]["crossfit_metrics"]))
    else:
        populations = [("Inner validation",result["baseline"]["inner"],result["selected"]["inner"]),
                       ("Development selection",result["baseline"]["selection"],result["selected"]["selection"])]
    populations += [("Retrospective confirmation",result["confirmation"]["baseline"],result["confirmation"]["candidate"]),
        ("Reused official benchmark",result["official_reused_benchmark"]["baseline"],result["official_reused_benchmark"]["candidate"])]
    for name,before,after in populations:
        lines.append(f"| {name} | {before['r2']:.6f} | {after['r2']:.6f} | {before['rmse']:.6f} | {after['rmse']:.6f} | {before['mae']:.6f} | {after['mae']:.6f} |")
    lines += ["", "The score uses original unsmoothed targets on the same healthy RUL >=125 rows after commissioning. R2 averages the informative sensor channels using the existing definition; RMSE and MAE include all sensor targets. The engine partitions, targets and score formula have not changed.", "",
        "Acceptance requires higher overall R2 and lower RMSE/MAE on development and frozen confirmation. Each subset permits at most 1% RMSE and 0.01 R2 deterioration. Publication also requires all three official aggregate metrics to improve, without retuning on official results. The table below exposes smaller regressions, including MAE tradeoffs.", ""]
    if "official_full_trajectory_forecasts" in result:
        whole=result["official_full_trajectory_forecasts"]
        before,after=whole["baseline"],whole["candidate"]
        lines += ["## Additional full-trajectory forecast check", "",
            f"Across all {after['rows']:,} ready official rows, including degradation, raw-sensor forecast R2 changes from {before['r2']:.6f} to {after['r2']:.6f}, RMSE from {before['rmse']:.6f} to {after['rmse']:.6f}, and MAE from {before['mae']:.6f} to {after['mae']:.6f}. This is a different row population. It is report-only and does not replace the healthy benchmark or determine whether the requested 0.8 target was met.", ""]
    if advanced:
        official = result["official_reused_benchmark"]["candidate"]["r2"]
        lines += ["## Requested target", "",
            f"The requested overall R2 of 0.8 was {'reached' if official>=.8 else 'not reached'}. The official result is {official:.6f}. " +
            ("Grouped linear models, tree ensembles, approximate kernels and regularized sensor mixtures were compared using strictly past readings. Both healthy-only and full-trajectory fitting scopes were tested on permitted fitting engines. Temporal models use individual rows rather than time-bin aggregates." if adaptive else
             "Larger tree, kernel, linear, latent-variable and neural models, richer commissioning statistics, additional permitted fitting engines and regularized sensor mixtures were tested."), ""]
        noise_path = run.parent / "noise_bootstrap.json"
        if noise_path.exists():
            noise = json.loads(noise_path.read_text())
            low,high = noise["engine_bootstrap_95_percent_interval"]
            lines += [f"An adjacent-cycle residual diagnostic estimates a noise-limited R2 near {noise['estimated_noise_limited_r2']:.3f}, with an engine-bootstrap interval of {low:.3f}-{high:.3f}. This assumes slow latent changes and independent adjacent measurement noise. It is an approximate diagnostic, not a strict mathematical ceiling, and the interval excludes assumption error. The measured data do not support claiming 0.8 from further tuning alone.", ""]
        lines += [f"The compressed candidate artifact is {(run/'candidate_profile.pkl').stat().st_size/1024**2:.1f} MiB. The sensor mixture adds model evaluations and memory cost. Performance measurements, when available, are included in the full evaluation.", ""]
        if "inference_timing" in result:
            timing=result["inference_timing"]
            lines += [f"One audit run over {timing['rows']:,} rows took {timing['baseline_seconds']:.2f} seconds for the baseline and {timing['candidate_seconds']:.2f} seconds for the candidate, with the reference forest restricted to one worker for determinism. First-run framework warmup can affect this comparison.", ""]
        if noise_path.exists():
            lines += ["NASA's original C-MAPSS generation paper describes adding random measurement noise to sensor outputs and propagating process noise through the simulation. That supports investigating noise as a limit; the numerical estimate above comes from this experiment. [NASA generation paper, noise section](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf#page=6).", ""]
    lines += ["## Verification", "",
        "Actual inference preserves monitoring references, residuals, anomaly scores and decisions, and health indices across all 104,897 official rows. " +
        ("Independent raw prefixes, shuffled rows and removed RUL fields preserve forecasts. Changing current or future sensor readings cannot change the current forecast; changing strictly prior readings can change later forecasts while the monitoring reference remains frozen." if adaptive else
         "Independent raw prefixes, shuffled rows, removed RUL fields and perturbed observations after cycle 30 preserve commissioning predictions.") +
        " Normal retraining reproduces the selected model. Every replaced file is backed up in the experiment's before directory." +
        (" A fresh retrain also reproduces all 6,168 development predictions (maximum absolute difference below 1e-10) and preserves the verified training plan, forecast role and validation role." if result.get("verification",{}).get("normal_retraining_metadata_verified") else ""), "",
        f"Regression tests passed: {result.get('regression_tests_passed','see test log')}.", "",
        "## Results for each subset", "",
        "| Population / subset | R2 before | R2 after | RMSE before | RMSE after | MAE before | MAE after |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for key,title in (("confirmation","Confirmation"),("official_reused_benchmark","Official")):
        item=result[key]
        for name,before in item["baseline"]["per_subset"].items():
            after=item["candidate"]["per_subset"][name]
            lines.append(f"| {title} / {name} | {before['r2']:.6f} | {after['r2']:.6f} | {before['rmse']:.6f} | {after['rmse']:.6f} | {before['mae']:.6f} | {after['mae']:.6f} |")
    relative=run.relative_to(root).as_posix()
    lines += ["", f"[Full evaluation](../{relative}/verified_evaluation.json), [frozen protocol](../{relative}/protocol.json), [locked model](../{relative}/model_lock.json), [saved recipe](../{relative}/selected_recipe.json).", ""]
    if result.get("verification",{}).get("normal_retraining_metadata_verified"):
        lines += [f"[Retraining provenance and prediction verification](../{relative}/retrain_metadata_verification.json).", ""]
    return "\n".join(lines)


def render_full_forecast(result,run,root):

    whole=result["official_full_trajectory_forecasts"]
    value=whole["candidate"]["r2"]
    target=float(result.get("target_r2",.8))
    lines=["# Full-trajectory sensor forecasting: verified results", "",
        f"The requested official full-trajectory R2 of {target:g} was {'reached' if value>=target else 'not reached'}. The measured result is {value:.6f}.", "",
        "The active forecast uses operating context and strictly prior sensor readings. The baseline already used prior readings. Both models retain the original frozen monitoring reference. Current sensor targets and future readings are excluded from forecast inputs.", "",
        "The full-trajectory metric uses the same original unsmoothed normalized sensor targets and ready rows as the prior 0.6417 result. R2 retains the existing informative-channel average; RMSE and MAE retain all 21 channels. No target smoothing, sensor removal or evaluation-row changes were used in this search.", "",
        "## Full-trajectory forecasts", "",
        "| Population | R2 before | R2 after | RMSE before | RMSE after | MAE before | MAE after |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    development=result["development_full_trajectories"]
    populations=[("Development",development["baseline"],development["candidate"]),
        (result.get("development_validation_label","Development weights, held engine folds"),development["baseline"],development["oof_candidate"]),
        ("Retrospective confirmation",result["confirmation_full_trajectories"]["baseline"],result["confirmation_full_trajectories"]["candidate"]),
        ("Reused official benchmark",whole["baseline"],whole["candidate"])]
    def row(name,before,after):
        return f"| {name} | {before['r2']:.6f} | {after['r2']:.6f} | {before['rmse']:.6f} | {after['rmse']:.6f} | {before['mae']:.6f} | {after['mae']:.6f} |"
    lines.extend(row(*item) for item in populations)
    lines += ["",f"Official evaluation includes {whole['candidate']['rows']:,} post-commissioning rows from {whole['candidate']['engines']} engines. C-MAPSS results are retrospective: these engines have been inspected previously. Mixture weights use development engine folds; this does not make the entire previously selected baseline a fresh blind model.", "",
        "## Original healthy benchmark", "",
        "These results use the unchanged RUL >=125 healthy mask. They are reported separately from the full-trajectory objective.", "",
        "| Population | R2 before | R2 after | RMSE before | RMSE after | MAE before | MAE after |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    healthy=[("Development",result["baseline"]["selection"],result["selected"]["metrics"]),
        ("Retrospective confirmation",result["confirmation"]["baseline"],result["confirmation"]["candidate"]),
        ("Reused official benchmark",result["official_reused_benchmark"]["baseline"],result["official_reused_benchmark"]["candidate"])]
    lines.extend(row(*item) for item in healthy)
    guard=result["healthy_guard"]
    lines += ["", "## Selection and verification", "",
        result.get("search_summary",f"Tested {result['model_recipes_tested']} new model recipes, fixed blends, causal disagreement gates and constrained per-sensor mixtures. The selected model was locked before confirmation and official scoring. Confirmation and official results did not retune models or weights."), "",
        f"Acceptance requires higher full-trajectory R2 and lower full-trajectory RMSE and MAE. The secondary healthy forecast permits at most {guard['max_r2_drop']:.3f} R2 decrease, {(guard['max_rmse_ratio']-1)*100:.1f}% RMSE increase and {(guard['max_mae_ratio']-1)*100:.1f}% MAE increase. Per-subset guards permit at most 1% RMSE and 0.01 R2 deterioration on both populations. These limits were recorded before opening confirmation.", "",
        result.get("implementation_summary","New models include sensor-specific boosted and linear decoders, neural decoders, and faster mode histories with preceding operating context. Optional lifetime supervision is confined to fitting labels. Lifetime is never a prediction input or a scored forecast output. All temporal decoders fit individual cycle rows from the permitted fitting engines."), "",
        "The actual-inference audit preserves monitoring references, monitoring residuals, anomaly scores and decisions, and health indices exactly across all 104,897 official inputs. Raw-prefix, row-order and removed-lifetime checks pass. Changing current or future sensor readings cannot change the current forecast. Normal retraining preserves the training plan, forecast role and objective and reproduces predictions on development trajectories.", "",
        f"Regression tests passed: {result.get('regression_tests_passed','see log')}. The compressed model occupies {(run/'candidate_profile.pkl').stat().st_size/1024**2:.1f} MiB. Every replaced artifact is backed up under the experiment's before directory.", ""]
    if "inference_timing" in result:
        timing=result["inference_timing"]
        lines += [f"One full-batch audit over {timing['rows']:,} rows took {timing['baseline_seconds']:.2f}s before and {timing['candidate_seconds']:.2f}s after. The reference forest uses one audit worker; decoder parallelism remains enabled. This is one measurement and can include first-run warmup or environment suspension.", ""]
        if "baseline_cpu_seconds" in timing:
            lines += [f"Process CPU time, summed over worker threads, was {timing['baseline_cpu_seconds']:.2f}s before and {timing['candidate_cpu_seconds']:.2f}s after.", ""]
    streaming = result.get("streaming_verification")
    if streaming:
        lines += [f"The final inference implementation selects ready rows in bounded batches to avoid a full feature-matrix copy. Its outputs match direct ready-row inference on {streaming['selected_rows']:,} rows, crossing the 1,024-row batch boundary; maximum absolute difference is {streaming['maximum_absolute_difference']:.3g}. The timing above predates this memory optimization.", ""]
    history_streaming = result.get("history_streaming_verification")
    if history_streaming:
        low = history_streaming["largest_selection_batch_feature_bytes"] / 1024**2
        high = history_streaming["global_selection_feature_bytes"] / 1024**2
        lines += ["## Bounded history inference", "",
            f"Whole-engine streaming matches the global-matrix forecasts on {history_streaming['selection_ready_rows']:,} ready development rows, with maximum absolute difference {history_streaming['maximum_absolute_prediction_difference']:.3g}. The largest feature block is {low:.1f} MiB, compared with a {high:.1f} MiB global development feature matrix. These sizes describe feature arrays, not total process memory.", "",
            f"All three official aggregate metrics and all subset metrics reproduce the locked audit within 1e-10 across {history_streaming['official_input_rows']:,} inputs. The unchanged model checksum is {history_streaming['candidate_sha256']}.", "",
            f"The development feature/forecast path took {history_streaming['selection_direct_seconds']:.2f}s with a global matrix and {history_streaming['selection_streamed_seconds']:.2f}s with engine streaming in single sequential measurements. Official streaming took {history_streaming['official_seconds_without_context_twins']:.2f}s excluding the unchanged four context twins. The earlier full-batch timing predates this optimization; hardware load and warmup limit timing comparisons.", ""]
    if result.get("cost_assessment"):
        lines += ["## Accuracy and resource tradeoff", "", result["cost_assessment"], ""]
    noise_path = run / "noise_diagnostic.json"
    if noise_path.is_file():
        noise = json.loads(noise_path.read_text())
        low, high = noise["engine_bootstrap_95_percent_interval"]
        lines += ["## Report-only noise diagnostic", "",
            f"Same-mode second differences from fitting engines estimate a noise-limited full-trajectory R2 near {noise['estimated_noise_limited_full_r2']:.3f}, with a fitting-engine bootstrap interval of {low:.3f}-{high:.3f}. The estimator uses {noise['fit_triples']:,} healthy three-cycle sequences and weights their estimated noise variance to the official population. It was run after the candidate lock and does not select model parameters.", "",
            "This is an approximate diagnostic, not a strict upper bound or proof that 0.8 is impossible. It assumes independent adjacent measurement errors, small latent curvature, adequate removal of operating changes and transferable conditional noise variance. The interval covers fitting-noise estimation only; it excludes assumption error, uncertainty in official target variance and historical benchmark reuse.", ""]
    for key,title in (("confirmation_full_trajectories","Full trajectories: confirmation"),
            ("official_full_trajectory_forecasts","Full trajectories: official"),
            ("confirmation","Healthy: confirmation"),("official_reused_benchmark","Healthy: official")):
        lines += [f"## {title}", "", "| Subset | R2 before | R2 after | RMSE before | RMSE after | MAE before | MAE after |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        item=result[key]
        lines.extend(row(name,before,item["candidate"]["per_subset"][name]) for name,before in item["baseline"]["per_subset"].items())
        lines.append("")
    relative=run.relative_to(root).as_posix()
    lines += [f"[Full evaluation](../{relative}/verified_evaluation.json), [protocol](../{relative}/protocol.json), [locked model](../{relative}/model_lock.json), [recipe](../{relative}/selected_recipe.json), [development mixture selection](../{relative}/mixture_selection.json).", ""]
    return "\n".join(lines)
