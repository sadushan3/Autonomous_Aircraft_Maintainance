# AI System for Autonomous Predictive Maintenance of Aircraft Engines

Production-ready FastAPI backend for serving two trained AI components exported from Google Colab:

- Autonomous RUL Reasoning and Lifecycle Planning Agent
- Autonomous Failure Risk Reasoning Agent

The API loads TensorFlow/Keras models, scikit-learn scalers, feature column JSON files, metadata JSON files, and manifests from local artifact folders. It exposes REST endpoints for health checks, artifact inspection, single-agent prediction, combined prediction, and demo requests.

## Folder Structure

```text
predictive-maintenance-backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas/
│   ├── services/
│   ├── routers/
│   └── utils/
├── artifacts/
│   ├── RUL_Lifecycle/FD001..FD004/
│   └── Failure_Risk/FD001..FD004/
├── sample_data/
├── tests/
├── requirements.txt
├── .env.example
├── README.md
└── run.py
```

## Copy Model Artifacts

Copy Colab-exported files into the matching FD subset folders.

Preferred backend-local layout:

RUL Lifecycle Agent:

```text
artifacts/RUL_Lifecycle/FD001/
├── FD001_best_tcn_FINAL.keras
├── FD001_scaler_FINAL.pkl
└── FD001_feature_columns_FINAL.json
```

Failure Risk Agent:

```text
artifacts/Failure_Risk/FD001/
├── FD001_bilstm_mc_failure_risk_FINAL.keras
├── FD001_scaler_FINAL.pkl
├── FD001_feature_columns_FINAL.json
└── FD001_bilstm_failure_risk_metadata_FINAL.json
```

Repeat for `FD002`, `FD003`, and `FD004`. The backend also supports flexible discovery with these patterns: `*best_tcn*.keras`, `*bilstm*.keras`, `*scaler*.pkl`, `*feature_columns*.json`, and `*metadata*.json`.

The artifact loader also supports these fallback layouts:

```text
models/RUL_Lifecycle/FD001/
models/Failure_Risk/FD001/
```

Flat component folders are also supported, as long as filenames include the FD subset:

```text
models/RUL_Lifecycle/FD001_best_tcn_FINAL.keras
models/Failure_Risk/FD001_bilstm_mc_failure_risk_FINAL.keras
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Run Server

```bash
python run.py
```

OpenAPI docs are available at:

```text
http://localhost:8000/docs
```

## API Examples

Health:

```bash
curl http://localhost:8000/api/health
```

Artifact status:

```bash
curl http://localhost:8000/api/artifacts/status
```

RUL prediction:

```bash
curl -X POST http://localhost:8000/api/rul/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

Failure risk prediction:

```bash
curl -X POST http://localhost:8000/api/failure-risk/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

Combined prediction:

```bash
curl -X POST http://localhost:8000/api/predict/combined \
  -H "Content-Type: application/json" \
  -d @request.json
```

Demo request body:

```bash
curl http://localhost:8000/api/demo/sample-request
```

Demo prediction using the generated sample body:

```bash
curl -X POST http://localhost:8000/api/demo/predict
```

## Postman

Import this collection into Postman:

```text
postman/Predictive_Maintenance_Backend.postman_collection.json
```

The collection uses `{{baseUrl}}`, defaulting to `http://localhost:8000`, and generates valid 70-cycle sample request bodies automatically in a collection pre-request script.

## Example JSON Request

```json
{
  "fd_subset": "FD001",
  "engine_id": "engine-001",
  "sequence": [
    {
      "cycle": 101,
      "op_setting_1": -0.0007,
      "op_setting_2": -0.0004,
      "op_setting_3": 100.0,
      "sensor_1": 518.67,
      "sensor_2": 642.58,
      "sensor_3": 1588.43,
      "sensor_4": 1401.32,
      "sensor_5": 14.62,
      "sensor_6": 21.61,
      "sensor_7": 553.90,
      "sensor_8": 2388.04,
      "sensor_9": 9050.17,
      "sensor_10": 1.30,
      "sensor_11": 47.20,
      "sensor_12": 521.72,
      "sensor_13": 2388.03,
      "sensor_14": 8125.55,
      "sensor_15": 8.4052,
      "sensor_16": 0.03,
      "sensor_17": 392.0,
      "sensor_18": 2388.0,
      "sensor_19": 100.0,
      "sensor_20": 39.08,
      "sensor_21": 23.3186
    }
  ],
  "previous_rul_predictions": [60.0, 58.5, 55.2],
  "previous_risk_scores": [0.1, 0.2, 0.3],
  "include_uncertainty": true,
  "include_debug": true
}
```

The real request must include at least 30 sequence rows. RUL models may require larger windows, such as 50 or 70 cycles, based on configuration.

## Example Combined Response

```json
{
  "engine_id": "engine-001",
  "fd_subset": "FD001",
  "cycle": 145,
  "rul_agent": {
    "fd_subset": "FD001",
    "predicted_rul": 34.7,
    "lower_bound": 14.7,
    "upper_bound": 54.7,
    "uncertainty_width": 40.0,
    "lifecycle_state": "Degrading",
    "degradation_speed": 1.2,
    "degradation_acceleration": 0.3,
    "confidence_collapse_flag": false,
    "sudden_health_drop_flag": false,
    "instability_flag": false,
    "model_used": "TCN_Conformal"
  },
  "failure_risk_agent": {
    "fd_subset": "FD001",
    "risk_10": 0.18,
    "risk_30": 0.64,
    "risk_50": 0.79,
    "uncertainty_10": 0.03,
    "uncertainty_30": 0.06,
    "uncertainty_50": 0.05,
    "threshold_10": 0.5,
    "threshold_30": 0.5,
    "threshold_50": 0.5,
    "predicted_fail_in_10": false,
    "predicted_fail_in_30": true,
    "predicted_fail_in_50": true,
    "control_risk": 0.64,
    "risk_velocity": 0.12,
    "risk_acceleration": 0.04,
    "risk_state": "Rising",
    "action_hint_for_supervisor": "monitor_closely",
    "model_used": "BiLSTM_MC_Dropout"
  },
  "supervisor_pre_decision": {
    "overall_health_state": "Degrading_Rising_Risk",
    "priority": "Medium",
    "recommendation": "monitor_closely",
    "reason": "RUL indicates degradation or 30-cycle failure risk is rising."
  }
}
```

## Tests

```bash
pytest
```

The included tests verify health, validation behavior, and artifact inspection. They do not require real model artifacts.

For a deeper local check after copying artifacts:

```bash
python scripts/check_backend.py
```

This prints registered routes, discovered artifact paths, sample request validation, and prediction output when artifacts are available.

## Limitations

- Models are trained on C-MAPSS simulation data.
- Failure labels are RUL-derived and should be interpreted as decision-support signals.
- This backend is decision support only and is not aircraft-certified software.
- The RL supervisor is not implemented here; the combined endpoint provides rule-based pre-decision support only.
