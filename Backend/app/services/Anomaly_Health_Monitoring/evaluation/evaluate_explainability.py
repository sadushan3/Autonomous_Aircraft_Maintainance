from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json, get_logger, read_csv


class ExplainabilityEvaluator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def _completeness(self, explanations: pd.DataFrame) -> Dict[str, Any]:


        required_fields = ["why", "which_sensors", "what_pattern", "how_severe"]
        valid = pd.DataFrame(False, index=explanations.index, columns=required_fields)
        for field in required_fields:
            if field not in explanations:
                continue
            if field == "which_sensors":
                valid[field] = explanations[field].map(lambda v: bool(self._sensor_list(v)))
            else:
                valid[field] = explanations[field].notna() & ~explanations[field].astype(str).str.strip().str.lower().isin(
                    ["", "nan", "none", "null", "[]", "{}"])
        missing_by_field = {field: int((~valid[field]).sum()) for field in required_fields}
        complete_rows = valid.all(axis=1)
        return {
            "completeness_fraction": round(float(complete_rows.mean()), 4) if len(valid) else 0.,
            "missing_by_field": missing_by_field,
            "missing_columns": [f for f in required_fields if f not in explanations],
        }

    @staticmethod
    def _literal(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return value
        try:
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return None

    @classmethod
    def _sensor_list(cls, value: Any) -> Optional[list]:
        parsed = cls._literal(value)
        if (not isinstance(parsed, list) or not parsed or
                not all(isinstance(v, str) and v.startswith("sensor_") for v in parsed) or
                len(set(parsed)) != len(parsed)):
            return None
        return parsed

    def _consistency(
        self, explanations: pd.DataFrame, root_causes: pd.DataFrame
    ) -> Dict[str, Any]:


        if explanations.empty or root_causes.empty:
            return {"consistency_fraction": None, "rows_checked": 0}
        if "which_sensors" not in explanations.columns or "top_sensors" not in root_causes.columns:
            return {"consistency_fraction": None, "rows_checked": 0}

        keys = ["unit", "cycle"]
        if any(c not in data for data in (explanations, root_causes) for c in keys):
            return {"consistency_fraction": 0., "rows_checked": 0, "identity_error": "missing keys"}
        if explanations.duplicated(keys).any() or root_causes.duplicated(keys).any():
            return {"consistency_fraction": 0., "rows_checked": 0, "identity_error": "duplicate keys"}
        merged = explanations.merge(
            root_causes[keys + ["top_sensors"]], on=keys, how="left", validate="one_to_one", indicator=True
        )
        if merged.empty:
            return {"consistency_fraction": None, "rows_checked": 0}

        matches = merged.apply(
            lambda row: self._sensor_list(row["which_sensors"]) is not None and
                        self._sensor_list(row["which_sensors"]) == self._sensor_list(row["top_sensors"]),
            axis=1,
        )
        return {
            "consistency_fraction": round(float(matches.mean()), 4),
            "rows_checked": int(len(merged)),
            "unmatched_rows": int((merged["_merge"] != "both").sum()),
            "root_cause_coverage_fraction": len(merged) / len(root_causes),
        }

    def _fidelity(self, explanations: pd.DataFrame) -> Dict[str, Any]:
        tolerance = float(self.config.explainability.fidelity_tolerance)
        status = explanations.get("shap_status", pd.Series("unavailable", index=explanations.index))
        computed = explanations[status.eq("computed")]
        errors, reference_errors, valid = [], [], []
        for _, row in computed.iterrows():
            try:
                entries = self._literal(row.shap_top_features)
                if not isinstance(entries, list) or not entries:
                    raise ValueError("No SHAP values")
                contributions = np.asarray([v["shap_value"] for v in entries], float)
                base, prediction = float(row.shap_base_value), float(row.shap_prediction)
                reference_error = float(row.shap_reference_error)
                error = abs(base + contributions.sum() - prediction)
                finite = np.isfinite([base, prediction, reference_error, *contributions]).all()
                scope = row.get("shap_explained_output") == "monitoring_reference_expected_sensor"
                valid.append(bool(finite and scope and error <= tolerance and reference_error <= tolerance))
                errors.append(float(error)); reference_errors.append(reference_error)
            except (AttributeError, TypeError, ValueError, KeyError):
                valid.append(False)
        health_valid = []
        health_status = explanations.get("health_explanation_status", pd.Series("unavailable", index=explanations.index))
        for _, row in explanations[health_status.eq("computed")].iterrows():
            evidence = self._literal(row.get("health_evidence"))
            try:
                terms = float(evidence["current_cycle_term"]) + float(evidence["history_term"])
                saved = float(evidence["health_index"])
                reconstructed = float(evidence["reconstructed_health_index"])
                expected_raw = 100. * np.exp(-float(evidence["decay_lambda"]) * float(evidence["excess_over_healthy"]))
                state_valid = True
                if "health_score" in evidence:
                    health_score = float(evidence["health_score"])
                    lower, upper = float(evidence["state_lower_bound"]), evidence["state_upper_bound"]
                    state_valid = (abs(float(evidence["reconstructed_health_score"]) - health_score) <= tolerance and
                                   abs(reconstructed + float(evidence["health_score_trend_adjustment"]) - health_score) <= tolerance and
                                   evidence["state_matches"] and health_score >= lower and
                                   (upper is None or health_score < float(upper)))
                health_valid.append(bool(np.isfinite([terms, saved, reconstructed, expected_raw]).all() and
                                         state_valid and
                                         max(abs(terms - saved), abs(saved - reconstructed),
                                             abs(expected_raw - float(evidence["raw_health_index"]))) <= tolerance))
            except (TypeError, ValueError, KeyError):
                health_valid.append(False)
        quantitative = []
        for _, row in explanations.iterrows():
            entries = self._literal(row.get("sensor_evidence"))
            try:
                numeric = [float(row.anomaly_score), float(row.threshold_used)]
                numeric.extend(float(v[k]) for v in entries for k in ("observed", "expected", "residual", "residual_z"))
                residuals_agree = all(abs(float(v["observed"]) - float(v["expected"]) - float(v["residual"])) <= tolerance
                                     for v in entries)
                quantitative.append(bool(entries and np.isfinite(numeric).all() and residuals_agree))
            except (TypeError, ValueError, AttributeError, KeyError):
                quantitative.append(False)
        return {
            "shap_rows_computed": len(computed), "shap_rows_failed": int(status.eq("failed").sum()),
            "shap_rows_not_sampled": int(status.eq("not_sampled").sum()),
            "shap_rows_disabled": int(status.eq("disabled").sum()),
            "shap_coverage_fraction": len(computed) / len(explanations) if len(explanations) else 0.,
            "shap_valid_fraction": float(np.mean(valid)) if valid else None,
            "max_additivity_error": max(errors) if errors else None,
            "max_reference_error": max(reference_errors) if reference_errors else None,
            "health_rows_computed": len(health_valid),
            "health_coverage_fraction": len(health_valid) / len(explanations) if len(explanations) else 0.,
            "health_valid_fraction": float(np.mean(health_valid)) if health_valid else None,
            "quantitative_evidence_fraction": float(np.mean(quantitative)) if quantitative else 0.,
        }

    def evaluate(
        self, explanations: pd.DataFrame, root_causes: pd.DataFrame
    ) -> Dict[str, Any]:


        completeness = self._completeness(explanations)
        consistency = self._consistency(explanations, root_causes)
        fidelity = self._fidelity(explanations)
        shap_requested = not (len(explanations) and fidelity["shap_rows_disabled"] == len(explanations))
        acceptance = {
            "mostly_complete": completeness["completeness_fraction"] >= 0.95,
            "sensor_consistency": consistency.get("consistency_fraction") == 1.,
            "quantitative_evidence": fidelity["quantitative_evidence_fraction"] >= .95,
            "shap_valid_for_attempted_rows": (fidelity["shap_valid_fraction"] == 1. and fidelity["shap_rows_failed"] == 0) if shap_requested else None,
            "health_fidelity": fidelity["health_coverage_fraction"] >= .95 and fidelity["health_valid_fraction"] == 1.,
        }
        acceptance["automated_checks_passed"] = all(v for v in acceptance.values() if v is not None)

        report: Dict[str, Any] = {
            "rows_explained": int(len(explanations)),
            "completeness": completeness,
            "consistency": consistency,
            "shap_computed": fidelity["shap_rows_computed"] > 0,
            "shap_requested": bool(shap_requested), "fidelity": fidelity,
            "acceptance": acceptance,
            "validation_scope": "Automated completeness, identity, evidence and numeric fidelity checks. Physical root causes still require expert/label validation.",
        }
        self.logger.info("Explainability evaluation: %s", report["acceptance"])
        return report

    def run(
        self,
        explanations: Optional[pd.DataFrame] = None,
        root_causes: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:


        self.config.ensure_directories()
        explanation_source = explanations
        if explanation_source is None:
            path = self.config.paths.processed_path(self.config.paths.explanation_reports_csv)
            explanation_source = read_csv(path, stage="evaluate_explainability") if path.is_file() else pd.DataFrame()

        root_cause_source = root_causes
        if root_cause_source is None:
            path = self.config.paths.processed_path(self.config.paths.root_cause_analysis_csv)
            root_cause_source = read_csv(path, stage="evaluate_explainability") if path.is_file() else pd.DataFrame()

        report = self.evaluate(explanation_source, root_cause_source)
        atomic_write_json(report, self.config.paths.reports / "explainability_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_explainability.json")
        return report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = ExplainabilityEvaluator(config)
    report = evaluator.run()
    logger.info("Explainability evaluation report: %s", report)


if __name__ == "__main__":
    main()
