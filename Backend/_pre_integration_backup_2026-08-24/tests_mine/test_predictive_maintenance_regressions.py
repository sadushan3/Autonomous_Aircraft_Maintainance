"""Regression tests for the Predictive Maintenance defects found in the audit.

Written with :mod:`unittest` so they run on the bundled interpreter with no new
dependency::

    python -m unittest discover -s tests -t .

``pytest tests`` collects them unchanged if pytest is installed later.

Every test here pins a specific defect that shipped, so each one names the
behaviour it prevents from coming back.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Predictive_Maintenance.config import Settings  # noqa: E402
from app.services.Predictive_Maintenance.adaptive_threshold_service import (  # noqa: E402
    AdaptiveThresholdService,
)
from app.services.Predictive_Maintenance.failure_risk_service import (  # noqa: E402
    FailureRiskService,
)
from app.services.Predictive_Maintenance.preprocessing import (  # noqa: E402
    PreprocessingService,
)
from app.services.Predictive_Maintenance.rul_lifecycle_service import (  # noqa: E402
    RULLifecycleService,
)


class RiskStateConsistencyTest(unittest.TestCase):
    """``risk_state`` must never contradict ``predicted_fail_in_30``.

    Both fields are emitted from the same response and consumed together by the
    Maintenance Supervisor, which reads ``action_hint_for_supervisor`` (derived
    from ``risk_state``). The shipped implementation floored the comparison at
    ``max(0.80, threshold_30)``, so on any subset with a calibrated cut below
    0.80 -- FD002's is 0.031 -- a window could report "will fail within 30
    cycles" and "Stable / continue_monitoring" simultaneously.
    """

    def test_low_calibrated_threshold_still_reports_critical(self) -> None:
        state = FailureRiskService._risk_state(
            risk_30=0.10, threshold_30=0.031, risk_velocity=0.0, risk_acceleration=0.0
        )
        self.assertEqual(state, "Critical")

    def test_no_contradiction_across_threshold_grid(self) -> None:
        for threshold in (0.005, 0.031, 0.091, 0.5, 0.76, 0.8, 0.95):
            for risk in (0.0, 0.001, 0.03, 0.1, 0.5, 0.79, 0.85, 0.999):
                predicted_fail = risk >= threshold
                state = FailureRiskService._risk_state(
                    risk_30=risk,
                    threshold_30=threshold,
                    risk_velocity=0.0,
                    risk_acceleration=0.0,
                )
                if predicted_fail:
                    self.assertEqual(
                        state,
                        "Critical",
                        msg=(
                            f"risk={risk} threshold={threshold} flagged failure "
                            f"but reported {state!r}"
                        ),
                    )

    def test_action_hint_escalates_whenever_failure_predicted(self) -> None:
        state = FailureRiskService._risk_state(
            risk_30=0.10, threshold_30=0.031, risk_velocity=0.0, risk_acceleration=0.0
        )
        self.assertEqual(
            FailureRiskService._action_hint(state),
            "maintenance_supervisor_priority_review",
        )


class ConfidenceCollapseTest(unittest.TestCase):
    """The collapse flag must depend on the input, not only on config constants.

    ``confidence_collapse_flag`` was ``2 * q_hat > RUL_UNCERTAINTY_WIDTH_THRESHOLD``
    and the config sets that threshold to exactly ``2 * q_hat`` for all four
    subsets, so the flag was False on every input ever scored. That made
    ``instability_flag`` always False and left the baseline fallback, the
    "Unstable" lifecycle state and three response fields unreachable.
    """

    def test_flag_is_not_constant_across_inputs(self) -> None:
        q_hat = 25.0
        observed = {
            RULLifecycleService.is_confidence_collapsed(rul, q_hat)
            for rul in (1.0, 5.0, 20.0, 30.0, 80.0, 200.0)
        }
        self.assertEqual(observed, {True, False}, "flag is still input-independent")

    def test_wide_band_around_small_estimate_collapses(self) -> None:
        self.assertTrue(RULLifecycleService.is_confidence_collapsed(10.0, 25.0))

    def test_same_band_around_large_estimate_does_not(self) -> None:
        self.assertFalse(RULLifecycleService.is_confidence_collapsed(200.0, 25.0))

    def test_width_matches_reported_lower_bound_clamping(self) -> None:
        predicted_rul, q_hat = 10.0, 25.0
        lower = max(0.0, predicted_rul - q_hat)
        upper = predicted_rul + q_hat
        self.assertAlmostEqual(
            RULLifecycleService.interval_width(predicted_rul, q_hat), upper - lower
        )

    def test_config_no_longer_makes_the_flag_dead(self) -> None:
        settings = Settings()
        for fd_subset, q_hat in settings.DEFAULT_RUL_Q_HAT.items():
            self.assertTrue(
                RULLifecycleService.is_confidence_collapsed(q_hat / 2.0, q_hat),
                msg=f"{fd_subset}: collapse unreachable at low RUL",
            )


class FeatureAlignmentTest(unittest.TestCase):
    """Zero-filling a missing feature must be recorded, not silent.

    A feature absent from the payload is filled with ``0.0`` and then scaled,
    which maps it to an extreme z-score rather than to "no information". The
    fill is kept (op_cluster one-hots are legitimately absent) but is now
    reported so the caller can see that a prediction is degraded.
    """

    def setUp(self) -> None:
        self.service = PreprocessingService()
        self.frame = pd.DataFrame({"op_setting_1": [1.0, 2.0], "sensor_1": [10.0, 11.0]})

    def test_absent_feature_is_recorded(self) -> None:
        aligned = self.service._align_features(
            self.frame, ["op_setting_1", "sensor_1", "sensor_99"]
        )
        self.assertEqual(aligned.attrs["zero_filled_features"], ["sensor_99"])

    def test_op_cluster_one_hots_are_not_reported_as_missing(self) -> None:
        aligned = self.service._align_features(
            self.frame, ["op_setting_1", "sensor_1", "op_cluster_0", "op_cluster_1"]
        )
        self.assertEqual(aligned.attrs["zero_filled_features"], [])

    def test_fully_supplied_frame_reports_nothing(self) -> None:
        aligned = self.service._align_features(self.frame, ["op_setting_1", "sensor_1"])
        self.assertEqual(aligned.attrs["zero_filled_features"], [])
        self.assertEqual(aligned.attrs["non_numeric_features"], [])

    def test_values_are_unchanged_by_the_reporting(self) -> None:
        aligned = self.service._align_features(self.frame, ["op_setting_1", "sensor_1"])
        self.assertEqual(aligned["sensor_1"].tolist(), [10.0, 11.0])


class AdaptiveThresholdSeedingTest(unittest.TestCase):
    """Feedback must adapt the calibrated threshold, never replace it.

    ``FAILURE_RISK_DEFAULT_THRESHOLDS`` is a flat 0.5 placeholder. An entry
    seeded from it discards the model-calibrated value on the very first
    operator click -- which is how FD001's 30-cycle cut came to serve 0.488 in
    place of its calibrated 0.8 after three clicks.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "adaptive_thresholds.json"
        self.service = AdaptiveThresholdService(
            self.path, {"FD001": {"10": 0.5, "30": 0.5, "50": 0.5}}
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_feedback_defers_to_calibrated(self) -> None:
        self.assertEqual(self.service.get_threshold("FD001", "30", 0.8), 0.8)

    def test_placeholder_seeded_entry_does_not_override_calibrated(self) -> None:
        self.service.record_feedback("FD001", "30", was_false_alarm=True)
        self.assertEqual(self.service.get_threshold("FD001", "30", 0.8), 0.8)

    def test_calibrated_seeded_entry_is_served(self) -> None:
        self.service.record_feedback("FD001", "30", was_false_alarm=True, seed=0.8)
        served = self.service.get_threshold("FD001", "30", 0.8)
        self.assertNotEqual(served, 0.8)
        self.assertGreater(served, 0.8, "a false alarm must raise the bar")
        self.assertLess(served, 0.9, "one click must not swing the threshold far")

    def test_confirmed_event_lowers_the_bar(self) -> None:
        self.service.record_feedback("FD001", "30", was_false_alarm=False, seed=0.8)
        self.assertLess(self.service.get_threshold("FD001", "30", 0.8), 0.8)

    def test_legacy_entry_without_provenance_is_ignored(self) -> None:
        self.path.write_text(
            '{"FD001": {"30": {"current": 0.488, "previous": 0.476, '
            '"false_alarm_count": 1, "confirmed_count": 2, "last_updated": null}}}',
            encoding="utf-8",
        )
        self.assertEqual(self.service.get_threshold("FD001", "30", 0.8), 0.8)


if __name__ == "__main__":
    unittest.main()
