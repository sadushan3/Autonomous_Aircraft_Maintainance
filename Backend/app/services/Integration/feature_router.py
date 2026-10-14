"""
Feature Router -- one uploaded CSV, component-specific feature views.

The system takes a single user-uploaded C-MAPSS file and feeds three
independently-trained components. They do *not* want the same columns, and
they emphatically do not want the same *preparation*: each owns a scaler and a
feature list that was fitted at its own training time. So this router does
exactly two things and nothing more:

1. **Validate and narrow once.** Confirm the 26 canonical C-MAPSS columns are
   present, then drop everything else -- foreign columns produced by sibling
   components (``rul_prediction``, ``failure_risk_score``, ...) are ignored,
   never an error.
2. **Hand each component the raw columns it declares it needs**, under the
   identity naming that component speaks.

What it deliberately does **not** do is transform values. Both downstream
branches apply their own persisted scaler:

* Anomaly & Health fits per-operating-mode scalers inside its own pipeline.
* RUL / Failure Risk loads ``{subset}_scaler_FINAL.pkl`` and aligns to
  ``{subset}_feature_columns_FINAL.json``.

Scaling here would double-scale one branch or the other and silently
invalidate a trained model, so the router emits raw values and lets each
component prepare them. That is the whole reason the split lives here rather
than in a shared preprocessing step.

Scope note: routing stops at the two component branches. The Maintenance
Supervisor consumes their *outputs* from the exchange layer and is not a
routing target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from app.utils.Anomaly_Health_Monitoring.model_utils import (
    SUBSET_COLUMN,
    get_identity_columns,
    get_op_setting_columns,
    get_required_columns,
    get_sensor_columns,
    select_required_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import get_logger

logger = get_logger(__name__)

#: Identity naming each branch speaks. Internally every Anomaly carry-forward
#: CSV is keyed on ``unit``; the Predictive-Maintenance schemas and the
#: exchange layer use ``engine_id``.
ANOMALY_ID_COLUMN = "unit"
PREDICTIVE_ID_COLUMN = "engine_id"


@dataclass
class FeatureView:
    """One component's slice of the uploaded data.

    Attributes:
        component: Which branch this view feeds.
        frame: The narrowed frame, in the identity naming that branch expects.
        columns: The columns handed over, in order.
        id_column: Name of the engine-identity column in ``frame``.
        notes: Anything the caller should know about this view.
    """

    component: str
    frame: pd.DataFrame
    columns: List[str]
    id_column: str
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-safe description of the view.

        Returns:
            Row/engine counts and the routed column list.
        """
        return {
            "component": self.component,
            "rows": int(len(self.frame)),
            "engines": int(self.frame[self.id_column].nunique()) if len(self.frame) else 0,
            "n_columns": len(self.columns),
            "columns": self.columns,
            "id_column": self.id_column,
            "notes": self.notes,
        }


class FeatureRouter:
    """Splits one validated C-MAPSS frame into per-component feature views."""

    #: Columns the Anomaly & Health component consumes. It derives operating
    #: mode, engineered features and per-mode scaling itself, so it needs only
    #: the raw canonical set.
    ANOMALY_FEATURES: List[str] = get_op_setting_columns() + get_sensor_columns()

    #: Columns the RUL / Failure Risk component consumes, mirroring
    #: ``app/services/Predictive_Maintenance/preprocessing.py::RAW_FEATURE_COLUMNS``.
    #: Identical to the anomaly set today -- both branches read the same raw
    #: physics -- but declared separately on purpose: these two lists are owned
    #: by different components and are free to diverge without one silently
    #: dragging the other along.
    PREDICTIVE_FEATURES: List[str] = get_op_setting_columns() + get_sensor_columns()

    def __init__(self, carry_columns: Optional[Sequence[str]] = None) -> None:
        """Initialise the router.

        Args:
            carry_columns: Extra columns to preserve on every view when the
                upload happens to contain them (for example ``RUL`` for
                offline evaluation, or ``subset_id`` so the Predictive
                branch can pick the right per-subset artefacts).
        """
        self.carry_columns = list(
            carry_columns if carry_columns is not None else ["RUL", SUBSET_COLUMN]
        )

    # ------------------------------------------------------------------ #
    # Common stage
    # ------------------------------------------------------------------ #
    def validate_and_narrow(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate the canonical schema once, then drop foreign columns.

        Args:
            frame: The raw uploaded frame.

        Returns:
            The frame narrowed to the 26 required columns plus any carried
            extras that were present.

        Raises:
            SchemaValidationError: If a required C-MAPSS column is missing,
                naming exactly which ones.
        """
        narrowed = select_required_columns(frame, extra_keep=self.carry_columns)
        logger.info(
            "Router validated upload: %d rows, %d engines, %d columns retained",
            len(narrowed),
            narrowed[ANOMALY_ID_COLUMN].nunique() if ANOMALY_ID_COLUMN in narrowed else 0,
            narrowed.shape[1],
        )
        return narrowed

    # ------------------------------------------------------------------ #
    # Per-component views
    # ------------------------------------------------------------------ #
    def _view(
        self,
        narrowed: pd.DataFrame,
        component: str,
        features: Sequence[str],
        id_column: str,
        notes: Sequence[str] = (),
    ) -> FeatureView:
        """Build one component's view from the narrowed frame.

        Args:
            narrowed: Output of :meth:`validate_and_narrow`.
            component: Branch name.
            features: Feature columns that branch declares it needs.
            id_column: Identity column name that branch expects.
            notes: Caller-facing remarks.

        Returns:
            The assembled :class:`FeatureView`.
        """
        wanted = list(features)
        carried = [c for c in self.carry_columns if c in narrowed.columns]
        keep = get_identity_columns() + wanted + carried

        view = narrowed.loc[:, [c for c in keep if c in narrowed.columns]].copy()
        if id_column != ANOMALY_ID_COLUMN:
            view = view.rename(columns={ANOMALY_ID_COLUMN: id_column})

        ordered = [id_column, "cycle"] + wanted + carried
        ordered = [c for c in ordered if c in view.columns]
        view = view.loc[:, ordered]

        missing = [c for c in wanted if c not in view.columns]
        note_list = list(notes)
        if missing:
            note_list.append(f"missing declared features: {missing}")
            logger.warning("%s view is missing declared features: %s", component, missing)

        return FeatureView(component, view, ordered, id_column, note_list)

    def anomaly_view(self, narrowed: pd.DataFrame) -> FeatureView:
        """Build the Anomaly & Health Monitoring feature view.

        Args:
            narrowed: Output of :meth:`validate_and_narrow`.

        Returns:
            A view keyed on ``unit`` carrying raw settings and sensors.
        """
        return self._view(
            narrowed,
            "Anomaly_Health_Monitoring",
            self.ANOMALY_FEATURES,
            ANOMALY_ID_COLUMN,
            notes=[
                "raw values -- this component fits its own per-operating-mode "
                "scalers downstream; pre-scaling here would double-scale it",
            ],
        )

    def predictive_view(self, narrowed: pd.DataFrame) -> FeatureView:
        """Build the RUL / Failure Risk feature view.

        Args:
            narrowed: Output of :meth:`validate_and_narrow`.

        Returns:
            A view keyed on ``engine_id`` carrying raw settings and sensors.
        """
        return self._view(
            narrowed,
            "Predictive_Maintenance",
            self.PREDICTIVE_FEATURES,
            PREDICTIVE_ID_COLUMN,
            notes=[
                "raw values -- this component applies its own persisted "
                "{subset}_scaler_FINAL.pkl and aligns to "
                "{subset}_feature_columns_FINAL.json",
                "its models also expect op_cluster_* one-hot features; nothing "
                "currently supplies an 'op_cluster' column, so those are "
                "zero-filled by its own _align_features (see route() notes)",
            ],
        )

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def route(self, frame: pd.DataFrame) -> Dict[str, FeatureView]:
        """Validate once, then produce every component view.

        Args:
            frame: The raw uploaded frame.

        Returns:
            A mapping of component name to its :class:`FeatureView`.

        Raises:
            SchemaValidationError: If a required C-MAPSS column is missing.
        """
        narrowed = self.validate_and_narrow(frame)
        views = {
            "Anomaly_Health_Monitoring": self.anomaly_view(narrowed),
            "Predictive_Maintenance": self.predictive_view(narrowed),
        }
        for name, view in views.items():
            logger.info(
                "Routed -> %-26s %d rows x %d cols (id=%s)",
                name,
                len(view.frame),
                len(view.columns),
                view.id_column,
            )
        return views

    def describe(self) -> Dict[str, Any]:
        """Return the routing contract, for the API and for documentation.

        Returns:
            A JSON-safe description of what each branch receives.
        """
        return {
            "required_upload_columns": get_required_columns(),
            "carried_when_present": self.carry_columns,
            "routes": {
                "Anomaly_Health_Monitoring": {
                    "id_column": ANOMALY_ID_COLUMN,
                    "features": self.ANOMALY_FEATURES,
                    "scaling": "component-owned (per operating mode)",
                },
                "Predictive_Maintenance": {
                    "id_column": PREDICTIVE_ID_COLUMN,
                    "features": self.PREDICTIVE_FEATURES,
                    "scaling": "component-owned (persisted per-subset scaler)",
                },
            },
            "not_a_routing_target": [
                "Maintenance_Supervisor -- consumes component outputs from the "
                "exchange layer, not routed features",
            ],
        }


def main() -> None:
    """Print the routing contract, and route raw_data.csv if it exists."""
    import json
    from pathlib import Path

    from app.config.Anomaly_Health_Monitoring.config import get_config

    print(json.dumps(FeatureRouter().describe(), indent=2))

    config = get_config()
    source = config.paths.processed_path(config.paths.raw_data_csv)
    if Path(source).is_file():
        frame = pd.read_csv(source)
        for name, view in FeatureRouter().route(frame).items():
            print(f"\n{name}: {json.dumps(view.summary()['n_columns'])} cols, "
                  f"{view.summary()['rows']} rows, id={view.id_column}")


if __name__ == "__main__":
    main()
