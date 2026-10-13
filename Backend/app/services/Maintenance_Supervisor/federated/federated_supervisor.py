"""
federated_supervisor.py

Federated Learning Module & Parameter Aggregator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Implements distributed client node simulation (`FederatedClient`) and Federated Averaging
(`FedAvg`) parameter aggregation (`FederatedSupervisorManager`). It allows multiple plant sites
or engine fleets to collaboratively train the Supervisor model without sharing raw sensor data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    PROCESSED_ROOT,
    REPORTS_ROOT,
    ARTIFACT_ROOT,
    SupervisorConfig,
)
from app.services.Maintenance_Supervisor.models.ppo_supervisor import PPOSupervisorModel
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_CFG = SupervisorConfig()

_REPORT_PATH: Final[Path] = REPORTS_ROOT / "federated_learning_report.json"


class FederatedClient:
    """Simulated local plant node holding private engine sensor data."""

    def __init__(self, client_id: str, df_client: pd.DataFrame, config: SupervisorConfig | None = None):
        self.client_id = client_id
        self.df_client = df_client.copy()
        self.cfg = config or _CFG
        self.model = PPOSupervisorModel(config=self.cfg)

    def train_local(self, global_weights: dict | None = None) -> dict:
        """Perform local training round starting from global parameters."""
        exclude = set(self.cfg.forbidden_features) | {"engine_id", "fd_subset", "cycle", "schema_version"}
        feature_cols = [c for c in self.df_client.columns if c not in exclude and pd.api.types.is_numeric_dtype(self.df_client[c])]

        X_local = self.df_client[feature_cols].fillna(0.0)
        y_local = self.df_client.get(self.cfg.target_column, "continue_operation")

        self.model.fit(X_local, y_local)

        if global_weights and self.model.net is not None:
            # Blend local update with global model weights
            local_weights = self.model.net.state_dict()
            blended = {}
            for k in local_weights.keys():
                blended[k] = 0.7 * local_weights[k] + 0.3 * global_weights[k]
            self.model.net.load_state_dict(blended)

        weights = self.model.net.state_dict() if self.model.net is not None else {}
        num_samples = len(self.df_client)
        return {"client_id": self.client_id, "weights": weights, "num_samples": num_samples}


class FederatedSupervisorManager:
    """Master Federated Parameter Aggregator enforcing FedAvg."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG
        self.global_model = PPOSupervisorModel(config=self.cfg)

    def federated_train_round(self, df_dataset: pd.DataFrame, num_rounds: int = 3) -> dict:
        section("FEDERATED LEARNING SIMULATION STARTED")
        start_time = time.perf_counter()

        engine_col = self.cfg.engine_id_column if self.cfg.engine_id_column in df_dataset.columns else "engine_id"
        if engine_col not in df_dataset.columns:
            df_dataset[engine_col] = 1

        engine_ids = df_dataset[engine_col].unique()
        # Partition data across simulated clients
        chunks = np.array_split(engine_ids, self.cfg.fl_num_clients)
        clients = []
        for idx, chunk in enumerate(chunks):
            client_df = df_dataset[df_dataset[engine_col].isin(chunk)]
            clients.append(FederatedClient(client_id=f"client_{idx+1}", df_client=client_df, config=self.cfg))

        logger.info("Created %d simulated federated client nodes.", len(clients))

        global_weights = None
        round_summaries = []

        for r in range(1, num_rounds + 1):
            logger.info("--- Federated Round %d/%d ---", r, num_rounds)
            client_updates = []
            total_samples = 0

            for client in clients:
                update = client.train_local(global_weights)
                client_updates.append(update)
                total_samples += update["num_samples"]

            # FedAvg Weight Aggregation
            if client_updates and client_updates[0]["weights"]:
                aggregated_weights = {}
                first_weights = client_updates[0]["weights"]

                for key in first_weights.keys():
                    weighted_sum = torch.zeros_like(first_weights[key], dtype=torch.float32)
                    for update in client_updates:
                        w = update["weights"][key].float()
                        weight_factor = update["num_samples"] / total_samples
                        weighted_sum += w * weight_factor
                    aggregated_weights[key] = weighted_sum

                global_weights = aggregated_weights
                logger.info("FedAvg successfully aggregated weights across %d clients.", len(clients))

            round_summaries.append({
                "round": r,
                "participating_clients": len(clients),
                "total_samples": total_samples,
            })

        duration = time.perf_counter() - start_time

        report = {
            "status": "success",
            "num_clients": len(clients),
            "num_rounds": num_rounds,
            "fedavg_aggregation": "completed",
            "total_dataset_samples": len(df_dataset),
            "rounds_summary": round_summaries,
            "duration_seconds": round(duration, 4),
        }

        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report, _REPORT_PATH)
        section("FEDERATED LEARNING SIMULATION COMPLETED")
        return report


def run_federated_simulation(input_path: Path | None = None) -> dict:
    if input_path is None:
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)
    manager = FederatedSupervisorManager()
    return manager.federated_train_round(df, num_rounds=3)


def main() -> int:
    try:
        res = run_federated_simulation()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in federated_supervisor: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
