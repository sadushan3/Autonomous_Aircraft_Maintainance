from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import get_sensor_columns
from app.utils.Anomaly_Health_Monitoring.utils import get_logger


SENSOR_SUBSYSTEMS: Dict[str, str] = {
    "sensor_1": "fan",
    "sensor_2": "lpc",
    "sensor_3": "hpc",
    "sensor_4": "hpc",
    "sensor_5": "fan",
    "sensor_6": "bypass",
    "sensor_7": "hpt",
    "sensor_8": "lpc",
    "sensor_9": "hpt",
    "sensor_10": "bypass",
    "sensor_11": "hpc",
    "sensor_12": "hpc",
    "sensor_13": "lpc",
    "sensor_14": "hpt",
    "sensor_15": "bleed",
    "sensor_16": "burner",
    "sensor_17": "bleed",
    "sensor_18": "fan",
    "sensor_19": "bypass",
    "sensor_20": "lpt",
    "sensor_21": "lpt",
}


SUBSYSTEM_EDGES: List[tuple] = [
    ("fan", "lpc"),
    ("lpc", "hpc"),
    ("hpc", "burner"),
    ("burner", "hpt"),
    ("hpt", "lpt"),
    ("lpt", "bleed"),
    ("hpc", "bleed"),
    ("hpt", "bleed"),
    ("fan", "bypass"),
    ("bypass", "lpt"),
]


SUBSYSTEM_DESCRIPTIONS: Dict[str, str] = {
    "fan": "Fan / inlet",
    "lpc": "Low-Pressure Compressor",
    "hpc": "High-Pressure Compressor",
    "burner": "Combustor / burner",
    "hpt": "High-Pressure Turbine",
    "lpt": "Low-Pressure Turbine",
    "bypass": "Bypass duct",
    "bleed": "Bleed air system",
    "general": "General gas-path",
}


class SensorDependencyGraph:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:


        graph = nx.DiGraph()

        for subsystem in SUBSYSTEM_DESCRIPTIONS:
            graph.add_node(subsystem, kind="subsystem", label=SUBSYSTEM_DESCRIPTIONS[subsystem])

        for sensor in get_sensor_columns():
            subsystem = SENSOR_SUBSYSTEMS.get(sensor, "general")
            graph.add_node(sensor, kind="sensor", subsystem=subsystem)
            graph.add_edge(sensor, subsystem, relation="observes", weight=1.0)

        for source, target in SUBSYSTEM_EDGES:
            graph.add_edge(source, target, relation="influences", weight=0.7)

        self.logger.info(
            "Sensor dependency graph built: %d node(s), %d edge(s)",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph


    def subsystem_of(self, sensor: str) -> str:


        return SENSOR_SUBSYSTEMS.get(sensor, "general")

    def sensors_in_subsystem(self, subsystem: str) -> List[str]:


        return [s for s, sub in SENSOR_SUBSYSTEMS.items() if sub == subsystem]

    def dominant_subsystem(self, sensors: List[str]) -> str:


        if not sensors:
            return "unknown"
        weights: Dict[str, float] = {}
        for rank, sensor in enumerate(sensors):
            subsystem = self.subsystem_of(sensor)
            weights[subsystem] = weights.get(subsystem, 0.0) + 1.0 / (rank + 1)
        return max(weights, key=weights.get)

    def downstream_subsystems(self, subsystem: str) -> List[str]:


        if subsystem not in self.graph:
            return []
        return [
            node
            for node in nx.descendants(self.graph, subsystem)
            if self.graph.nodes[node].get("kind") == "subsystem"
        ]

    def upstream_subsystems(self, subsystem: str) -> List[str]:


        if subsystem not in self.graph:
            return []
        return [
            node
            for node in nx.ancestors(self.graph, subsystem)
            if self.graph.nodes[node].get("kind") == "subsystem"
        ]

    def related_sensors(self, sensor: str, hops: int = 1) -> List[str]:


        subsystem = self.subsystem_of(sensor)
        nearby = {subsystem}
        frontier = {subsystem}
        for _ in range(max(hops, 0)):
            next_frontier = set()
            for node in frontier:
                next_frontier |= set(self.graph.successors(node)) | set(self.graph.predecessors(node))
            next_frontier = {
                n for n in next_frontier if self.graph.nodes.get(n, {}).get("kind") == "subsystem"
            }
            nearby |= next_frontier
            frontier = next_frontier

        related = [
            s for s, sub in SENSOR_SUBSYSTEMS.items() if sub in nearby and s != sensor
        ]
        return related

    def describe(self) -> Dict[str, object]:


        nodes = list(self.graph.nodes)
        edges = [
            {
                "source": u,
                "target": v,
                "relation": data.get("relation", "influences"),
                "weight": float(data.get("weight", 1.0)),
            }
            for u, v, data in self.graph.edges(data=True)
        ]
        subsystems = {
            subsystem: self.sensors_in_subsystem(subsystem)
            for subsystem in SUBSYSTEM_DESCRIPTIONS
        }
        return {
            "nodes": nodes,
            "edges": edges,
            "subsystems": subsystems,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        }


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    graph = SensorDependencyGraph(config)
    description = graph.describe()
    logger.info("Graph: %d nodes, %d edges", description["n_nodes"], description["n_edges"])
    logger.info("HPC sensors: %s", graph.sensors_in_subsystem("hpc"))
    logger.info("Dominant subsystem for ['sensor_3','sensor_11']: %s", graph.dominant_subsystem(["sensor_3", "sensor_11"]))
    logger.info("Downstream of hpc: %s", graph.downstream_subsystems("hpc"))


if __name__ == "__main__":
    main()
