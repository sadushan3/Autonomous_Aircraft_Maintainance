"""
Sensor dependency graph: the physical structure the reasoning layer walks.

C-MAPSS's 21 sensors observe a small number of physical subsystems along the
gas path -- fan, low-pressure compressor (LPC), high-pressure compressor (HPC),
combustor, high-pressure turbine (HPT), low-pressure turbine (LPT), and the
bleed/nozzle system.  A degradation in one subsystem shows up across several
sensors at once because they are thermodynamically coupled (e.g. HPC
efficiency loss raises HPC outlet temperature/pressure and downstream fuel
flow together).  This module encodes that coupling as a directed graph using
``networkx``, built once from a fixed, documented mapping -- not learned or
inferred from data.

The mapping follows the sensor descriptions published with the C-MAPSS dataset
(Saxena & Goebel, 2008): e.g. sensor_2/sensor_3/sensor_4 are HPC-adjacent
temperatures, sensor_11/sensor_12 are HPC pressures/ratios, sensor_7/sensor_9
relate to fuel flow and HPT-adjacent pressure.

This graph is used by :mod:`rule_engine` to map "which sensors deviated" to
"which subsystem is implicated", and every edge is explicitly a *dependency*,
never a claim of proven causation -- matching ``hard_causal_claim=False``
elsewhere in the reasoning layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import get_sensor_columns
from app.utils.Anomaly_Health_Monitoring.utils import get_logger

#: Sensor -> subsystem membership, following the published C-MAPSS sensor list.
#: Sensors not clearly attributable to one subsystem are grouped under
#: 'general' (e.g. fan-inlet temperature, which reflects overall gas-path state).
SENSOR_SUBSYSTEMS: Dict[str, str] = {
    "sensor_1": "fan",       # Total temperature at fan inlet
    "sensor_2": "lpc",       # Total temperature at LPC outlet
    "sensor_3": "hpc",       # Total temperature at HPC outlet
    "sensor_4": "hpc",       # Total temperature at LPT outlet (thermally HPC-coupled)
    "sensor_5": "fan",       # Pressure at fan inlet
    "sensor_6": "bypass",    # Total pressure in bypass-duct
    "sensor_7": "hpt",       # Total pressure at HPC outlet
    "sensor_8": "lpc",       # Physical fan speed
    "sensor_9": "hpt",       # Physical core speed
    "sensor_10": "bypass",   # Engine pressure ratio
    "sensor_11": "hpc",      # Static pressure at HPC outlet
    "sensor_12": "hpc",      # Ratio of fuel flow to Ps30
    "sensor_13": "lpc",      # Corrected fan speed
    "sensor_14": "hpt",      # Corrected core speed
    "sensor_15": "bleed",    # Bypass ratio
    "sensor_16": "burner",   # Burner fuel-air ratio
    "sensor_17": "bleed",    # Bleed enthalpy
    "sensor_18": "fan",      # Demanded fan speed
    "sensor_19": "bypass",   # Demanded corrected fan speed
    "sensor_20": "lpt",      # HPT coolant bleed
    "sensor_21": "lpt",      # LPT coolant bleed
}

#: Directed thermodynamic dependency edges: source subsystem influences target.
#: This is the *documented gas-path order*, not a statistically fitted graph.
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

#: Human-readable descriptions used by the explanation generator.
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
    """Builds and queries the fixed sensor-subsystem dependency graph."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise and build the graph.

        Args:
            config: Configuration (currently unused, accepted for interface
                consistency and future weighting configuration).
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        """Construct the directed sensor/subsystem graph.

        Returns:
            A :class:`networkx.DiGraph` with sensor nodes, subsystem nodes,
            sensor -> subsystem membership edges, and subsystem -> subsystem
            dependency edges.
        """
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

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def subsystem_of(self, sensor: str) -> str:
        """Return the subsystem a sensor belongs to.

        Args:
            sensor: Sensor column name.

        Returns:
            The subsystem label, or ``'general'`` when unmapped.
        """
        return SENSOR_SUBSYSTEMS.get(sensor, "general")

    def sensors_in_subsystem(self, subsystem: str) -> List[str]:
        """Return every sensor belonging to a subsystem.

        Args:
            subsystem: Subsystem label.

        Returns:
            The sensor names mapped to that subsystem.
        """
        return [s for s, sub in SENSOR_SUBSYSTEMS.items() if sub == subsystem]

    def dominant_subsystem(self, sensors: List[str]) -> str:
        """Determine which subsystem a set of contributing sensors implicates most.

        Args:
            sensors: Sensor names ranked by contribution (highest first is
                weighted most heavily via simple positional weighting).

        Returns:
            The most-implicated subsystem label, or ``'unknown'`` when the
            input is empty.
        """
        if not sensors:
            return "unknown"
        weights: Dict[str, float] = {}
        for rank, sensor in enumerate(sensors):
            subsystem = self.subsystem_of(sensor)
            weights[subsystem] = weights.get(subsystem, 0.0) + 1.0 / (rank + 1)
        return max(weights, key=weights.get)

    def downstream_subsystems(self, subsystem: str) -> List[str]:
        """Return subsystems thermodynamically downstream of the given one.

        Args:
            subsystem: A subsystem label.

        Returns:
            The reachable downstream subsystem labels.
        """
        if subsystem not in self.graph:
            return []
        return [
            node
            for node in nx.descendants(self.graph, subsystem)
            if self.graph.nodes[node].get("kind") == "subsystem"
        ]

    def upstream_subsystems(self, subsystem: str) -> List[str]:
        """Return subsystems thermodynamically upstream of the given one.

        Args:
            subsystem: A subsystem label.

        Returns:
            The reachable upstream subsystem labels.
        """
        if subsystem not in self.graph:
            return []
        return [
            node
            for node in nx.ancestors(self.graph, subsystem)
            if self.graph.nodes[node].get("kind") == "subsystem"
        ]

    def related_sensors(self, sensor: str, hops: int = 1) -> List[str]:
        """Find sensors that observe thermodynamically related subsystems.

        Used by the reasoning layer to check whether a deviation is isolated to
        one sensor or corroborated by physically coupled neighbours -- a
        multi-sensor pattern is much stronger evidence than a single reading.

        Args:
            sensor: The sensor to search around.
            hops: How many subsystem-graph hops to include.

        Returns:
            Other sensor names observing a nearby subsystem.
        """
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
        """Return a serialisable description of the graph for the API layer.

        Returns:
            A mapping of nodes, edges and subsystem membership.
        """
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
    """Build the graph standalone and print its structure."""
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
