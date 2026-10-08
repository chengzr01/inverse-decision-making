"""General Threshold diffusion on the claim graphs used by visualize.py.

Active means adopting the intervention represented by the selected seed claims.
Justice votes are a configurable aggregation of adoption, not historical votes.
Only support relationships transmit influence by default: treating objections as
negative weights would violate the monotone model in model.md. Other edge types
can be included explicitly as nonnegative channels.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping

InfluenceFunction = Callable[[frozenset[str]], float]


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


class GeneralThresholdModel:
    """Reusable graph with arbitrary monotone local influence functions.

    Accept a case record or its ``information`` graph. Edges are directed from
    sources to targets; multi-endpoint edges expand into all source/target pairs,
    as in visualize.py. Repeated pairs contribute once. The default local rule
    is the fraction of distinct incoming neighbors that are active (zero for
    isolated nodes). Supply ``influence_functions={node_id: callable}`` to use
    nonlinear rules; each callable receives a frozenset of active incoming IDs.
    Callers must ensure these functions are monotone; exhaustive verification
    would require exponentially many evaluations. Runtime values are validated.
    """

    def __init__(self, data: Mapping, *,
                 influence_functions: Mapping[str, InfluenceFunction] | None = None,
                 edge_types: Iterable[str] = ("support",)):
        graph = data.get("information", data)
        records = graph["nodes"]
        self.nodes = tuple(node["id"] for node in records)
        if any(not isinstance(node, str) for node in self.nodes):
            raise ValueError("Node IDs must be strings")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Node IDs must be unique")
        self.justices: dict[str, tuple[str, ...]] = {}
        groups: dict[str, list[str]] = {}
        for node in records:
            justice = node["justice"]
            if not isinstance(justice, str):
                raise ValueError("Justice names must be strings")
            groups.setdefault(justice, []).append(node["id"])
        self.justices = {name: tuple(ids) for name, ids in groups.items()}
        incoming = {node: set() for node in self.nodes}
        allowed = set(edge_types)
        for edge in graph.get("edges", []):
            for field in ("sources", "targets"):
                if not isinstance(edge[field], list) or not edge[field]:
                    raise ValueError(f"Edge {field} must be a nonempty list")
                self.validate_nodes(edge[field])
            if edge["type"] in allowed:
                for target in edge["targets"]:
                    incoming[target].update(edge["sources"])
        self.neighbors = {node: frozenset(ids) for node, ids in incoming.items()}
        self.influence_functions = dict(influence_functions or {})
        self.validate_nodes(self.influence_functions)
        if any(not callable(fn) for fn in self.influence_functions.values()):
            raise ValueError("Influence functions must be callable")

    def validate_nodes(self, nodes: Iterable[str]) -> frozenset[str]:
        if isinstance(nodes, str):
            raise ValueError("Provide an iterable of node IDs, not a single string")
        result = frozenset(nodes)
        unknown = result.difference(self.nodes)
        if unknown:
            raise ValueError(f"Unknown node IDs: {sorted(unknown)}")
        return result

    def influence(self, node: str, active: frozenset[str]) -> float:
        neighbors = self.neighbors[node]
        adopted = neighbors & active
        if node in self.influence_functions:
            return _unit(self.influence_functions[node](adopted), f"Influence for {node}")
        return len(adopted) / len(neighbors) if neighbors else 0.0

    def sample_thresholds(self, rng: random.Random) -> dict[str, float]:
        """Draw once per node; reuse the mapping across seed sets for comparison."""
        return {node: rng.random() for node in self.nodes}

    def simulate(self, selected_nodes: Iterable[str], *,
                 thresholds: Mapping[str, float] | None = None,
                 random_seed: int | None = None,
                 justice_threshold: float | Mapping[str, float] = 0.5) -> dict:
        """Run synchronous, progressive rounds until no further activation.

        Explicit thresholds must cover every node exactly. At equality a node
        activates, including theta=0 and g(empty)=0, as specified in model.md.
        A justice supports if their active fraction reaches justice_threshold;
        ties support. Return JSON-compatible IDs, rounds, thresholds, and votes.
        """
        active = self.validate_nodes(selected_nodes)
        if thresholds is None:
            theta = self.sample_thresholds(random.Random(random_seed))
        else:
            if set(thresholds) != set(self.nodes):
                raise ValueError("Thresholds must contain exactly all node IDs")
            theta = {node: _unit(thresholds[node], f"Threshold for {node}")
                     for node in self.nodes}
        if isinstance(justice_threshold, Mapping):
            if set(justice_threshold) != set(self.justices):
                raise ValueError("Justice thresholds must contain exactly all justice names")
            cutoffs = {name: _unit(justice_threshold[name], f"Justice threshold for {name}")
                       for name in self.justices}
        else:
            cutoff = _unit(justice_threshold, "Justice threshold")
            cutoffs = dict.fromkeys(self.justices, cutoff)
        rounds = [[node for node in self.nodes if node in active]]
        while True:
            new = [node for node in self.nodes
                   if node not in active and self.influence(node, active) >= theta[node]]
            if not new:
                break
            rounds.append(new)
            active = active.union(new)
        votes = {}
        for justice, nodes in self.justices.items():
            adopted = [node for node in nodes if node in active]
            fraction = len(adopted) / len(nodes)
            votes[justice] = {"decision": "support" if fraction >= cutoffs[justice] else "object",
                              "active_nodes": adopted, "active_count": len(adopted),
                              "total_count": len(nodes), "active_fraction": fraction}
        return {"selected_nodes": rounds[0],
                "active_nodes": [node for node in self.nodes if node in active],
                "spread": len(active), "rounds": rounds, "thresholds": theta,
                "decisions": {name: vote["decision"] for name, vote in votes.items()},
                "justices": votes}


def simulate_decisions(data: Mapping, selected_nodes: Iterable[str], *,
                       influence_functions: Mapping[str, InfluenceFunction] | None = None,
                       edge_types: Iterable[str] = ("support",), **kwargs) -> dict:
    """Convenience wrapper; repeated analyses should reuse GeneralThresholdModel."""
    return GeneralThresholdModel(data, influence_functions=influence_functions,
                                 edge_types=edge_types).simulate(selected_nodes, **kwargs)
