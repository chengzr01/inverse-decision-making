"""Heuristic cardinality-budget influence maximization for model.md.

The objective is expected total active claim nodes, including seeds, rather than
justice votes. Greedy has the 1-1/e guarantee only for submodular local rules and
exact spread evaluations; finite Monte Carlo estimates here carry no certified
approximation guarantee. Random search can explore synergistic seed combinations
under unrestricted monotone rules. Neither routine promises an optimal solution.

Example (also callable from visualize.py)::

    model = GeneralThresholdModel(case)
    solution = greedy_selection(model, budget=3, simulations=200, random_seed=7)
    votes = model.simulate(solution['selected_nodes'], random_seed=7)
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable
from itertools import combinations

try:  # Support both direct sibling imports and package imports.
    from .decisions import GeneralThresholdModel
except ImportError:
    from decisions import GeneralThresholdModel


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if allow_zero else 1}")


def _scenarios(model: GeneralThresholdModel, simulations: int, rng: random.Random) -> list[dict]:
    _positive_integer(simulations, "simulations")
    return [model.sample_thresholds(rng) for _ in range(simulations)]


def _estimate(model: GeneralThresholdModel, seeds: Iterable[str], scenarios: list[dict]) -> dict:
    seeds = model.validate_nodes(seeds)
    counts = [model.simulate(seeds, thresholds=theta)["spread"] for theta in scenarios]
    mean = sum(counts) / len(counts)
    error = (math.sqrt(sum((count - mean) ** 2 for count in counts)
                       / (len(counts) - 1) / len(counts)) if len(counts) > 1 else None)
    return {"expected_spread": mean, "standard_error": error, "simulations": len(counts)}


def estimate_spread(model: GeneralThresholdModel, selected_nodes: Iterable[str], *,
                    simulations: int = 1000, random_seed: int | None = None) -> dict:
    """Monte Carlo mean and standard error (None for a single simulation)."""
    return _estimate(model, selected_nodes,
                     _scenarios(model, simulations, random.Random(random_seed)))


def _candidates(model: GeneralThresholdModel, budget: int,
                candidates: Iterable[str] | None) -> list[str]:
    _positive_integer(budget, "budget", allow_zero=True)
    allowed = model.validate_nodes(model.nodes if candidates is None else candidates)
    ordered = [node for node in model.nodes if node in allowed]
    if budget > len(ordered):
        raise ValueError("Budget exceeds the number of candidate nodes")
    return ordered


def _result(model, selected, scenarios, rng, simulations, method, evaluations, **extra):
    training = _estimate(model, selected, scenarios)
    # Fresh draws avoid reporting an estimate biased upward by selecting on it.
    validation = _estimate(model, selected, _scenarios(model, simulations, rng))
    return {"method": method, "selected_nodes": selected, "budget": len(selected),
            **validation, "selection_expected_spread": training["expected_spread"],
            "evaluated_sets": evaluations, **extra}


def greedy_selection(model: GeneralThresholdModel, budget: int, *,
                     candidates: Iterable[str] | None = None,
                     simulations: int = 1000, random_seed: int | None = None) -> dict:
    """Add the node with largest estimated marginal spread until |S|=budget.

    All comparisons share threshold draws, reducing comparison noise and making
    the sampled objective monotone. Ties follow input node order. ``steps`` gives
    each selected node's marginal gain on these training scenarios. Final spread
    and standard error use independent validation draws.
    """
    remaining = _candidates(model, budget, candidates)
    rng = random.Random(random_seed)
    scenarios = _scenarios(model, simulations, rng)
    selected, steps = [], []
    baseline = _estimate(model, selected, scenarios)["expected_spread"]
    evaluations = 1
    for _ in range(budget):
        scores = [(node, _estimate(model, selected + [node], scenarios)["expected_spread"])
                  for node in remaining]
        evaluations += len(scores)
        node, score = max(scores, key=lambda item: item[1])
        selected.append(node)
        remaining.remove(node)
        steps.append({"node": node, "marginal_gain": score - baseline,
                      "expected_spread": score})
        baseline = score
    return _result(model, selected, scenarios, rng, simulations, "greedy",
                   evaluations, steps=steps)


def random_search(model: GeneralThresholdModel, budget: int, *,
                  candidates: Iterable[str] | None = None, trials: int = 100,
                  simulations: int = 1000, random_seed: int | None = None) -> dict:
    """Compare up to trials distinct budget-sized sets, allowing joint effects.

    If trials covers all combinations, enumerate them; otherwise draw distinct
    random subsets. Even exhaustive enumeration optimizes only a sampled spread
    estimate, so it is not a guarantee of the true expected-spread optimum.
    """
    pool = _candidates(model, budget, candidates)
    _positive_integer(trials, "trials")
    rng = random.Random(random_seed)
    scenarios = _scenarios(model, simulations, rng)
    total = math.comb(len(pool), budget)
    limit = min(trials, total)
    if limit == total:
        proposals = combinations(pool, budget)
    else:
        seen = set()
        sampled = []
        while len(sampled) < limit:
            indices = tuple(sorted(rng.sample(range(len(pool)), budget)))
            if indices not in seen:
                seen.add(indices)
                sampled.append(tuple(pool[i] for i in indices))
        proposals = iter(sampled)
    best, best_score = [], -math.inf
    for proposal in proposals:
        score = _estimate(model, proposal, scenarios)["expected_spread"]
        if score > best_score:
            best, best_score = list(proposal), score
    return _result(model, best, scenarios, rng, simulations, "random_search", limit)
