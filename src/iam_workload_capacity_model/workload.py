"""Request weights and distribution-free workload calibration."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from .models import RequestObservation


@dataclass(frozen=True)
class WeightedEmpiricalDistribution:
    """A positive workload distribution represented by weighted observations."""

    values: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.weights) or not self.values:
            raise ValueError("values and weights must be non-empty and aligned")
        if any(value < 0 for value in self.values):
            raise ValueError("workload values must be non-negative")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("distribution weights must be positive")

    @property
    def mean(self) -> float:
        """Weighted mean workload."""

        total_weight = sum(self.weights)
        return sum(value * weight for value, weight in zip(self.values, self.weights)) / total_weight

    def quantile(self, probability: float) -> float:
        """Weighted empirical quantile."""

        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        pairs = sorted(zip(self.values, self.weights))
        target = probability * sum(self.weights)
        cumulative = 0.0
        for value, weight in pairs:
            cumulative += weight
            if cumulative >= target:
                return value
        return pairs[-1][0]

    def sample(self, rng: random.Random) -> float:
        """Draw one workload value from the weighted empirical distribution."""

        return rng.choices(self.values, weights=self.weights, k=1)[0]


def request_workload(request: RequestObservation) -> float:
    """Apply the request workload contract to one observation."""

    return request.workload


def aggregate_workload(
    requests: Iterable[RequestObservation],
) -> dict[str, tuple[int, float]]:
    """Return request count and total workload by period."""

    aggregates: dict[str, tuple[int, float]] = {}
    for request in requests:
        count, workload = aggregates.get(request.period, (0, 0.0))
        aggregates[request.period] = (count + 1, workload + request_workload(request))
    return aggregates


def fit_workload_distributions(
    requests: Sequence[RequestObservation],
    *,
    forgetting_factor: float = 0.995,
) -> dict[str, WeightedEmpiricalDistribution]:
    """Fit recent-data-weighted empirical distributions by approval outcome.

    Requests must be supplied in chronological order. The forgetting factor
    controls how quickly older completed observations lose influence.
    """

    if not 0.0 < forgetting_factor <= 1.0:
        raise ValueError("forgetting_factor must be in (0, 1]")

    by_outcome: dict[str, list[tuple[float, float]]] = {
        "approved": [],
        "rejected": [],
    }
    for age, request in enumerate(reversed(requests)):
        if request.outcome in by_outcome:
            by_outcome[request.outcome].append(
                (request_workload(request), forgetting_factor**age)
            )

    distributions: dict[str, WeightedEmpiricalDistribution] = {}
    for outcome, observations in by_outcome.items():
        if not observations:
            raise ValueError(f"no observations for outcome: {outcome}")
        values, weights = zip(*observations)
        distributions[outcome] = WeightedEmpiricalDistribution(
            tuple(values), tuple(weights)
        )
    return distributions
