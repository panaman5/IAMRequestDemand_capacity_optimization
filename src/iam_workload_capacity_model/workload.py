"""Request weights and distribution-free workload calibration.

This module turns request-level observations into the continuous workload
signal used by the forecast. It uses a recent-data-weighted empirical
distribution instead of forcing positive, potentially skewed request weights
into a Normal distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable, Sequence

from .request_contract import RequestObservation


@dataclass(frozen=True)
class WeightedEmpiricalDistribution:
    """A positive workload distribution represented by weighted observations.

    The values are individual observed request workloads. The weights are
    statistical recency weights, not capacity units. Keeping the observations
    themselves lets the compound forecast preserve skew and high-cost tails.
    """

    values: tuple[float, ...]
    weights: tuple[float, ...]
    _cumulative_weights: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.values) != len(self.weights) or not self.values:
            raise ValueError("values and weights must be non-empty and aligned")
        if any(value < 0 for value in self.values):
            raise ValueError("workload values must be non-negative")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("distribution weights must be positive")
        cumulative: list[float] = []
        running_total = 0.0
        for weight in self.weights:
            running_total += weight
            cumulative.append(running_total)
        object.__setattr__(self, "_cumulative_weights", tuple(cumulative))

    @property
    def mean(self) -> float:
        """Return the recency-weighted mean workload per request."""

        total_weight = sum(self.weights)
        return sum(value * weight for value, weight in zip(self.values, self.weights)) / total_weight

    def quantile(self, probability: float) -> float:
        """Return a weighted empirical quantile of request workload."""

        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        # Sorting is needed because the observations arrive in time order,
        # while a quantile is defined over the workload value axis.
        pairs = sorted(zip(self.values, self.weights))
        target = probability * sum(self.weights)
        cumulative = 0.0
        for value, weight in pairs:
            cumulative += weight
            if cumulative >= target:
                return value
        return pairs[-1][0]

    def sample(self, rng: random.Random) -> float:
        """Draw one request workload using the empirical probabilities."""

        # ``random.choices`` can accept precomputed cumulative weights. This
        # avoids rebuilding the same cumulative sum for every simulated
        # request, which matters when the history contains thousands of rows.
        return rng.choices(self.values, cum_weights=self._cumulative_weights, k=1)[0]


def request_workload(request: RequestObservation) -> float:
    """Apply the request workload contract to one observed request."""

    return request.workload


def aggregate_workload(
    requests: Iterable[RequestObservation],
) -> dict[str, tuple[int, float]]:
    """Return ``period -> (request_count, total_workload)``.

    The result is the historical time series used to inspect the relationship
    between request volume and capacity workload before fitting forecasts.
    """

    aggregates: dict[str, tuple[int, float]] = {}
    for request in requests:
        # A request is assigned to its active workload bucket by the adapter.
        # A later version can split task intervals across buckets when needed.
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
    controls how quickly older completed observations lose influence. The
    function does not fit a parametric shape; it estimates a distribution by
    retaining the observed workload values and changing their probabilities.
    """

    if not 0.0 < forgetting_factor <= 1.0:
        raise ValueError("forgetting_factor must be in (0, 1]")

    by_outcome: dict[str, list[tuple[float, float]]] = {
        "approved": [],
        "rejected": [],
    }
    for age, request in enumerate(reversed(requests)):
        if request.outcome in by_outcome:
            # ``age == 0`` is the newest observation, so it receives weight 1.
            # Older observations receive progressively less influence.
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
