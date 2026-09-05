"""Transparent forecast components for request demand and workload."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from .workload import WeightedEmpiricalDistribution


@dataclass(frozen=True)
class CountForecast:
    """Predictive parameters for request arrivals in one future bucket."""

    mean: float
    dispersion: float


@dataclass
class NegativeBinomialModel:
    """Moment-fitted Negative Binomial baseline with an online demand state."""

    mean: float
    dispersion: float
    state: float = 0.0
    omega: float = 0.0
    persistence: float = 0.90
    learning_rate: float = 0.10

    @classmethod
    def fit(cls, counts: Sequence[int]) -> "NegativeBinomialModel":
        """Fit mean and dispersion by moments as a dependency-free baseline."""

        if not counts or any(count < 0 for count in counts):
            raise ValueError("counts must be non-empty and non-negative")
        mean = sum(counts) / len(counts)
        variance = sum((count - mean) ** 2 for count in counts) / len(counts)
        dispersion = max(0.0, (variance - mean) / (mean**2)) if mean else 0.0
        return cls(mean=mean, dispersion=dispersion, state=math.log(mean or 1.0))

    def forecast(self, horizon: int = 1) -> CountForecast:
        """Return the current predictive mean and dispersion."""

        if horizon < 1:
            raise ValueError("horizon must be positive")
        return CountForecast(
            mean=max(0.0, math.exp(self.state)),
            dispersion=self.dispersion,
        )

    def update(self, observed_count: int) -> CountForecast:
        """Apply a score-driven online correction after a bucket closes."""

        forecast = self.forecast()
        score = (observed_count - forecast.mean) / (
            1.0 + forecast.dispersion * forecast.mean
        )
        self.state = (
            self.omega
            + self.persistence * self.state
            + self.learning_rate * score
        )
        return self.forecast()

    def sample_count(self, forecast: CountForecast, rng: random.Random) -> int:
        """Sample a count through the Gamma-Poisson mixture representation."""

        if forecast.mean <= 0.0:
            return 0
        if forecast.dispersion <= 0.0:
            return _sample_poisson(forecast.mean, rng)
        shape = 1.0 / forecast.dispersion
        scale = forecast.dispersion * forecast.mean
        poisson_rate = rng.gammavariate(shape, scale)
        return _sample_poisson(poisson_rate, rng)


@dataclass
class RejectionRateModel:
    """Binomial rejection-rate baseline with an online log-odds state."""

    probability: float
    state: float
    persistence: float = 0.90
    learning_rate: float = 0.05

    @classmethod
    def fit(cls, decisions: int, rejected: int) -> "RejectionRateModel":
        """Fit the initial rejection probability from observed decisions."""

        if decisions <= 0 or not 0 <= rejected <= decisions:
            raise ValueError("rejected must be between zero and decisions")
        probability = min(1.0 - 1e-9, max(1e-9, rejected / decisions))
        state = math.log(probability / (1.0 - probability))
        return cls(probability=probability, state=state)

    def forecast(self) -> float:
        """Return the current rejection probability."""

        return 1.0 / (1.0 + math.exp(-self.state))

    def update(self, decisions: int, rejected: int) -> float:
        """Correct the log-odds state from a newly observed bucket."""

        if decisions <= 0 or not 0 <= rejected <= decisions:
            raise ValueError("rejected must be between zero and decisions")
        probability = self.forecast()
        score = rejected - decisions * probability
        self.state = self.persistence * self.state + self.learning_rate * score
        self.probability = self.forecast()
        return self.probability


def compound_workload_forecast(
    count_model: NegativeBinomialModel,
    rejection_model: RejectionRateModel,
    distributions: dict[str, WeightedEmpiricalDistribution],
    *,
    simulations: int = 5000,
    seed: int = 20260905,
) -> tuple[float, ...]:
    """Simulate total workload from count, outcome, and weight distributions."""

    if simulations < 1:
        raise ValueError("simulations must be positive")
    rng = random.Random(seed)
    forecast = count_model.forecast()
    rejection_probability = rejection_model.forecast()
    totals: list[float] = []
    for _ in range(simulations):
        request_count = count_model.sample_count(forecast, rng)
        total = 0.0
        for _ in range(request_count):
            outcome = "rejected" if rng.random() < rejection_probability else "approved"
            total += distributions[outcome].sample(rng)
        totals.append(total)
    return tuple(sorted(totals))


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    """Return a simple empirical quantile from sorted or unsorted values."""

    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("values must be non-empty and probability must be valid")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(probability * len(ordered)))
    return ordered[index]


def _sample_poisson(rate: float, rng: random.Random) -> int:
    """Sample a Poisson value using Knuth's algorithm for the model prototype."""

    if rate <= 0.0:
        return 0
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1
