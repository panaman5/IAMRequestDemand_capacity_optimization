"""Parameterized orchestration for the IAM workload-capacity model.

This module contains the algorithmic flow only. It does not create synthetic
requests, read a vendor API, or print a demo report. A caller supplies
chronological ``RequestObservation`` objects and a ``ForecastConfig``; the
flow connects the model components and returns structured results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .capacity import CapacityPlan, plan_capacity
from .forecast import (
    NegativeBinomialModel,
    RejectionRateModel,
    compound_workload_forecast,
)
from .models import RequestObservation
from .workload import (
    WeightedEmpiricalDistribution,
    aggregate_workload,
    fit_workload_distributions,
)


@dataclass(frozen=True)
class ForecastConfig:
    """Operational parameters supplied by the caller of the modeling flow.

    ``total_capacity`` is the capacity available during one forecast bucket.
    The other fields control statistical calibration, simulation size, and the
    request-priority risk policy. Keeping them outside the algorithm makes the
    flow reusable for different IAM environments and planning periods.
    """

    total_capacity: float
    protection_probability: float = 0.95
    forgetting_factor: float = 0.995
    simulations: int = 5000
    seed: int | None = 20260905

    def __post_init__(self) -> None:
        if self.total_capacity < 0.0:
            raise ValueError("total_capacity must be non-negative")
        if not 0.0 <= self.protection_probability <= 1.0:
            raise ValueError("protection_probability must be between zero and one")
        if not 0.0 < self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if self.simulations < 1:
            raise ValueError("simulations must be positive")


@dataclass(frozen=True)
class ForecastResult:
    """Structured output of one parameterized forecast run.

    The result deliberately exposes intermediate fitted components as well as
    the final capacity plan. This makes the flow inspectable and lets a
    caller attach its own reporting, dashboard, or validation layer.
    """

    count_model: NegativeBinomialModel
    rejection_model: RejectionRateModel
    workload_distributions: dict[str, WeightedEmpiricalDistribution]
    simulated_workloads: tuple[float, ...]
    capacity_plan: CapacityPlan


def run_forecast(
    requests: Sequence[RequestObservation],
    config: ForecastConfig,
) -> ForecastResult:
    """Run the complete forecast flow on caller-supplied IAM observations.

    The input observations must be chronological because the workload
    calibration uses recency weights. The flow performs five algorithmic
    stages:

    1. aggregate historical request counts by period;
    2. fit request-count and rejection-rate models;
    3. fit recent-data-weighted workload distributions by outcome;
    4. simulate future total workload;
    5. reserve the configured workload quantile for requests.

    Data acquisition and synthetic scenarios stay outside this function.
    """

    observations = tuple(requests)
    if not observations:
        raise ValueError("requests must be non-empty")
    if any(request.outcome not in {"approved", "rejected"} for request in observations):
        raise ValueError("request outcomes must be approved or rejected")

    # The historical count series is derived from the request observations;
    # the flow does not assume how those observations were acquired.
    aggregates = aggregate_workload(observations)
    counts = tuple(count for count, _ in aggregates.values())

    decisions = len(observations)
    rejected = sum(request.outcome == "rejected" for request in observations)

    count_model = NegativeBinomialModel.fit(counts)
    rejection_model = RejectionRateModel.fit(decisions, rejected)
    distributions = fit_workload_distributions(
        observations,
        forgetting_factor=config.forgetting_factor,
    )
    simulated = compound_workload_forecast(
        count_model,
        rejection_model,
        distributions,
        simulations=config.simulations,
        seed=config.seed,
    )
    plan = plan_capacity(
        simulated,
        total_capacity=config.total_capacity,
        protection_probability=config.protection_probability,
    )

    return ForecastResult(
        count_model=count_model,
        rejection_model=rejection_model,
        workload_distributions=distributions,
        simulated_workloads=simulated,
        capacity_plan=plan,
    )
