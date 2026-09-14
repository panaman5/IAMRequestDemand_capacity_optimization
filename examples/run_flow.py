"""Parameterized entry point for the IAM workload-capacity flow.

This module intentionally contains no request data. An adapter, application,
or test supplies ``RequestObservation`` objects and configuration parameters.
The actual orchestration lives in ``iam_workload_capacity_model.flow``.
"""

from __future__ import annotations

from typing import Sequence

from iam_workload_capacity_model import (
    ForecastConfig,
    ForecastResult,
    RequestObservation,
    run_forecast,
)


def run_flow(
    requests: Sequence[RequestObservation],
    *,
    total_capacity: float,
    protection_probability: float = 0.95,
    forgetting_factor: float = 0.995,
    simulations: int = 5000,
    seed: int | None = 20260905,
) -> ForecastResult:
    """Run the model with caller-provided data and parameters.

    This function is an application boundary, not a test fixture. A real IAM
    adapter can call it with observations loaded from its own data source.
    """

    config = ForecastConfig(
        total_capacity=total_capacity,
        protection_probability=protection_probability,
        forgetting_factor=forgetting_factor,
        simulations=simulations,
        seed=seed,
    )
    return run_forecast(requests, config)
