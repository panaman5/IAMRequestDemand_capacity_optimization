"""Deterministic synthetic data for the 120-period capacity experiment.

This file is deliberately separate from the modeling code. It creates a
realistic-looking IAM request history with variable daily volume, an
approximately 11% rejection rate, and heterogeneous request workloads.
"""

from __future__ import annotations

import math
import random

from iam_workload_capacity_model import BackendTask, RequestObservation


DEFAULT_PERIODS = 120
DEFAULT_MEAN_REQUESTS = 200
DEFAULT_REJECTION_RATE = 0.11
DEFAULT_SEED = 20260920


def _sample_poisson(rate: float, rng: random.Random) -> int:
    """Sample a Poisson count without adding a dependency to the example."""

    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _sample_overdispersed_count(
    mean: float,
    dispersion: float,
    rng: random.Random,
) -> int:
    """Generate a Negative-Binomial-like count through a Gamma-Poisson mix."""

    shape = 1.0 / dispersion
    scale = dispersion * mean
    poisson_rate = rng.gammavariate(shape, scale)
    return _sample_poisson(poisson_rate, rng)


def experiment_requests(
    *,
    periods: int = DEFAULT_PERIODS,
    mean_requests: int = DEFAULT_MEAN_REQUESTS,
    rejection_rate: float = DEFAULT_REJECTION_RATE,
    seed: int = DEFAULT_SEED,
) -> tuple[RequestObservation, ...]:
    """Return chronological request observations for the portfolio experiment.

    The daily mean stays close to ``mean_requests`` but changes modestly over
    time. Counts are overdispersed, so the number of requests is not exactly
    constant in every period. Workload values vary independently across
    requests and include a small high-cost tail.
    """

    if periods < 1 or mean_requests < 1:
        raise ValueError("periods and mean_requests must be positive")
    if not 0.0 < rejection_rate < 1.0:
        raise ValueError("rejection_rate must be between zero and one")

    rng = random.Random(seed)
    observations: list[RequestObservation] = []

    for period_index in range(periods):
        # A mild cyclical demand pattern makes the historical count series
        # visibly non-flat without introducing a separate seasonal model.
        cycle = 1.0 + 0.08 * math.sin(2.0 * math.pi * period_index / 30.0)
        daily_mean = mean_requests * cycle
        request_count = _sample_overdispersed_count(daily_mean, 0.12, rng)

        for request_index in range(request_count):
            approved = rng.random() >= rejection_rate
            workflow_seconds = rng.uniform(1.5, 3.5)
            workflow_units = rng.uniform(0.45, 0.75)
            pre_workload = rng.uniform(0.8, 1.6)

            tasks: tuple[BackendTask, ...]
            if approved:
                task_list = [
                    BackendTask(
                        task_id="provision",
                        active_seconds=rng.uniform(6.0, 13.0),
                        capacity_units=rng.uniform(0.9, 1.15),
                    ),
                    BackendTask(
                        task_id="audit",
                        active_seconds=rng.uniform(2.0, 4.0),
                        capacity_units=rng.uniform(0.35, 0.65),
                    ),
                ]
                # A small tail of expensive requests makes protection
                # quantiles visibly different from the mean.
                if rng.random() < 0.08:
                    task_list.append(
                        BackendTask(
                            task_id="extended_backend_step",
                            active_seconds=rng.uniform(12.0, 28.0),
                            capacity_units=rng.uniform(0.8, 1.3),
                        )
                    )
                tasks = tuple(task_list)
            else:
                # Rejected requests still retain pre-work and approval
                # workflow execution, but no post-approval backend tasks.
                tasks = ()

            observations.append(
                RequestObservation(
                    request_id=f"experiment-{period_index:03d}-{request_index:04d}",
                    period=f"day-{period_index + 1:03d}",
                    outcome="approved" if approved else "rejected",
                    pre_workload=pre_workload,
                    tasks=tasks,
                    workflow_overhead_seconds=workflow_seconds,
                    workflow_capacity_units=workflow_units,
                )
            )

    return tuple(observations)
