"""Protected request capacity and spare-capacity calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .forecast import empirical_quantile


@dataclass(frozen=True)
class CapacityPlan:
    """Capacity decision for a future period."""

    expected_workload: float
    protected_request_capacity: float
    total_capacity: float
    spare_capacity: float
    capacity_deficit: float


def plan_capacity(
    simulated_workloads: Sequence[float],
    total_capacity: float,
    *,
    protection_probability: float = 0.95,
) -> CapacityPlan:
    """Reserve request capacity at a chosen workload quantile.

    ``total_capacity`` is the capacity available in the same units and time
    bucket as the simulated workload values.
    """

    if not simulated_workloads:
        raise ValueError("simulated_workloads must be non-empty")
    if total_capacity < 0.0:
        raise ValueError("total_capacity must be non-negative")
    protected = empirical_quantile(simulated_workloads, protection_probability)
    spare = max(0.0, total_capacity - protected)
    deficit = max(0.0, protected - total_capacity)
    return CapacityPlan(
        expected_workload=sum(simulated_workloads) / len(simulated_workloads),
        protected_request_capacity=protected,
        total_capacity=total_capacity,
        spare_capacity=spare,
        capacity_deficit=deficit,
    )
