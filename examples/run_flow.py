"""Minimal end-to-end modeling flow; intentionally not a test suite.

The example is deliberately small enough to read from top to bottom. It shows
the complete bridge from request observations to a protected request-capacity
decision without connecting to a real IAM product yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from iam_workload_capacity_model import (  # noqa: E402
    BackendTask,
    NegativeBinomialModel,
    RejectionRateModel,
    RequestObservation,
    compound_workload_forecast,
    fit_workload_distributions,
    plan_capacity,
)
from iam_workload_capacity_model.forecast import empirical_quantile  # noqa: E402


def historical_requests() -> tuple[RequestObservation, ...]:
    """Return synthetic history showing the model contract.

    The repeated pattern is only demonstration data. It is not intended to
    represent a real IAM distribution or to provide model quality evidence.
    """

    requests: list[RequestObservation] = []
    for index in range(180):
        # Every fifth request is rejected, which creates a visible approval
        # mixture while keeping the example deterministic.
        approved = index % 5 != 0
        tasks = (
            BackendTask("provision", 8.0 + (index % 4), 1.0),
            BackendTask("audit", 3.0, 0.5),
        )
        requests.append(
            RequestObservation(
                request_id=f"req-{index}",
                period=f"day-{index // 12}",
                outcome="approved" if approved else "rejected",
                pre_workload=1.0,
                tasks=tasks,
                workflow_overhead_seconds=2.0,
                workflow_capacity_units=0.5,
            )
        )
    return tuple(requests)


def main() -> None:
    # 1. Collect the historical signals needed by the three model components.
    history = historical_requests()
    counts_by_period: dict[str, int] = {}
    rejected = 0
    decisions = 0
    for request in history:
        counts_by_period[request.period] = counts_by_period.get(request.period, 0) + 1
        if request.outcome in {"approved", "rejected"}:
            decisions += 1
            rejected += request.outcome == "rejected"

    # 2. Fit the request-count and rejection models from historical aggregates.
    count_model = NegativeBinomialModel.fit(tuple(counts_by_period.values()))
    rejection_model = RejectionRateModel.fit(decisions, rejected)

    # 3. Calibrate recent-data-weighted workload distributions by outcome.
    distributions = fit_workload_distributions(history, forgetting_factor=0.995)

    # 4. Combine count, approval outcome, and request weights through Monte
    # Carlo simulation to obtain a predictive total-workload distribution.
    simulated = compound_workload_forecast(
        count_model,
        rejection_model,
        distributions,
        simulations=2000,
    )
    # 5. Protect P95 request workload; only the residual is spare capacity.
    plan = plan_capacity(simulated, total_capacity=1000.0)

    print("IAM workload-capacity modeling flow")
    print("-----------------------------------")
    print(f"forecast request mean: {count_model.forecast().mean:.2f}")
    print(f"forecast rejection rate: {rejection_model.forecast():.2%}")
    print(f"approved mean workload/request: {distributions['approved'].mean:.2f}")
    print(f"rejected mean workload/request: {distributions['rejected'].mean:.2f}")
    print(f"expected total workload: {plan.expected_workload:.2f}")
    print(f"P95 protected request capacity: {plan.protected_request_capacity:.2f}")
    print(f"spare capacity: {plan.spare_capacity:.2f}")
    print(f"capacity deficit: {plan.capacity_deficit:.2f}")
    print(f"P50 total workload: {empirical_quantile(simulated, 0.50):.2f}")


if __name__ == "__main__":
    main()
