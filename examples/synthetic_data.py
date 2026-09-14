"""Synthetic observations used only by the separate demo/test scenario."""

from __future__ import annotations

from iam_workload_capacity_model import BackendTask, RequestObservation


def historical_requests() -> tuple[RequestObservation, ...]:
    """Return deterministic fixture data for exercising the modeling flow.

    This is deliberately outside the flow implementation. It represents a
    test scenario, not a claim about a real IAM system or its distribution.
    """

    requests: list[RequestObservation] = []
    for index in range(180):
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
