"""Separate synthetic scenario for exercising the parameterized flow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from iam_workload_capacity_model import ForecastConfig, run_forecast  # noqa: E402
from synthetic_data import historical_requests  # noqa: E402


def main() -> None:
    """Run the fixture scenario and print its outputs for inspection."""

    result = run_forecast(
        historical_requests(),
        ForecastConfig(
            total_capacity=1000.0,
            protection_probability=0.95,
            forgetting_factor=0.995,
            simulations=2000,
            seed=20260905,
        ),
    )
    plan = result.capacity_plan
    print("IAM workload-capacity synthetic scenario")
    print("----------------------------------------")
    print(f"forecast request mean: {result.count_model.forecast().mean:.2f}")
    print(f"forecast rejection rate: {result.rejection_model.forecast():.2%}")
    print(
        "approved mean workload/request: "
        f"{result.workload_distributions['approved'].mean:.2f}"
    )
    print(
        "rejected mean workload/request: "
        f"{result.workload_distributions['rejected'].mean:.2f}"
    )
    print(f"expected total workload: {plan.expected_workload:.2f}")
    print(f"P95 protected request capacity: {plan.protected_request_capacity:.2f}")
    print(f"spare capacity: {plan.spare_capacity:.2f}")
    print(f"capacity deficit: {plan.capacity_deficit:.2f}")


if __name__ == "__main__":
    main()
