"""Run the larger synthetic capacity experiment and export its results."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from iam_workload_capacity_model import ForecastConfig, run_forecast  # noqa: E402
from iam_workload_capacity_model.forecast import empirical_quantile  # noqa: E402
from iam_workload_capacity_model.workload import aggregate_workload  # noqa: E402
from experiment_data import experiment_requests  # noqa: E402


REPO_ROOT = Path(__file__).parents[1]
DATA_PATH = REPO_ROOT / "data" / "iam_capacity_experiment_120_periods.csv"
RESULT_PATH = REPO_ROOT / "output" / "experiment" / "iam_capacity_experiment_summary.json"


def _histogram(values: list[float], bins: int = 32) -> list[dict[str, float | int]]:
    """Compress simulation values into bins for a readable visualization."""

    low, high = min(values), max(values)
    if low == high:
        return [{"x0": low, "x1": high + 1.0, "count": len(values)}]
    width = (high - low) / bins
    result = []
    for index in range(bins):
        x0 = low + index * width
        x1 = high if index == bins - 1 else low + (index + 1) * width
        count = sum(
            x0 <= value < x1 or (index == bins - 1 and value == x1)
            for value in values
        )
        result.append({"x0": x0, "x1": x1, "count": count})
    return result


def _write_observations_csv(requests) -> None:
    """Persist the generated observations separately from the model code."""

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "request_id",
                "period",
                "outcome",
                "pre_workload",
                "workflow_overhead_seconds",
                "workflow_capacity_units",
                "task_count",
                "workload",
            ]
        )
        for request in requests:
            writer.writerow(
                [
                    request.request_id,
                    request.period,
                    request.outcome,
                    request.pre_workload,
                    request.workflow_overhead_seconds,
                    request.workflow_capacity_units,
                    len(request.tasks),
                    request.workload,
                ]
            )


def main() -> None:
    requests = experiment_requests()
    config = ForecastConfig(
        total_capacity=3_000.0,
        protection_probability=0.95,
        forgetting_factor=0.995,
        simulations=10_000,
        seed=20260920,
    )
    result = run_forecast(requests, config)
    aggregates = aggregate_workload(requests)
    simulated = result.simulated_workloads
    rejection_rate = sum(request.outcome == "rejected" for request in requests) / len(requests)
    count_forecast = result.count_model.forecast()
    count_rng = random.Random((config.seed or 0) + 1)
    simulated_counts = [
        result.count_model.sample_count(count_forecast, count_rng)
        for _ in range(config.simulations)
    ]

    periods = [
        {
            "period": period,
            "request_count": count,
            "total_workload": workload,
        }
        for period, (count, workload) in aggregates.items()
    ]
    summary = {
        "configuration": {
            "periods": 120,
            "mean_requests": 200,
            "observed_requests": len(requests),
            "observed_rejection_rate": rejection_rate,
            "simulations": config.simulations,
            "protection_probability": config.protection_probability,
            "total_capacity": config.total_capacity,
            "seed": config.seed,
        },
        "request_workload": {
            "approved_mean": result.workload_distributions["approved"].mean,
            "rejected_mean": result.workload_distributions["rejected"].mean,
            "approved_p75": result.workload_distributions["approved"].quantile(0.75),
            "approved_p95": result.workload_distributions["approved"].quantile(0.95),
            "rejected_p95": result.workload_distributions["rejected"].quantile(0.95),
        },
        "request_count_forecast": {
            "mean": count_forecast.mean,
            "dispersion": count_forecast.dispersion,
            "p50": empirical_quantile(simulated_counts, 0.50),
            "p75": empirical_quantile(simulated_counts, 0.75),
            "p90": empirical_quantile(simulated_counts, 0.90),
            "p95": empirical_quantile(simulated_counts, 0.95),
            "p99": empirical_quantile(simulated_counts, 0.99),
            "histogram": _histogram(simulated_counts),
        },
        "rejection_forecast": {
            "observed_rate": rejection_rate,
            "forecast_rate": result.rejection_model.forecast(),
        },
        "total_workload_forecast": {
            "p50": empirical_quantile(simulated, 0.50),
            "p75": empirical_quantile(simulated, 0.75),
            "p90": empirical_quantile(simulated, 0.90),
            "p95": empirical_quantile(simulated, 0.95),
            "p99": empirical_quantile(simulated, 0.99),
        },
        "capacity_plan": {
            "expected_workload": result.capacity_plan.expected_workload,
            "protected_request_capacity": result.capacity_plan.protected_request_capacity,
            "spare_capacity": result.capacity_plan.spare_capacity,
            "capacity_deficit": result.capacity_plan.capacity_deficit,
        },
        "periods": periods,
        "simulated_workloads": list(simulated),
    }

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_observations_csv(requests)
    RESULT_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print("IAM capacity experiment")
    print("----------------------")
    print(f"observed requests: {len(requests):,}")
    print(f"observed periods: {len(periods)}")
    print(f"observed rejection rate: {rejection_rate:.2%}")
    print(f"forecast request count mean: {count_forecast.mean:.2f}")
    print(f"forecast request count P95: {summary['request_count_forecast']['p95']:.0f}")
    print(f"approved mean workload/request: {summary['request_workload']['approved_mean']:.2f}")
    print(f"rejected mean workload/request: {summary['request_workload']['rejected_mean']:.2f}")
    print(f"simulations: {config.simulations:,}")
    print(f"P95 protected request capacity: {result.capacity_plan.protected_request_capacity:.2f}")
    print(f"spare capacity: {result.capacity_plan.spare_capacity:.2f}")
    print(f"capacity deficit: {result.capacity_plan.capacity_deficit:.2f}")
    print(f"data: {DATA_PATH}")
    print(f"summary: {RESULT_PATH}")


if __name__ == "__main__":
    main()
