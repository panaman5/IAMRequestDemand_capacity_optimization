"""Evaluate the IAM forecast with a separate rolling-origin scenario.

This script deliberately does not change the main fitting experiment. It uses
the existing parameterized flow repeatedly: fit on the observations available
before an evaluation period, forecast that next period, and compare the
forecast with the held-out observations.
"""

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
OUTPUT_DIR = REPO_ROOT / "output" / "evaluation"
CSV_PATH = OUTPUT_DIR / "walk_forward_periods.csv"
SUMMARY_PATH = OUTPUT_DIR / "walk_forward_summary.json"

MIN_TRAINING_PERIODS = 84
SIMULATIONS_PER_ORIGIN = 1_000
EVALUATION_SEED = 20260921


def _quantile(values: list[float] | tuple[float, ...], probability: float) -> float:
    """Use the repository's empirical quantile convention for a local sample."""

    return empirical_quantile(values, probability)


def _period_requests(requests, period: str):
    """Return all requests belonging to one evaluation period."""

    return [request for request in requests if request.period == period]


def main() -> None:
    requests = experiment_requests()
    historical_periods = list(aggregate_workload(requests))
    if len(historical_periods) <= MIN_TRAINING_PERIODS:
        raise ValueError("the dataset must contain periods after the training prefix")

    records: list[dict[str, float | int | str]] = []
    for origin, period in enumerate(
        historical_periods[MIN_TRAINING_PERIODS:],
        start=MIN_TRAINING_PERIODS,
    ):
        train_periods = set(historical_periods[:origin])
        training_requests = [request for request in requests if request.period in train_periods]
        held_out_requests = _period_requests(requests, period)

        config = ForecastConfig(
            total_capacity=3_000.0,
            protection_probability=0.95,
            forgetting_factor=0.995,
            simulations=SIMULATIONS_PER_ORIGIN,
            seed=EVALUATION_SEED + origin,
        )
        fitted = run_forecast(training_requests, config)

        count_forecast = fitted.count_model.forecast()
        count_rng = random.Random(EVALUATION_SEED + 10_000 + origin)
        simulated_counts = [
            fitted.count_model.sample_count(count_forecast, count_rng)
            for _ in range(SIMULATIONS_PER_ORIGIN)
        ]

        simulated_workloads = fitted.simulated_workloads
        rejection_probability = fitted.rejection_model.forecast()
        expected_weight = (
            (1.0 - rejection_probability)
            * fitted.workload_distributions["approved"].mean
            + rejection_probability
            * fitted.workload_distributions["rejected"].mean
        )

        actual_count = len(held_out_requests)
        actual_workload = sum(request.workload for request in held_out_requests)
        actual_average_weight = actual_workload / actual_count if actual_count else 0.0
        records.append(
            {
                "period": period,
                "training_periods": origin,
                "actual_request_count": actual_count,
                "predicted_request_mean": count_forecast.mean,
                "predicted_request_p50": _quantile(simulated_counts, 0.50),
                "predicted_request_p95": _quantile(simulated_counts, 0.95),
                "actual_workload": actual_workload,
                "predicted_workload_p50": _quantile(simulated_workloads, 0.50),
                "predicted_workload_p95": _quantile(simulated_workloads, 0.95),
                "actual_average_weight": actual_average_weight,
                "predicted_average_weight": expected_weight,
                "predicted_rejection_rate": rejection_probability,
                "actual_rejection_rate": sum(
                    request.outcome == "rejected" for request in held_out_requests
                )
                / actual_count,
            }
        )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    count_errors = [
        abs(row["predicted_request_mean"] - row["actual_request_count"])
        for row in records
    ]
    workload_errors = [
        abs(row["predicted_workload_p50"] - row["actual_workload"])
        for row in records
    ]
    weight_errors = [
        abs(row["predicted_average_weight"] - row["actual_average_weight"])
        for row in records
    ]
    summary = {
        "scenario": {
            "training_periods": MIN_TRAINING_PERIODS,
            "evaluation_periods": len(records),
            "simulations_per_origin": SIMULATIONS_PER_ORIGIN,
            "protection_probability": 0.95,
            "total_capacity": 3_000.0,
            "seed": EVALUATION_SEED,
        },
        "metrics": {
            "request_count_mae": mean(count_errors),
            "request_count_p95_coverage": mean(
                [
                    row["actual_request_count"] <= row["predicted_request_p95"]
                    for row in records
                ]
            ),
            "workload_p50_mae": mean(workload_errors),
            "workload_p95_coverage": mean(
                [row["actual_workload"] <= row["predicted_workload_p95"] for row in records]
            ),
            "average_weight_mae": mean(weight_errors),
            "average_predicted_rejection_rate": mean(
                [row["predicted_rejection_rate"] for row in records]
            ),
            "average_actual_rejection_rate": mean(
                [row["actual_rejection_rate"] for row in records]
            ),
        },
        "periods": records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print("IAM walk-forward evaluation")
    print("---------------------------")
    print(f"training periods: {MIN_TRAINING_PERIODS}")
    print(f"evaluation periods: {len(records)}")
    print(f"request-count MAE: {summary['metrics']['request_count_mae']:.2f}")
    print(f"request-count P95 coverage: {summary['metrics']['request_count_p95_coverage']:.2%}")
    print(f"workload P50 MAE: {summary['metrics']['workload_p50_mae']:.2f}")
    print(f"workload P95 coverage: {summary['metrics']['workload_p95_coverage']:.2%}")
    print(f"average-weight MAE: {summary['metrics']['average_weight_mae']:.2f}")
    print(f"period data: {CSV_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
