"""Compare stationary, seasonal, and shock count-forecast scenarios.

The existing IAM experiment remains unchanged. This file isolates request-count
behavior so we can tell whether an error comes from the count model or from the
later workload-weight transformation.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from iam_workload_capacity_model.forecast import (  # noqa: E402
    CountForecast,
    NegativeBinomialModel,
    empirical_quantile,
)


REPO_ROOT = Path(__file__).parents[1]
OUTPUT_DIR = REPO_ROOT / "output" / "count_scenarios"
SUMMARY_PATH = OUTPUT_DIR / "count_model_scenarios_summary.json"
PERIODS_PATH = OUTPUT_DIR / "count_model_scenarios_periods.csv"

PERIODS = 120
TRAINING_PERIODS = 84
MEAN_COUNT = 200.0
DISPERSION = 0.12
SEASONAL_PERIOD = 30
SCENARIOS = ("stationary", "seasonal", "strong_seasonal", "shock")
EVALUATION_SIMULATIONS = 1_000
SEED = 20260922


def _sample_poisson(rate: float, rng: random.Random) -> int:
    """Sample a Poisson count without external numerical dependencies."""

    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _sample_negative_binomial(
    mean: float,
    dispersion: float,
    rng: random.Random,
) -> int:
    """Generate a Negative Binomial count through a Gamma-Poisson mixture."""

    shape = 1.0 / dispersion
    scale = dispersion * mean
    poisson_rate = rng.gammavariate(shape, scale)
    return _sample_poisson(poisson_rate, rng)


def _generate_counts(
    *,
    scenario: str,
    seed: int,
) -> tuple[int, ...]:
    """Generate one deterministic count series for a named scenario."""

    rng = random.Random(seed)
    counts: list[int] = []
    for period in range(PERIODS):
        mean = MEAN_COUNT
        if scenario in {"seasonal", "shock"}:
            mean *= 1.0 + 0.08 * math.sin(2.0 * math.pi * period / SEASONAL_PERIOD)
        if scenario == "strong_seasonal":
            # A deliberately stronger predictable signal, so the benefit of
            # a seasonal model can be separated from irreducible noise.
            mean *= 1.0 + 0.25 * math.sin(2.0 * math.pi * period / SEASONAL_PERIOD)
        if scenario == "shock" and rng.random() < 0.06:
            # This shock is intentionally not predictable from history.
            mean *= 2.2
        counts.append(_sample_negative_binomial(mean, DISPERSION, rng))
    return tuple(counts)


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float]:
    """Solve a small 3x3 linear system with Gaussian elimination."""

    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for pivot in range(3):
        largest = max(range(pivot, 3), key=lambda row: abs(augmented[row][pivot]))
        augmented[pivot], augmented[largest] = augmented[largest], augmented[pivot]
        divisor = augmented[pivot][pivot]
        if abs(divisor) < 1e-12:
            raise ValueError("seasonal design matrix is singular")
        augmented[pivot] = [value / divisor for value in augmented[pivot]]
        for row in range(3):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            augmented[row] = [
                current - factor * normalized
                for current, normalized in zip(augmented[row], augmented[pivot])
            ]
    return tuple(augmented[row][3] for row in range(3))


@dataclass(frozen=True)
class SeasonalNegativeBinomialModel:
    """A lightweight seasonal NB model for the isolated scenario experiment.

    The log mean is fitted with a sine/cosine seasonal basis. Dispersion is
    estimated from the remaining count variance. This is intentionally kept in
    the experiment script until the comparison justifies promoting it into the
    reusable forecasting module.
    """

    coefficients: tuple[float, float, float]
    dispersion: float
    seasonal_period: int

    @classmethod
    def fit(
        cls,
        counts: tuple[int, ...],
        *,
        seasonal_period: int,
    ) -> "SeasonalNegativeBinomialModel":
        design: list[list[float]] = []
        target: list[float] = []
        for period, count in enumerate(counts):
            phase = 2.0 * math.pi * period / seasonal_period
            design.append([1.0, math.sin(phase), math.cos(phase)])
            target.append(math.log(count + 1.0))

        matrix = [
            [sum(row[column] * row[index] for row in design) for index in range(3)]
            for column in range(3)
        ]
        vector = [
            sum(row[column] * value for row, value in zip(design, target))
            for column in range(3)
        ]
        coefficients = _solve_3x3(matrix, vector)

        # Estimate dispersion from the pooled series, as the baseline does.
        # Estimating it from seasonal residuals alone would make the interval
        # artificially narrow after a noisy short fit.
        baseline = NegativeBinomialModel.fit(counts)
        dispersion = baseline.dispersion
        return cls(coefficients, dispersion, seasonal_period)

    def forecast(self, period: int) -> CountForecast:
        """Return the next-period mean and dispersion."""

        phase = 2.0 * math.pi * period / self.seasonal_period
        features = (1.0, math.sin(phase), math.cos(phase))
        mean = max(
            0.0,
            math.exp(sum(coefficient * feature for coefficient, feature in zip(self.coefficients, features))) - 1.0,
        )
        return CountForecast(mean=mean, dispersion=self.dispersion)


def _sample_forecast(forecast: CountForecast, rng: random.Random) -> int:
    """Sample one count from a fitted CountForecast."""

    if forecast.mean <= 0.0:
        return 0
    if forecast.dispersion <= 0.0:
        return _sample_poisson(forecast.mean, rng)
    shape = 1.0 / forecast.dispersion
    scale = forecast.dispersion * forecast.mean
    return _sample_poisson(rng.gammavariate(shape, scale), rng)


def _evaluate_model(
    counts: tuple[int, ...],
    *,
    model_name: str,
    fit_model,
    forecast_model,
    scenario: str,
) -> list[dict[str, float | int | str]]:
    """Run rolling-origin forecasts for one model/scenario pair."""

    records = []
    for origin in range(TRAINING_PERIODS, len(counts)):
        model = fit_model(counts[:origin])
        forecast = forecast_model(model, origin)
        rng = random.Random(SEED + origin + (1 if model_name == "seasonal_nb" else 0))
        simulated = sorted(
            _sample_forecast(forecast, rng) for _ in range(EVALUATION_SIMULATIONS)
        )
        actual = counts[origin]
        records.append(
            {
                "scenario": scenario,
                "model": model_name,
                "period": origin + 1,
                "actual_count": actual,
                "predicted_mean": forecast.mean,
                "predicted_p50": empirical_quantile(simulated, 0.50),
                "predicted_p95": empirical_quantile(simulated, 0.95),
            }
        )
    return records


def main() -> None:
    all_records: list[dict[str, float | int | str]] = []
    for scenario in SCENARIOS:
        counts = _generate_counts(scenario=scenario, seed=SEED)
        all_records.extend(
            _evaluate_model(
                counts,
                scenario=scenario,
                model_name="stationary_nb",
                fit_model=NegativeBinomialModel.fit,
                forecast_model=lambda model, _period: model.forecast(),
            )
        )
        all_records.extend(
            _evaluate_model(
                counts,
                scenario=scenario,
                model_name="seasonal_nb",
                fit_model=lambda values: SeasonalNegativeBinomialModel.fit(
                    values,
                    seasonal_period=SEASONAL_PERIOD,
                ),
                forecast_model=lambda model, period: model.forecast(period),
            )
        )

    def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
        errors = [abs(row["predicted_mean"] - row["actual_count"]) for row in rows]
        coverage = [row["actual_count"] <= row["predicted_p95"] for row in rows]
        return {
            "evaluation_periods": len(rows),
            "mean_actual_count": sum(row["actual_count"] for row in rows) / len(rows),
            "mean_predicted_count": sum(row["predicted_mean"] for row in rows) / len(rows),
            "mae": sum(errors) / len(errors),
            "p95_coverage": sum(coverage) / len(coverage),
        }

    summary = {
        "configuration": {
            "periods": PERIODS,
            "training_periods": TRAINING_PERIODS,
            "evaluation_periods": PERIODS - TRAINING_PERIODS,
            "mean_count": MEAN_COUNT,
            "generator_dispersion": DISPERSION,
            "seasonal_period": SEASONAL_PERIOD,
            "simulations_per_origin": EVALUATION_SIMULATIONS,
            "seed": SEED,
        },
        "results": {
            f"{scenario}_{model}": summarize(
                [row for row in all_records if row["scenario"] == scenario and row["model"] == model]
            )
            for scenario in SCENARIOS
            for model in ("stationary_nb", "seasonal_nb")
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    with PERIODS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)

    print("Count-model scenario comparison")
    print("-------------------------------")
    for key, metrics in summary["results"].items():
        print(
            f"{key}: MAE={metrics['mae']:.2f}, "
            f"P95 coverage={metrics['p95_coverage']:.2%}"
        )
    print(f"period data: {PERIODS_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
