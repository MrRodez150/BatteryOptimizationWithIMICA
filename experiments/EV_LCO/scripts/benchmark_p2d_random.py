from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.p2d_evaluator import P2DEvaluator
from src.optimization.problem import EV_LCO_VARIABLES


DEFAULT_RESULTS_DIRECTORY = (
    WORKSPACE_ROOT / "experiments" / "EV_LCO" / "results"
)


def _sample_design(rng: np.random.Generator) -> dict[str, float | int]:
    design: dict[str, float | int] = {}
    for variable in EV_LCO_VARIABLES:
        if variable.kind == "integer":
            design[variable.name] = int(
                rng.integers(
                    math.ceil(variable.lower),
                    math.floor(variable.upper) + 1,
                )
            )
        else:
            design[variable.name] = float(
                rng.uniform(variable.lower, variable.upper)
            )
    return design


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, percentile))


def run_benchmark(
    count: int,
    seed: int,
    output_path: Path,
    evaluator: P2DEvaluator,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be a positive integer.")

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    elapsed_times: list[float] = []
    successful_times: list[float] = []
    wall_start = time.perf_counter()

    for individual_index in range(1, count + 1):
        design = _sample_design(rng)
        start = time.perf_counter()
        try:
            evaluation = evaluator.evaluate(design)
        except Exception as error:
            elapsed_s = time.perf_counter() - start
            elapsed_times.append(elapsed_s)
            results.append(
                {
                    "individual": individual_index,
                    "status": "failed",
                    "elapsed_s": elapsed_s,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "design": design,
                    "objectives": {},
                }
            )
            print(
                f"[{individual_index}/{count}] FAILED "
                f"{elapsed_s:.2f} s: {type(error).__name__}: {error}",
                flush=True,
            )
            continue

        elapsed_s = time.perf_counter() - start
        elapsed_times.append(elapsed_s)
        successful_times.append(elapsed_s)
        objectives = evaluation["objectives"]
        results.append(
            {
                "individual": individual_index,
                "status": "success",
                "elapsed_s": elapsed_s,
                "error_type": "",
                "error_message": "",
                "design": design,
                "objectives": objectives,
            }
        )
        objective_summary = ", ".join(
            f"{name}={value:.6g}" for name, value in objectives.items()
        )
        print(
            f"[{individual_index}/{count}] OK {elapsed_s:.2f} s: "
            f"{objective_summary}",
            flush=True,
        )

    total_wall_s = time.perf_counter() - wall_start
    successful_count = len(successful_times)
    failed_count = count - successful_count
    summary = {
        "requested_individuals": count,
        "successful_individuals": successful_count,
        "failed_individuals": failed_count,
        "failure_percentage": 100.0 * failed_count / count,
        "success_percentage": 100.0 * successful_count / count,
        "seed": seed,
        "mesh": {
            "positive_cells": evaluator.positive_cells,
            "separator_cells": evaluator.separator_cells,
            "negative_cells": evaluator.negative_cells,
            "radial_cells": evaluator.radial_cells,
        },
        "time_step_s": evaluator.time_step_s,
        "aging_cycles": evaluator.aging_cycles,
        "total_wall_time_s": total_wall_s,
        "evaluation_time_s": {
            "mean": mean(elapsed_times),
            "median": median(elapsed_times),
            "population_stddev": pstdev(elapsed_times),
            "p95": _percentile(elapsed_times, 95.0),
            "mean_successful": (
                mean(successful_times) if successful_times else None
            ),
            "mean_failed": (
                mean(
                    [
                        float(result["elapsed_s"])
                        for result in results
                        if result["status"] == "failed"
                    ]
                )
                if failed_count
                else None
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "individual",
        "status",
        "elapsed_s",
        "error_type",
        "error_message",
        *[variable.name for variable in EV_LCO_VARIABLES],
        "specific_energy",
        "ohmic_energy_loss_j_m2",
        "capacity_fade_fraction",
        "price",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {
                "individual": result["individual"],
                "status": result["status"],
                "elapsed_s": f"{result['elapsed_s']:.6f}",
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                **result["design"],
                **result["objectives"],
            }
            writer.writerow(row)

    summary_path = output_path.with_suffix(".summary.json")
    summary["results_csv"] = str(output_path)
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate random EV/LCO designs with P2D and report runtime and "
            "failure percentage."
        )
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-step-s", type=float, default=10.0)
    parser.add_argument("--aging-cycles", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=None)
    parser.add_argument("--positive-cells", type=int, default=8)
    parser.add_argument("--separator-cells", type=int, default=3)
    parser.add_argument("--negative-cells", type=int, default=8)
    parser.add_argument("--radial-cells", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to a timestamped file in experiments/EV_LCO/results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (
        DEFAULT_RESULTS_DIRECTORY / f"p2d_random_benchmark_{timestamp}.csv"
    )
    evaluator = P2DEvaluator(
        positive_cells=args.positive_cells,
        separator_cells=args.separator_cells,
        negative_cells=args.negative_cells,
        radial_cells=args.radial_cells,
        time_step_s=args.time_step_s,
        maximum_steps=args.maximum_steps,
        aging_cycles=args.aging_cycles,
    )
    summary = run_benchmark(
        count=args.count,
        seed=args.seed,
        output_path=output_path,
        evaluator=evaluator,
    )
    print("\nBenchmark summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
