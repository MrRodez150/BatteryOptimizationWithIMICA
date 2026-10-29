import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.optimization import EVLCOProblem, IMICA, NSGA3
from src.optimization.comparison_metrics import (
    evaluate_front,
    pooled_feasible_reference,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare IMICA-NDR with NSGA-III under a shared EV/LCO problem."
    )
    parser.add_argument("--evaluations", type=int, default=2000)
    parser.add_argument("--population-size", type=int, default=40,
                        help="NSGA-III population and size of each IMICA island.")
    parser.add_argument("--islands", type=int, default=5)
    parser.add_argument("--migration-interval", type=int, default=10)
    parser.add_argument("--migrants", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(ROOT) / "experiments" / "EV_LCO" / "results"
        / "imica_nsga3_comparison.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_population = args.population_size * args.islands
    if args.evaluations < total_population:
        raise ValueError(
            "Evaluation budget must cover IMICA's initial population "
            "across all islands."
        )
    if args.seeds < 1:
        raise ValueError("seeds must be at least 1.")

    runs = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        problem = EVLCOProblem()
        imica = IMICA(
            problem,
            population_size=args.population_size,
            islands=args.islands,
            migration_interval=args.migration_interval,
            migrants=args.migrants,
            seed=seed,
        ).run(args.evaluations)

        nsga3 = NSGA3(
            problem,
            population_size=total_population,
            seed=seed,
        ).run(args.evaluations)
        runs.append((seed, imica, nsga3))
        print(
            f"Seed {seed}: IMICA {imica['evaluations']} evals, "
            f"NSGA-III {nsga3['evaluations']} evals"
        )

    rows = []
    for seed, imica_result, nsga3_result in runs:
        pooled_reference = pooled_feasible_reference(
            [imica_result["archive"], nsga3_result["archive"]]
        )
        imica_metrics = evaluate_front(
            imica_result["archive"],
            [
                individual
                for population in imica_result["populations"]
                for individual in population
            ],
            pooled_reference,
        )
        nsga3_metrics = evaluate_front(
            nsga3_result["archive"],
            nsga3_result["population"],
            pooled_reference,
        )
        for algorithm, result, metrics in (
            ("IMICA-NDR", imica_result, imica_metrics),
            ("NSGA-III", nsga3_result, nsga3_metrics),
        ):
            rows.append(
                {
                    "seed": seed,
                    "algorithm": algorithm,
                    "evaluations": result["evaluations"],
                    "generations": result["generations"],
                    **metrics,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to: {args.output}")
    print(
        "Metrics use the pooled feasible nondominated approximation as the "
        "reference front. This is not a known Pareto front and cannot prove "
        "global optimality."
    )


if __name__ == "__main__":
    main()
