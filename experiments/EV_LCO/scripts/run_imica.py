import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.optimization import EVLCOProblem, IMICA


def main():
    optimizer = IMICA(
        EVLCOProblem(),
        population_size=8,
        islands=5,
        migration_interval=5,
        migrants=1,
        seed=42,
    )
    result = optimizer.run(max_evaluations=128)

    print(f"Evaluations: {result['evaluations']}")
    print(f"Archive size: {len(result['archive'])}")
    for island, (name, value) in enumerate(
        zip(result["indicator_names"], result["indicator_values"])
    ):
        print(f"Island {island}: {name}={value:.6g}")
    if not any(individual.feasible for individual in result["archive"]):
        print(
            "No feasible design was found. The current battery evaluator is "
            "provisional; do not interpret these values as research results."
        )
    for index, individual in enumerate(result["archive"], start=1):
        print(
            f"{index}: objectives={individual.objectives.tolist()}, "
            f"feasible={individual.feasible}, "
            f"design={individual.design}"
        )


if __name__ == "__main__":
    main()
