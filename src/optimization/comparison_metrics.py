from __future__ import annotations

import numpy as np

from .imica import Individual
from .nsga3 import _associate, _reference_directions


def _nondominated_objectives(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 2:
        return points

    dominated = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        no_worse = np.all(points <= point, axis=1)
        strictly_better = np.any(points < point, axis=1)
        dominated[index] = np.any(no_worse & strictly_better)
    return points[~dominated]


def _hypervolume_recursive(points: np.ndarray, reference: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    points = points[np.all(points < reference, axis=1)]
    if len(points) == 0:
        return 0.0
    points = _nondominated_objectives(points)
    if points.shape[1] == 1:
        return float(max(0.0, reference[0] - np.min(points[:, 0])))

    levels = np.unique(points[:, -1])
    volume = 0.0
    for index, level in enumerate(levels):
        upper = levels[index + 1] if index + 1 < len(levels) else reference[-1]
        if upper <= level:
            continue
        active = points[points[:, -1] <= level, :-1]
        volume += (upper - level) * _hypervolume_recursive(
            active, reference[:-1]
        )
    return float(volume)


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    """Exact minimization hypervolume for a finite set of points."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if points.ndim != 2 or reference.ndim != 1:
        raise ValueError("Expected a 2D point array and a 1D reference point.")
    if points.shape[1] != len(reference):
        raise ValueError("Point and reference dimensions must match.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(reference)):
        raise ValueError("Hypervolume inputs must be finite.")
    return _hypervolume_recursive(points, reference)


def _igd_plus(approximation: np.ndarray, reference_front: np.ndarray) -> float:
    distances = np.maximum(
        approximation[None, :, :] - reference_front[:, None, :], 0.0
    )
    nearest = np.min(np.linalg.norm(distances, axis=2), axis=1)
    return float(np.mean(nearest))


def pooled_feasible_reference(
    archives: list[list[Individual]],
) -> np.ndarray | None:
    feasible = [
        individual.objectives
        for archive in archives
        for individual in archive
        if individual.feasible
    ]
    if not feasible:
        return None
    return _nondominated_objectives(np.vstack(feasible))


def evaluate_front(
    archive: list[Individual],
    final_population: list[Individual],
    reference_front: np.ndarray | None,
) -> dict[str, float | int | None]:
    feasible_archive = [
        individual.objectives for individual in archive if individual.feasible
    ]
    feasible_fraction = (
        float(np.mean([individual.feasible for individual in final_population]))
        if final_population
        else 0.0
    )

    metrics: dict[str, float | int | None] = {
        "archive_size": len(feasible_archive),
        "feasible_fraction": feasible_fraction,
        "hypervolume": None,
        "igd_plus": None,
        "direction_coverage": None,
        "spacing_cv": None,
    }
    if reference_front is None or len(feasible_archive) == 0:
        return metrics

    reference_front = _nondominated_objectives(reference_front)
    objectives = np.vstack(feasible_archive)
    ideal = np.min(reference_front, axis=0)
    span = np.ptp(reference_front, axis=0)
    span[span == 0.0] = 1.0
    normalized_reference = (reference_front - ideal) / span
    normalized_approximation = (objectives - ideal) / span
    normalized_approximation = _nondominated_objectives(normalized_approximation)

    directions = _reference_directions(
        normalized_approximation.shape[1], max(100, len(normalized_approximation))
    )
    associations, _ = _associate(normalized_approximation, directions)
    occupied = len(np.unique(associations))

    if len(normalized_approximation) > 1:
        pairwise = np.linalg.norm(
            normalized_approximation[:, None, :]
            - normalized_approximation[None, :, :],
            axis=2,
        )
        np.fill_diagonal(pairwise, np.inf)
        nearest = np.min(pairwise, axis=1)
        mean_distance = float(np.mean(nearest))
        spacing_cv = (
            float(np.std(nearest) / mean_distance)
            if mean_distance > 0.0
            else 0.0
        )
    else:
        spacing_cv = 0.0

    metrics.update(
        {
            "hypervolume": hypervolume(
                normalized_approximation,
                np.full(normalized_approximation.shape[1], 1.1),
            ),
            "igd_plus": _igd_plus(
                normalized_approximation, normalized_reference
            ),
            "direction_coverage": occupied / len(directions),
            "spacing_cv": spacing_cv,
        }
    )
    return metrics
