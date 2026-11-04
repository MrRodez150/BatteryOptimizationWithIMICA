from __future__ import annotations

from typing import Protocol

import numpy as np


class QualityIndicator(Protocol):
    name: str
    requires_reference: bool

    def __call__(
        self, points: np.ndarray, reference_front: np.ndarray | None = None
    ) -> float: ...


def _nondominated(points: np.ndarray) -> np.ndarray:
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
    points = _nondominated(points)
    if points.shape[1] == 1:
        return float(max(0.0, reference[0] - np.min(points[:, 0])))

    levels = np.unique(points[:, -1])
    volume = 0.0
    for index, level in enumerate(levels):
        upper = levels[index + 1] if index + 1 < len(levels) else reference[-1]
        if upper > level:
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


def _simplex_directions(objective_count: int, divisions: int = 4) -> np.ndarray:
    directions = []

    def generate(prefix: list[int], remaining: int, dimensions: int) -> None:
        if dimensions == 1:
            directions.append(
                np.asarray(prefix + [remaining], dtype=float) / divisions
            )
            return
        for value in range(remaining + 1):
            generate(prefix + [value], remaining - value, dimensions - 1)

    generate([], divisions, objective_count)
    return np.asarray(directions, dtype=float)


class SMS:
    """S-metric selection: maximize dominated hypervolume."""

    name = "HV"
    requires_reference = False

    def __call__(
        self, points: np.ndarray, reference_front: np.ndarray | None = None
    ) -> float:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("Indicator points must be a 2D array.")
        return hypervolume(points, np.full(points.shape[1], 1.1))


class R2:
    """Minimization R2 indicator using weighted Chebyshev scalarizations."""

    name = "R2"
    requires_reference = False

    def __init__(self, weights: np.ndarray | None = None) -> None:
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and (
            self.weights.ndim != 2
            or not np.all(np.isfinite(self.weights))
            or np.any(self.weights < 0.0)
            or np.any(np.sum(self.weights, axis=1) <= 0.0)
        ):
            raise ValueError("R2 weights must be a finite nonnegative 2D array.")

    def __call__(
        self, points: np.ndarray, reference_front: np.ndarray | None = None
    ) -> float:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("Indicator points must be a 2D array.")
        if len(points) == 0:
            return float("inf")
        weights = (
            _simplex_directions(points.shape[1])
            if self.weights is None
            else self.weights
        )
        if weights.shape[1] != points.shape[1]:
            raise ValueError("R2 weight dimension must match objective dimension.")
        weights = np.maximum(weights, 1e-6)
        scalarized = np.max(
            np.abs(points[:, None, :]) * weights[None, :, :], axis=2
        )
        return float(np.mean(np.min(scalarized, axis=0)))


class IGDPlus:
    """Inverted generational distance plus to a reference front."""

    name = "IGD+"
    requires_reference = True

    def __call__(
        self, points: np.ndarray, reference_front: np.ndarray | None = None
    ) -> float:
        approximation = np.asarray(points, dtype=float)
        reference = _required_reference(reference_front)
        if len(approximation) == 0:
            return float("inf")
        distances = np.maximum(
            approximation[None, :, :] - reference[:, None, :], 0.0
        )
        return float(np.mean(np.min(np.linalg.norm(distances, axis=2), axis=1)))


class EpsilonPlus:
    """Additive epsilon-plus indicator to a reference front."""

    name = "epsilon+"
    requires_reference = True

    def __call__(
        self, points: np.ndarray, reference_front: np.ndarray | None = None
    ) -> float:
        approximation = np.asarray(points, dtype=float)
        reference = _required_reference(reference_front)
        if len(approximation) == 0:
            return float("inf")
        pairwise_worsening = approximation[None, :, :] - reference[:, None, :]
        return float(np.max(np.min(np.max(pairwise_worsening, axis=2), axis=1)))


class DeltaP:
    """Maximum of generational distance and inverted generational distance."""

    name = "DeltaP"
    requires_reference = True

    def __call__(
        self, points: np.ndarray, reference_front: np.ndarray | None = None
    ) -> float:
        approximation = np.asarray(points, dtype=float)
        reference = _required_reference(reference_front)
        if len(approximation) == 0:
            return float("inf")
        distances = np.linalg.norm(
            approximation[:, None, :] - reference[None, :, :], axis=2
        )
        generational_distance = float(np.mean(np.min(distances, axis=1)))
        inverted_generational_distance = float(
            np.mean(np.min(distances, axis=0))
        )
        return max(generational_distance, inverted_generational_distance)


def _required_reference(reference_front: np.ndarray | None) -> np.ndarray:
    if reference_front is None:
        raise ValueError("This indicator requires a reference front.")
    reference = np.asarray(reference_front, dtype=float)
    if reference.ndim != 2 or len(reference) == 0:
        raise ValueError("Reference front must be a non-empty 2D array.")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference front must contain only finite values.")
    return reference


IGDplus = IGDPlus
EpsPlus = EpsilonPlus
