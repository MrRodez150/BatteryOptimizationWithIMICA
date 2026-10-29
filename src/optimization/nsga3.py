from __future__ import annotations

from math import comb
from typing import Any

import numpy as np

from .imica import (
    Individual,
    _fronts,
    _nondominated_reference,
)
from .operators import make_offspring, sample_design
from .problem import Variable


def _reference_directions(
    objective_count: int, minimum_count: int
) -> np.ndarray:
    if objective_count < 2:
        raise ValueError("NSGA-III requires at least two objectives.")
    if minimum_count < 1:
        raise ValueError("Population size must be positive.")

    divisions = 1
    while (
        comb(divisions + objective_count - 1, objective_count - 1)
        < minimum_count
    ):
        divisions += 1

    directions: list[list[float]] = []

    def generate(prefix: list[int], remaining: int, dimensions: int) -> None:
        if dimensions == 1:
            directions.append(
                [value / divisions for value in prefix + [remaining]]
            )
            return
        for value in range(remaining + 1):
            generate(prefix + [value], remaining - value, dimensions - 1)

    generate([], divisions, objective_count)
    return np.asarray(directions, dtype=float)


def _normalize_objectives(population: list[Individual]) -> np.ndarray:
    objectives = np.vstack([individual.objectives for individual in population])
    ideal = np.min(objectives, axis=0)
    shifted = objectives - ideal
    objective_count = shifted.shape[1]
    weights = np.full((objective_count, objective_count), 1e-6)
    np.fill_diagonal(weights, 1.0)
    asf = np.max(shifted[:, None, :] / weights[None, :, :], axis=2)
    extreme_indices = np.argmin(asf, axis=0)
    extreme_points = shifted[extreme_indices]

    try:
        plane = np.linalg.solve(extreme_points, np.ones(objective_count))
        intercepts = 1.0 / plane
    except np.linalg.LinAlgError:
        intercepts = np.max(shifted, axis=0)

    fallback = np.max(shifted, axis=0)
    invalid = (
        ~np.isfinite(intercepts)
        | (intercepts <= 1e-12)
        | (intercepts < np.max(shifted, axis=0) * 1e-6)
    )
    intercepts[invalid] = fallback[invalid]
    intercepts[intercepts <= 1e-12] = 1.0
    return shifted / intercepts


def _associate(
    normalized_objectives: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    direction_norms = np.linalg.norm(directions, axis=1)
    unit_directions = directions / direction_norms[:, None]
    projections = normalized_objectives @ unit_directions.T
    distances = np.linalg.norm(
        normalized_objectives[:, None, :]
        - projections[:, :, None] * unit_directions[None, :, :],
        axis=2,
    )
    associations = np.argmin(distances, axis=1)
    perpendicular_distances = distances[np.arange(len(distances)), associations]
    return associations, perpendicular_distances


def _environmental_selection(
    combined: list[Individual],
    population_size: int,
    directions: np.ndarray,
    rng: np.random.Generator,
) -> list[Individual]:
    fronts = _fronts(combined)
    selected_indices: list[int] = []
    split_front: list[int] | None = None

    for front in fronts:
        if len(selected_indices) + len(front) <= population_size:
            selected_indices.extend(front)
        else:
            split_front = front
            break

    if split_front is not None:
        normalized = _normalize_objectives(combined)
        associations, distances = _associate(normalized, directions)
        niche_count = np.bincount(
            associations[selected_indices], minlength=len(directions)
        )
        candidates_by_niche: list[list[int]] = [
            [] for _ in range(len(directions))
        ]
        for index in split_front:
            candidates_by_niche[int(associations[index])].append(index)

        needed = population_size - len(selected_indices)
        while needed > 0:
            available = [
                niche
                for niche, candidates in enumerate(candidates_by_niche)
                if candidates
            ]
            minimum_count = min(niche_count[niche] for niche in available)
            least_occupied = [
                niche
                for niche in available
                if niche_count[niche] == minimum_count
            ]
            niche = int(rng.choice(least_occupied))
            candidates = candidates_by_niche[niche]

            if niche_count[niche] == 0:
                chosen_position = int(rng.integers(len(candidates)))
            else:
                chosen_position = int(
                    np.argmin([distances[index] for index in candidates])
                )
            selected_indices.append(candidates.pop(chosen_position))
            niche_count[niche] += 1
            needed -= 1

    return [combined[index] for index in selected_indices]


class NSGA3:
    """NSGA-III using the same mixed-variable operators and evaluator as IMICA."""

    def __init__(
        self,
        problem: Any,
        variables: tuple[Variable, ...] | None = None,
        population_size: int = 200,
        seed: int | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        self.problem = problem
        self.variables = variables or tuple(problem.variables)
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0
        self.archive: list[Individual] = []
        self.reference_directions = _reference_directions(
            len(problem.objective_names), population_size
        )

    def _evaluate(self, design: dict[str, float | int]) -> Individual:
        evaluation = self.problem.evaluate(design)
        objectives = np.asarray(evaluation["objectives"], dtype=float)
        constraints = np.asarray(evaluation["constraints"], dtype=float)
        if objectives.ndim != 1 or constraints.ndim != 1:
            raise ValueError("Problem objectives and constraints must be 1D.")
        if not np.all(np.isfinite(objectives)) or not np.all(
            np.isfinite(constraints)
        ):
            raise ValueError("Problem returned non-finite objectives or constraints.")
        self.evaluations += 1
        return Individual(design, objectives, constraints, evaluation)

    def _select_parent(
        self,
        population: list[Individual],
        ranks: dict[int, int],
    ) -> Individual:
        first_index, second_index = (
            int(index)
            for index in self.rng.choice(len(population), size=2, replace=False)
        )
        if ranks[first_index] != ranks[second_index]:
            winner = (
                first_index
                if ranks[first_index] < ranks[second_index]
                else second_index
            )
        else:
            winner = first_index if self.rng.random() < 0.5 else second_index
        return population[winner]

    @staticmethod
    def _ranks(population: list[Individual]) -> dict[int, int]:
        return {
            index: rank
            for rank, front in enumerate(_fronts(population))
            for index in front
        }

    def run(self, max_evaluations: int) -> dict[str, Any]:
        if max_evaluations < self.population_size:
            raise ValueError(
                "max_evaluations must cover the initial NSGA-III population."
            )

        self.evaluations = 0
        self.archive = []
        population = [
            self._evaluate(sample_design(self.variables, self.rng))
            for _ in range(self.population_size)
        ]
        self.archive = _nondominated_reference(
            [], population, self.population_size
        )
        generations = 0

        while self.evaluations < max_evaluations:
            ranks = self._ranks(population)
            offspring: list[Individual] = []
            while (
                len(offspring) < self.population_size
                and self.evaluations < max_evaluations
            ):
                first = self._select_parent(population, ranks)
                second = self._select_parent(population, ranks)
                children = make_offspring(
                    first.design, second.design, self.variables, self.rng
                )
                for design in children:
                    if (
                        len(offspring) >= self.population_size
                        or self.evaluations >= max_evaluations
                    ):
                        break
                    offspring.append(self._evaluate(design))

            if offspring:
                self.archive = _nondominated_reference(
                    self.archive + population + offspring,
                    [],
                    self.population_size,
                )
                population = _environmental_selection(
                    population + offspring,
                    self.population_size,
                    self.reference_directions,
                    self.rng,
                )
            generations += 1

        return {
            "archive": self.archive.copy(),
            "population": population,
            "evaluations": self.evaluations,
            "generations": generations,
            "reference_directions": self.reference_directions.copy(),
        }
