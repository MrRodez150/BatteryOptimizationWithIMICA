from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import make_offspring, sample_design
from .problem import Variable


@dataclass
class Individual:
    design: dict[str, float | int]
    objectives: np.ndarray
    constraints: np.ndarray
    evaluation: dict[str, Any]

    @property
    def violation(self) -> float:
        return float(np.maximum(self.constraints, 0.0).sum())

    @property
    def feasible(self) -> bool:
        return bool(np.all(self.constraints <= 0.0))


def _dominates(first: Individual, second: Individual) -> bool:
    if first.feasible != second.feasible:
        return first.feasible
    if not first.feasible and not np.isclose(first.violation, second.violation):
        return first.violation < second.violation
    return bool(
        np.all(first.objectives <= second.objectives)
        and np.any(first.objectives < second.objectives)
    )


def _fronts(population: list[Individual]) -> list[list[int]]:
    dominates: list[list[int]] = [[] for _ in population]
    dominated_by = np.zeros(len(population), dtype=int)
    first_front: list[int] = []

    for i, individual in enumerate(population):
        for j, other in enumerate(population):
            if i == j:
                continue
            if _dominates(individual, other):
                dominates[i].append(j)
            elif _dominates(other, individual):
                dominated_by[i] += 1
        if dominated_by[i] == 0:
            first_front.append(i)

    result: list[list[int]] = []
    current = first_front
    while current:
        result.append(current)
        following: list[int] = []
        for index in current:
            for dominated_index in dominates[index]:
                dominated_by[dominated_index] -= 1
                if dominated_by[dominated_index] == 0:
                    following.append(dominated_index)
        current = following
    return result


def _normalized_objectives(population: list[Individual]) -> np.ndarray:
    values = np.vstack([individual.objectives for individual in population])
    minimum = np.min(values, axis=0)
    span = np.ptp(values, axis=0)
    span[span == 0.0] = 1.0
    return (values - minimum) / span


def _riesz_contributions(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    distances = np.maximum(distances, 1e-12)
    exponent = points.shape[1] ** 2
    pair_energy = distances ** (-exponent)
    return np.sum(pair_energy, axis=1)


def _reference_coordinates(
    individuals: list[Individual],
) -> tuple[np.ndarray, np.ndarray]:
    objectives = np.vstack([individual.objectives for individual in individuals])
    feasible = np.array([individual.feasible for individual in individuals])
    if np.any(feasible):
        eligible_indices = np.flatnonzero(feasible)
        return objectives[feasible], eligible_indices

    violations = np.vstack(
        [np.maximum(individual.constraints, 0.0) for individual in individuals]
    )
    violation_scale = np.max(violations, axis=0)
    violation_scale[violation_scale == 0.0] = 1.0
    scaled_violations = violations / violation_scale
    return np.column_stack((objectives, scaled_violations)), np.arange(
        len(individuals)
    )


def _nondominated_reference(
    previous: list[Individual],
    candidates: list[Individual],
    capacity: int,
) -> list[Individual]:
    combined = previous + candidates
    if not combined:
        return []

    coordinates, candidate_indices = _reference_coordinates(combined)

    # Equal reference coordinates are redundant and can otherwise make the
    # Riesz energy singular.
    unique_coordinates, unique_positions = np.unique(
        coordinates, axis=0, return_index=True
    )
    reference_indices = candidate_indices[unique_positions]

    dominated = np.zeros(len(unique_coordinates), dtype=bool)
    for i, point in enumerate(unique_coordinates):
        no_worse = np.all(unique_coordinates <= point, axis=1)
        strictly_better = np.any(unique_coordinates < point, axis=1)
        dominated[i] = np.any(no_worse & strictly_better)
    reference_indices = reference_indices[~dominated]

    reference = [combined[index] for index in reference_indices]
    if len(reference) > capacity:
        while len(reference) > capacity:
            coordinates, _ = _reference_coordinates(reference)
            minimum = np.min(coordinates, axis=0)
            span = np.ptp(coordinates, axis=0)
            span[span == 0.0] = 1.0
            normalized = (coordinates - minimum) / span
            contribution = _riesz_contributions(normalized)
            del reference[int(np.argmax(contribution))]
    return reference


def _survivors(
    population: list[Individual],
    capacity: int,
    reference: list[Individual] | None = None,
) -> list[Individual]:
    if reference:
        seen: set[int] = set()
        unique_population = []
        for individual in population + reference:
            if id(individual) not in seen:
                seen.add(id(individual))
                unique_population.append(individual)
        population = unique_population

    selected: list[Individual] = []
    for front_indices in _fronts(population):
        front = [population[index] for index in front_indices]
        remaining = capacity - len(selected)
        if len(front) <= remaining:
            selected.extend(front)
            continue

        # Remove the point contributing most to the normalized Riesz energy;
        # this eliminates locally crowded points from the split front.
        while len(front) > remaining:
            normalized = _normalized_objectives(front)
            contribution = _riesz_contributions(normalized)
            del front[int(np.argmax(contribution))]
        selected.extend(front)
        break
    return selected


class IMICA:
    """Island-based multi-objective evolutionary optimizer.

    The implementation uses constraint-aware Pareto sorting, SBX/polynomial
    mutation, Riesz-energy survivor selection, and periodic ring migration.
    """

    def __init__(
        self,
        problem: Any,
        variables: tuple[Variable, ...] | None = None,
        population_size: int = 40,
        islands: int = 5,
        migration_interval: int = 10,
        migrants: int = 2,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.variables = variables or tuple(problem.variables)
        self.population_size = population_size
        self.island_count = islands
        self.migration_interval = migration_interval
        self.migrant_count = migrants
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0
        self.archive: list[Individual] = []
        self.references: list[list[Individual]] = []
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.island_count < 1:
            raise ValueError("islands must be at least 1.")
        if self.migration_interval < 1:
            raise ValueError("migration_interval must be at least 1.")
        if not 0 <= self.migrant_count < self.population_size:
            raise ValueError("migrants must be in [0, population_size).")

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
        self, population: list[Individual], ranks: dict[int, int]
    ) -> Individual:
        candidates = self.rng.choice(len(population), size=2, replace=False)
        first_index, second_index = int(candidates[0]), int(candidates[1])
        if ranks[first_index] != ranks[second_index]:
            winner = first_index if ranks[first_index] < ranks[second_index] else second_index
        else:
            front = [population[first_index], population[second_index]]
            if _dominates(front[0], front[1]):
                winner = first_index
            elif _dominates(front[1], front[0]):
                winner = second_index
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

    def _update_archive(self, candidates: list[Individual]) -> None:
        combined = self.archive + candidates
        if not combined:
            return
        self.archive = [
            combined[index]
            for index in _fronts(combined)[0]
        ]
        if len(self.archive) > self.population_size * self.island_count:
            self.archive = _survivors(
                self.archive, self.population_size * self.island_count
            )

    def _update_reference(
        self, island_index: int, candidates: list[Individual]
    ) -> None:
        self.references[island_index] = _nondominated_reference(
            self.references[island_index],
            candidates,
            self.population_size,
        )

    def _migrate(
        self,
        populations: list[list[Individual]],
        references: list[list[Individual]],
    ) -> None:
        if self.migrant_count == 0 or self.island_count < 2:
            return
        migrants = []
        for population in populations:
            ranked = _survivors(population, self.population_size)
            migrants.append(ranked[: self.migrant_count])

        for island_index, population in enumerate(populations):
            incoming = migrants[(island_index - 1) % self.island_count]
            references[island_index] = _nondominated_reference(
                references[island_index],
                population + incoming,
                self.population_size,
            )
            populations[island_index] = _survivors(
                population + incoming,
                self.population_size,
                references[island_index],
            )

    def run(self, max_evaluations: int) -> dict[str, Any]:
        if max_evaluations < self.population_size * self.island_count:
            raise ValueError(
                "max_evaluations must cover the initial population of every island."
            )

        self.evaluations = 0
        self.archive = []
        self.references = []
        populations = [
            [
                self._evaluate(sample_design(self.variables, self.rng))
                for _ in range(self.population_size)
            ]
            for _ in range(self.island_count)
        ]
        self.references = [
            _nondominated_reference([], population, self.population_size)
            for population in populations
        ]
        self._update_archive(
            [individual for population in populations for individual in population]
        )

        generation = 0
        while self.evaluations < max_evaluations:
            for island_index, population in enumerate(populations):
                if self.evaluations >= max_evaluations:
                    break
                ranks = self._ranks(population)
                offspring: list[Individual] = []
                while len(offspring) < self.population_size and self.evaluations < max_evaluations:
                    first = self._select_parent(population, ranks)
                    second = self._select_parent(population, ranks)
                    child_designs = make_offspring(
                        first.design, second.design, self.variables, self.rng
                    )
                    for design in child_designs:
                        if (
                            len(offspring) >= self.population_size
                            or self.evaluations >= max_evaluations
                        ):
                            break
                        offspring.append(self._evaluate(design))

                if offspring:
                    self._update_archive(offspring)
                    self._update_reference(island_index, population + offspring)
                    populations[island_index] = _survivors(
                        population + offspring,
                        self.population_size,
                        self.references[island_index],
                    )

            generation += 1
            if generation % self.migration_interval == 0:
                self._migrate(populations, self.references)

        return {
            "archive": self.archive.copy(),
            "populations": populations,
            "references": [reference.copy() for reference in self.references],
            "evaluations": self.evaluations,
            "generations": generation,
        }
