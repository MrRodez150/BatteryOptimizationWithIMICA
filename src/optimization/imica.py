from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .indicators import (
    DeltaP,
    EpsilonPlus,
    IGDPlus,
    QualityIndicator,
    R2,
    SMS,
)
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
        if len(selected) == capacity:
            break
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


def _reference_objectives(reference: list[Individual]) -> np.ndarray:
    if not reference:
        return np.empty((0, 0), dtype=float)
    feasible = [individual for individual in reference if individual.feasible]
    candidates = feasible or reference
    front = _fronts(candidates)[0]
    return np.unique(
        np.vstack([candidates[index].objectives for index in front]), axis=0
    )


def _normalize_indicator_inputs(
    points: np.ndarray, reference_front: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack((points, reference_front))
    ideal = np.min(combined, axis=0)
    span = np.ptp(combined, axis=0)
    span[span == 0.0] = 1.0
    return (points - ideal) / span, (reference_front - ideal) / span


def _indicator_value(
    indicator: QualityIndicator,
    points: np.ndarray,
    reference_front: np.ndarray | None,
) -> float:
    value = float(indicator(points, reference_front))
    if not np.isfinite(value):
        raise ValueError(
            f"Indicator {indicator.name!r} returned a non-finite value."
        )
    return value


def _indicator_survivors(
    population: list[Individual],
    capacity: int,
    indicator: QualityIndicator,
    reference: list[Individual],
) -> tuple[list[Individual], float]:
    reference_front = _reference_objectives(reference)
    selected: list[Individual] = []
    split_front: list[Individual] | None = None

    for front_indices in _fronts(population):
        if len(selected) == capacity:
            break
        front = [population[index] for index in front_indices]
        if len(selected) + len(front) <= capacity:
            selected.extend(front)
        else:
            split_front = front
            break

    if split_front is not None:
        remaining = capacity - len(selected)
        points = np.vstack([individual.objectives for individual in split_front])
        if reference_front.size == 0:
            reference_front = points.copy()

        while len(split_front) > remaining:
            current_points = np.vstack(
                [individual.objectives for individual in split_front]
            )
            normalized_current, normalized_reference = _normalize_indicator_inputs(
                current_points, reference_front
            )
            reference_arg = (
                normalized_reference
                if indicator.requires_reference
                else None
            )
            current_value = _indicator_value(
                indicator, normalized_current, reference_arg
            )
            contributions = []
            for index in range(len(split_front)):
                reduced = np.delete(normalized_current, index, axis=0)
                reduced_value = _indicator_value(indicator, reduced, reference_arg)
                contributions.append(abs(current_value - reduced_value))
            del split_front[int(np.argmin(contributions))]

        selected.extend(split_front)

    selected_front = _fronts(selected)
    front_objectives = np.vstack(
        [selected[index].objectives for index in selected_front[0]]
    )
    if reference_front.size == 0:
        reference_front = front_objectives.copy()
    normalized_front, normalized_reference = _normalize_indicator_inputs(
        front_objectives, reference_front
    )
    reference_arg = normalized_reference if indicator.requires_reference else None
    value = _indicator_value(indicator, normalized_front, reference_arg)
    return selected, value


class IMICA:
    """Indicator-driven island optimizer following the original IMICA design.

    Each island is assigned one quality indicator. Pareto fronts enforce
    constraint-aware survival, and the island's indicator truncates an
    overflowing front. NDR references are maintained separately per island.
    """

    _DEFAULT_INDICATORS = (SMS, R2, IGDPlus, EpsilonPlus, DeltaP)

    def __init__(
        self,
        problem: Any,
        variables: tuple[Variable, ...] | None = None,
        population_size: int = 40,
        islands: int | None = None,
        indicators: Sequence[QualityIndicator] | None = None,
        migration_interval: int = 10,
        migrants: int = 2,
        seed: int | None = None,
        select_mode: str = "Rand",
    ) -> None:
        self.problem = problem
        self.variables = variables or tuple(problem.variables)
        if indicators is None:
            island_count = 5 if islands is None else islands
            self.indicators = tuple(
                self._DEFAULT_INDICATORS[index % len(self._DEFAULT_INDICATORS)]()
                for index in range(island_count)
            )
        else:
            self.indicators = tuple(indicators)
            island_count = len(self.indicators) if islands is None else islands
            if island_count != len(self.indicators):
                raise ValueError("Provide exactly one indicator per island.")

        self.population_size = population_size
        self.island_count = island_count
        self.migration_interval = migration_interval
        self.migrant_count = migrants
        self.select_mode = select_mode
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
        if self.select_mode not in {"Rand", "Roulette"}:
            raise ValueError("select_mode must be 'Rand' or 'Roulette'.")
        for indicator in self.indicators:
            if (
                not callable(indicator)
                or not getattr(indicator, "name", None)
                or not isinstance(getattr(indicator, "requires_reference", None), bool)
            ):
                raise ValueError(
                    "Each indicator must be callable, named, and declare "
                    "whether it requires a reference front."
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

    def _select_parents(
        self, population: list[Individual], ranks: dict[int, int]
    ) -> tuple[Individual, Individual]:
        if self.select_mode == "Rand":
            indices = self.rng.choice(len(population), size=2, replace=False)
        else:
            probabilities = np.asarray(
                [1.0 / (ranks[index] + 1.0) for index in range(len(population))]
            )
            probabilities /= np.sum(probabilities)
            indices = self.rng.choice(
                len(population), size=2, replace=False, p=probabilities
            )
        return population[int(indices[0])], population[int(indices[1])]

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
        self.archive = [combined[index] for index in _fronts(combined)[0]]
        if len(self.archive) > self.population_size * self.island_count:
            self.archive = _survivors(
                self.archive, self.population_size * self.island_count
            )

    def _record_indicator(
        self,
        history: list[dict[str, Any]],
        generation: int,
        island_index: int,
        population: list[Individual],
        reference: list[Individual],
        value: float,
    ) -> None:
        history.append(
            {
                "generation": generation,
                "island": island_index,
                "indicator": self.indicators[island_index].name,
                "value": value,
                "feasible_count": sum(individual.feasible for individual in population),
                "reference_size": len(reference),
            }
        )

    def _migrate(
        self,
        populations: list[list[Individual]],
        history: list[dict[str, Any]],
        generation: int,
    ) -> None:
        if self.migrant_count == 0 or self.island_count < 2:
            return
        incoming: list[list[Individual]] = [
            [] for _ in range(self.island_count)
        ]
        for source_index, population in enumerate(populations):
            for destination_index in range(self.island_count):
                if source_index == destination_index:
                    continue
                migrant_indices = self.rng.choice(
                    len(population), size=self.migrant_count, replace=False
                )
                incoming[destination_index].extend(
                    population[int(index)] for index in migrant_indices
                )

        for island_index in range(self.island_count):
            candidates = populations[island_index] + incoming[island_index]
            reference = self.references[island_index]
            populations[island_index], value = _indicator_survivors(
                candidates,
                self.population_size,
                self.indicators[island_index],
                reference,
            )
            self.references[island_index] = _nondominated_reference(
                reference, populations[island_index], self.population_size
            )
            self._record_indicator(
                history,
                generation,
                island_index,
                populations[island_index],
                reference,
                value,
            )

    def run(self, max_evaluations: int) -> dict[str, Any]:
        if max_evaluations < self.population_size * self.island_count:
            raise ValueError(
                "max_evaluations must cover the initial population of every island."
            )

        self.evaluations = 0
        self.archive = []
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

        indicator_history: list[dict[str, Any]] = []
        for island_index, population in enumerate(populations):
            value = self._score_population(island_index, population)
            self._record_indicator(
                indicator_history,
                0,
                island_index,
                population,
                self.references[island_index],
                value,
            )

        generation = 0
        while self.evaluations < max_evaluations:
            for island_index, population in enumerate(populations):
                if self.evaluations >= max_evaluations:
                    break
                ranks = self._ranks(population)
                offspring: list[Individual] = []
                while (
                    len(offspring) < self.population_size
                    and self.evaluations < max_evaluations
                ):
                    first, second = self._select_parents(population, ranks)
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
                    candidates = population + offspring
                    self._update_archive(offspring)
                    reference = self.references[island_index]
                    populations[island_index], value = _indicator_survivors(
                        candidates,
                        self.population_size,
                        self.indicators[island_index],
                        reference,
                    )
                    self.references[island_index] = _nondominated_reference(
                        reference, populations[island_index], self.population_size
                    )
                    self._record_indicator(
                        indicator_history,
                        generation + 1,
                        island_index,
                        populations[island_index],
                        reference,
                        value,
                    )

            generation += 1
            if generation % self.migration_interval == 0:
                self._migrate(populations, indicator_history, generation)

        final_values = [
            self._score_population(index, population)
            for index, population in enumerate(populations)
        ]
        return {
            "archive": self.archive.copy(),
            "populations": populations,
            "references": [reference.copy() for reference in self.references],
            "indicator_names": [indicator.name for indicator in self.indicators],
            "indicator_values": final_values,
            "indicator_history": indicator_history,
            "evaluations": self.evaluations,
            "generations": generation,
        }

    def _score_population(
        self, island_index: int, population: list[Individual]
    ) -> float:
        front = _fronts(population)[0]
        points = np.vstack([population[index].objectives for index in front])
        reference_front = _reference_objectives(self.references[island_index])
        if reference_front.size == 0:
            reference_front = points.copy()
        normalized_points, normalized_reference = _normalize_indicator_inputs(
            points, reference_front
        )
        reference_arg = (
            normalized_reference
            if self.indicators[island_index].requires_reference
            else None
        )
        return _indicator_value(
            self.indicators[island_index], normalized_points, reference_arg
        )
