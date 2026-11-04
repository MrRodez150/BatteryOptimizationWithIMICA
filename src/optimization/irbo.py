from __future__ import annotations

from typing import Any

import numpy as np

from .imica import (
    Individual,
    _dominates,
    _fronts,
    _nondominated_reference,
    _survivors,
)
from .operators import make_offspring, sample_design
from .problem import Variable


class IRBO:
    """Island Riez-Based Optimizer (IRBO), the NDR/Riesz benchmark variant."""

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
            winner = (
                first_index
                if ranks[first_index] < ranks[second_index]
                else second_index
            )
        else:
            first = population[first_index]
            second = population[second_index]
            if _dominates(first, second):
                winner = first_index
            elif _dominates(second, first):
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
        self.archive = [combined[index] for index in _fronts(combined)[0]]
        if len(self.archive) > self.population_size * self.island_count:
            self.archive = _survivors(
                self.archive, self.population_size * self.island_count
            )

    def _update_reference(
        self, island_index: int, candidates: list[Individual]
    ) -> None:
        self.references[island_index] = _nondominated_reference(
            self.references[island_index], candidates, self.population_size
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
                while (
                    len(offspring) < self.population_size
                    and self.evaluations < max_evaluations
                ):
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
