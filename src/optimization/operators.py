from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .problem import Variable


def sample_design(
    variables: Sequence[Variable], rng: np.random.Generator
) -> dict[str, float | int]:
    design: dict[str, float | int] = {}
    for variable in variables:
        value = rng.uniform(variable.lower, variable.upper)
        design[variable.name] = (
            int(rng.integers(int(variable.lower), int(variable.upper) + 1))
            if variable.kind == "integer"
            else float(value)
        )

    fraction_names = ("efp", "efo", "efn")
    by_name = {variable.name: variable for variable in variables}
    if all(name in by_name for name in fraction_names):
        fractions = rng.dirichlet((1.0, 1.0, 1.0, 1.0))[:3]
        for name, value in zip(fraction_names, fractions):
            bounds = by_name[name]
            design[name] = float(np.clip(value, bounds.lower, bounds.upper))
    return design


def _reflect(value: float, lower: float, upper: float) -> float:
    width = upper - lower
    if width <= 0:
        raise ValueError("Variable upper bound must exceed its lower bound.")
    offset = (value - lower) % (2.0 * width)
    if offset > width:
        offset = 2.0 * width - offset
    return lower + offset


def make_offspring(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    variables: Sequence[Variable],
    rng: np.random.Generator,
    crossover_probability: float = 0.9,
    mutation_probability: float | None = None,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    if mutation_probability is None:
        mutation_probability = 1.0 / len(variables)

    children: list[dict[str, float | int]] = [{}, {}]
    for variable in variables:
        low, high = variable.lower, variable.upper
        x1, x2 = float(first[variable.name]), float(second[variable.name])

        if rng.random() < crossover_probability and not np.isclose(x1, x2):
            u = rng.random()
            beta = (
                (2.0 * u) ** (1.0 / (eta_c + 1.0))
                if u <= 0.5
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
            )
            values = (
                0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2),
                0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2),
            )
            if rng.random() < 0.5:
                values = values[::-1]
        else:
            values = (x1, x2)

        for index, value in enumerate(values):
            value = _reflect(value, low, high)
            if rng.random() < mutation_probability:
                u = rng.random()
                delta = (
                    (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    if u < 0.5
                    else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                )
                value = _reflect(value + delta * (high - low), low, high)
            if variable.kind == "integer":
                children[index][variable.name] = int(
                    np.clip(round(value), low, high)
                )
            else:
                children[index][variable.name] = float(value)

    return children[0], children[1]
