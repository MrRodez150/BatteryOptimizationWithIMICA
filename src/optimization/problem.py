from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.core.battery import EVLCOBattery


@dataclass(frozen=True)
class Variable:
    name: str
    lower: float
    upper: float
    kind: str = "real"


EV_LCO_VARIABLES = (
    Variable("C", 0.2, 4.0),
    Variable("la", 12e-6, 30e-6),
    Variable("lp", 40e-6, 250e-6),
    Variable("lo", 10e-6, 100e-6),
    Variable("ln", 40e-6, 250e-6),
    Variable("lz", 12e-6, 30e-6),
    Variable("Lh", 40e-3, 100e-3),
    Variable("Rp", 0.2e-6, 20e-6),
    Variable("Rn", 0.5e-6, 50e-6),
    Variable("Rcell", 4e-3, 25e-3),
    Variable("efp", 0.01, 0.6),
    Variable("efo", 0.01, 0.6),
    Variable("efn", 0.01, 0.6),
    Variable("Ns", 1, 100, "integer"),
    Variable("Np", 1, 100, "integer"),
)

OBJECTIVE_NAMES = ("specific_energy", "sei_growth", "temp_increase", "price")
CONSTRAINT_NAMES = (
    "upper_violation",
    "lower_violation",
    "volume_fraction_violation",
)


class EVLCOProblem:
    """Adapt the EV/LCO battery evaluator to IMICA's minimization interface."""

    variables = EV_LCO_VARIABLES
    objective_names = OBJECTIVE_NAMES
    constraint_names = CONSTRAINT_NAMES

    def __init__(self, battery: EVLCOBattery | None = None) -> None:
        self.battery = battery or EVLCOBattery()

    def evaluate(self, design: Mapping[str, float]) -> dict[str, Any]:
        missing = {variable.name for variable in self.variables} - design.keys()
        if missing:
            raise ValueError(f"Design is missing variables: {sorted(missing)}")

        result = self.battery.evaluate(dict(design))
        raw_objectives = result["objectives"]
        raw_constraints = result["constraints"]

        # The battery model reports energy as a positive quantity; IMICA minimizes.
        objectives = (
            -float(raw_objectives["specific_energy"]),
            float(raw_objectives["sei_growth"]),
            float(raw_objectives["temp_increase"]),
            float(raw_objectives["price"]),
        )
        constraints = (
            float(raw_constraints["upper_violation"]) / self.battery.v_pack,
            float(raw_constraints["lower_violation"]) / self.battery.v_pack,
            float(raw_constraints["volume_fraction_violation"]),
        )
        return {
            "design": dict(design),
            "objectives": objectives,
            "constraints": constraints,
            "raw_result": result,
        }
