from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .particle import FARADAY_CONSTANT_C_MOL


GAS_CONSTANT_J_MOL_K = 8.31446261815324


@dataclass(frozen=True)
class SEIGrowthParameters:
    """Reaction-limited SEI parameters from the PyBaMM Ai2020 set."""

    exchange_current_density_a_m2: float = 1.5e-7
    resistivity_ohm_m: float = 2.0e5
    partial_molar_volume_m3_mol: float = 9.585e-5
    lithium_to_sei_mole_ratio: float = 2.0
    open_circuit_potential_v: float = 0.4
    initial_thickness_m: float = 5.0e-9
    transfer_coefficient: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "exchange_current_density_a_m2",
            "resistivity_ohm_m",
            "partial_molar_volume_m3_mol",
            "lithium_to_sei_mole_ratio",
            "initial_thickness_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if not np.isfinite(self.open_circuit_potential_v):
            raise ValueError("open_circuit_potential_v must be finite.")
        if not 0.0 < self.transfer_coefficient < 1.0:
            raise ValueError("transfer_coefficient must be in (0, 1).")

    def current_density_a_m2(
        self, sei_overpotential_v: np.ndarray, temperature_k: np.ndarray
    ) -> np.ndarray:
        """Return the negative reduction current for reaction-limited SEI growth."""
        overpotential = np.asarray(sei_overpotential_v, dtype=float)
        temperature = np.asarray(temperature_k, dtype=float)
        if overpotential.shape != temperature.shape:
            raise ValueError("SEI overpotential and temperature shapes must match.")
        if (
            not np.all(np.isfinite(overpotential))
            or not np.all(np.isfinite(temperature))
            or np.any(temperature <= 0.0)
        ):
            raise ValueError("SEI overpotential and temperature must be finite.")
        exponent = (
            -self.transfer_coefficient
            * FARADAY_CONSTANT_C_MOL
            * overpotential
            / (GAS_CONSTANT_J_MOL_K * temperature)
        )
        if np.any(exponent > 700.0):
            raise RuntimeError("SEI reaction rate overflowed its exponential.")
        current = -self.exchange_current_density_a_m2 * np.exp(exponent)
        if not np.all(np.isfinite(current)):
            raise RuntimeError("SEI reaction current is non-finite.")
        return current


AI2020_SEI_PARAMETERS = SEIGrowthParameters()
