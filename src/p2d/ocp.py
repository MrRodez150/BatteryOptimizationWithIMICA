from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabulatedOpenCircuitPotential:
    """Source-traceable OCP table with linear interpolation and no extrapolation."""

    stoichiometry: np.ndarray
    voltage_v: np.ndarray
    source: str
    reference_temperature_k: float = 298.15
    entropy_coefficient_v_k: np.ndarray | None = None

    def __post_init__(self) -> None:
        stoichiometry = np.asarray(self.stoichiometry, dtype=float).copy()
        voltage = np.asarray(self.voltage_v, dtype=float).copy()
        if stoichiometry.ndim != 1 or stoichiometry.size < 2:
            raise ValueError("stoichiometry must be a one-dimensional array of length >= 2.")
        if voltage.shape != stoichiometry.shape:
            raise ValueError("voltage_v must have the same shape as stoichiometry.")
        if (
            not np.all(np.isfinite(stoichiometry))
            or np.any(stoichiometry <= 0.0)
            or np.any(stoichiometry >= 1.0)
            or np.any(np.diff(stoichiometry) <= 0.0)
        ):
            raise ValueError(
                "Stoichiometry values must be finite, strictly increasing, and in (0, 1)."
            )
        if not np.all(np.isfinite(voltage)) or np.any(voltage <= 0.0):
            raise ValueError("OCP voltages must be positive and finite.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("A nonempty literature or data-source description is required.")
        if (
            not np.isfinite(self.reference_temperature_k)
            or self.reference_temperature_k <= 0.0
        ):
            raise ValueError("reference_temperature_k must be positive and finite.")

        entropy = self.entropy_coefficient_v_k
        if entropy is not None:
            entropy = np.asarray(entropy, dtype=float).copy()
            if entropy.shape != stoichiometry.shape or not np.all(np.isfinite(entropy)):
                raise ValueError(
                    "entropy_coefficient_v_k must be finite and match stoichiometry."
                )
            entropy.setflags(write=False)

        stoichiometry.setflags(write=False)
        voltage.setflags(write=False)
        object.__setattr__(self, "stoichiometry", stoichiometry)
        object.__setattr__(self, "voltage_v", voltage)
        object.__setattr__(self, "entropy_coefficient_v_k", entropy)

    def voltage_at(
        self,
        stoichiometry: np.ndarray | float,
        temperature_k: np.ndarray | float | None = None,
    ) -> np.ndarray:
        theta = np.asarray(stoichiometry, dtype=float)
        if not np.all(np.isfinite(theta)):
            raise ValueError("Stoichiometry must be finite.")
        if np.any(theta < self.stoichiometry[0]) or np.any(
            theta > self.stoichiometry[-1]
        ):
            raise ValueError(
                "Stoichiometry is outside the tabulated range; OCP extrapolation "
                "is disabled."
            )
        voltage = np.interp(theta, self.stoichiometry, self.voltage_v)
        if temperature_k is None:
            return voltage
        temperature = np.asarray(temperature_k, dtype=float)
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("temperature_k must be positive and finite.")
        if self.entropy_coefficient_v_k is None:
            if np.any(temperature != self.reference_temperature_k):
                raise ValueError(
                    "Temperature-dependent OCP requested without an entropy "
                    "coefficient table."
                )
            return voltage
        entropy = np.interp(theta, self.stoichiometry, self.entropy_coefficient_v_k)
        return voltage + entropy * (temperature - self.reference_temperature_k)
