from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .particle import FARADAY_CONSTANT_C_MOL, GAS_CONSTANT_J_MOL_K


def reaction_overpotential_v(
    solid_potential_v: np.ndarray | float,
    electrolyte_potential_v: np.ndarray | float,
    open_circuit_potential_v: np.ndarray | float,
) -> np.ndarray:
    """Return eta = phi_s - phi_e - U, matching the V1 residual convention."""
    solid, electrolyte, open_circuit = np.broadcast_arrays(
        np.asarray(solid_potential_v, dtype=float),
        np.asarray(electrolyte_potential_v, dtype=float),
        np.asarray(open_circuit_potential_v, dtype=float),
    )
    if not (
        np.all(np.isfinite(solid))
        and np.all(np.isfinite(electrolyte))
        and np.all(np.isfinite(open_circuit))
    ):
        raise ValueError("Phase and open-circuit potentials must be finite.")
    return solid - electrolyte - open_circuit


@dataclass(frozen=True)
class ButlerVolmerKinetics:
    """Symmetric Butler-Volmer kinetics using the legacy V1 rate constant.

    The legacy rate constant times the concentration product gives a molar
    flux (mol m^-2 s^-1). Faraday's constant converts that flux to A m^-2.
    """

    rate_constant_reference: float
    activation_energy_j_mol: float
    maximum_concentration_mol_m3: float
    reference_temperature_k: float = 298.15

    def __post_init__(self) -> None:
        for name in (
            "rate_constant_reference",
            "maximum_concentration_mol_m3",
            "reference_temperature_k",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if (
            not np.isfinite(self.activation_energy_j_mol)
            or self.activation_energy_j_mol < 0.0
        ):
            raise ValueError("activation_energy_j_mol must be finite and nonnegative.")

    def rate_constant_at(self, temperature_k: np.ndarray | float) -> np.ndarray:
        temperature = np.asarray(temperature_k, dtype=float)
        if (
            not np.all(np.isfinite(temperature))
            or np.any(temperature <= 0.0)
        ):
            raise ValueError("temperature_k must be positive and finite.")
        exponent = (
            self.activation_energy_j_mol
            / GAS_CONSTANT_J_MOL_K
            * (1.0 / self.reference_temperature_k - 1.0 / temperature)
        )
        return self.rate_constant_reference * np.exp(exponent)

    def exchange_current_molar_flux_mol_m2_s(
        self,
        electrolyte_concentration_mol_m3: np.ndarray | float,
        surface_solid_concentration_mol_m3: np.ndarray | float,
        temperature_k: np.ndarray | float,
    ) -> np.ndarray:
        electrolyte, solid, temperature = np.broadcast_arrays(
            np.asarray(electrolyte_concentration_mol_m3, dtype=float),
            np.asarray(surface_solid_concentration_mol_m3, dtype=float),
            np.asarray(temperature_k, dtype=float),
        )
        if (
            not np.all(np.isfinite(electrolyte))
            or np.any(electrolyte <= 0.0)
            or not np.all(np.isfinite(solid))
            or np.any(solid <= 0.0)
            or np.any(solid >= self.maximum_concentration_mol_m3)
            or not np.all(np.isfinite(temperature))
            or np.any(temperature <= 0.0)
        ):
            raise ValueError(
                "Electrolyte/solid concentrations and temperature must be "
                "finite and inside physical bounds."
            )
        exchange_flux = self.rate_constant_at(temperature) * np.sqrt(
            electrolyte
            * solid
            * (self.maximum_concentration_mol_m3 - solid)
        )
        if not np.all(np.isfinite(exchange_flux)) or np.any(exchange_flux <= 0.0):
            raise ValueError("Exchange-flux correlation produced a nonphysical value.")
        return exchange_flux

    def exchange_current_density_a_m2(
        self,
        electrolyte_concentration_mol_m3: np.ndarray | float,
        surface_solid_concentration_mol_m3: np.ndarray | float,
        temperature_k: np.ndarray | float,
    ) -> np.ndarray:
        exchange_flux = self.exchange_current_molar_flux_mol_m2_s(
            electrolyte_concentration_mol_m3,
            surface_solid_concentration_mol_m3,
            temperature_k,
        )
        return FARADAY_CONSTANT_C_MOL * exchange_flux

    def molar_flux_mol_m2_s(
        self,
        overpotential_v: np.ndarray | float,
        electrolyte_concentration_mol_m3: np.ndarray | float,
        surface_solid_concentration_mol_m3: np.ndarray | float,
        temperature_k: np.ndarray | float,
    ) -> np.ndarray:
        overpotential, electrolyte, solid, temperature = np.broadcast_arrays(
            np.asarray(overpotential_v, dtype=float),
            np.asarray(electrolyte_concentration_mol_m3, dtype=float),
            np.asarray(surface_solid_concentration_mol_m3, dtype=float),
            np.asarray(temperature_k, dtype=float),
        )
        if not np.all(np.isfinite(overpotential)):
            raise ValueError("Overpotential must be finite.")
        exchange_flux = self.exchange_current_molar_flux_mol_m2_s(
            electrolyte, solid, temperature
        )
        exponent = FARADAY_CONSTANT_C_MOL * overpotential / (
            2.0 * GAS_CONSTANT_J_MOL_K * temperature
        )
        with np.errstate(over="raise", invalid="raise"):
            try:
                flux = 2.0 * exchange_flux * np.sinh(exponent)
            except FloatingPointError as error:
                raise ValueError(
                    "Overpotential is outside the numerically representable "
                    "Butler-Volmer range."
                ) from error
        return flux

    def current_density_a_m2(
        self,
        overpotential_v: np.ndarray | float,
        electrolyte_concentration_mol_m3: np.ndarray | float,
        surface_solid_concentration_mol_m3: np.ndarray | float,
        temperature_k: np.ndarray | float,
    ) -> np.ndarray:
        overpotential, electrolyte, solid, temperature = np.broadcast_arrays(
            np.asarray(overpotential_v, dtype=float),
            np.asarray(electrolyte_concentration_mol_m3, dtype=float),
            np.asarray(surface_solid_concentration_mol_m3, dtype=float),
            np.asarray(temperature_k, dtype=float),
        )
        if not np.all(np.isfinite(overpotential)):
            raise ValueError("Overpotential must be finite.")
        flux = self.molar_flux_mol_m2_s(
            overpotential, electrolyte, solid, temperature
        )
        return FARADAY_CONSTANT_C_MOL * flux

    def overpotential_v(
        self,
        current_density_a_m2: np.ndarray | float,
        electrolyte_concentration_mol_m3: np.ndarray | float,
        surface_solid_concentration_mol_m3: np.ndarray | float,
        temperature_k: np.ndarray | float,
    ) -> np.ndarray:
        current, electrolyte, solid, temperature = np.broadcast_arrays(
            np.asarray(current_density_a_m2, dtype=float),
            np.asarray(electrolyte_concentration_mol_m3, dtype=float),
            np.asarray(surface_solid_concentration_mol_m3, dtype=float),
            np.asarray(temperature_k, dtype=float),
        )
        if not np.all(np.isfinite(current)):
            raise ValueError("Current density must be finite.")
        exchange_current = self.exchange_current_density_a_m2(
            electrolyte, solid, temperature
        )
        return (
            2.0
            * GAS_CONSTANT_J_MOL_K
            * temperature
            / FARADAY_CONSTANT_C_MOL
            * np.arcsinh(current / (2.0 * exchange_current))
        )

    @staticmethod
    def volumetric_reaction_current_a_m3(
        interfacial_area_m_inv: np.ndarray | float,
        current_density_a_m2: np.ndarray | float,
    ) -> np.ndarray:
        area, current = np.broadcast_arrays(
            np.asarray(interfacial_area_m_inv, dtype=float),
            np.asarray(current_density_a_m2, dtype=float),
        )
        if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("Interfacial area must be positive and finite.")
        if not np.all(np.isfinite(current)):
            raise ValueError("Current density must be finite.")
        return area * current
