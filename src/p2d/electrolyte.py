from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import ElectrochemicalGrid1D
from .particle import FARADAY_CONSTANT_C_MOL, GAS_CONSTANT_J_MOL_K


def electrolyte_diffusivity_m2_s(
    concentration_mol_m3: np.ndarray | float,
    temperature_k: np.ndarray | float,
) -> np.ndarray:
    """Return the V1 Valoen-Reimers electrolyte diffusivity correlation."""
    concentration, temperature = np.broadcast_arrays(
        np.asarray(concentration_mol_m3, dtype=float),
        np.asarray(temperature_k, dtype=float),
    )
    if (
        not np.all(np.isfinite(concentration))
        or np.any(concentration <= 0.0)
        or not np.all(np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("Electrolyte concentration and temperature must be positive.")
    denominator = temperature - 229.0 - 5.0e-3 * concentration
    if np.any(denominator <= 0.0):
        raise ValueError("State is outside the diffusivity correlation domain.")
    exponent = (
        -4.43
        - 54.0 / denominator
        - 0.22e-3 * concentration
    )
    diffusivity = 1.0e-4 * np.power(10.0, exponent)
    if not np.all(np.isfinite(diffusivity)) or np.any(diffusivity <= 0.0):
        raise ValueError("Diffusivity correlation produced a nonphysical value.")
    return diffusivity


def electrolyte_conductivity_s_m(
    concentration_mol_m3: np.ndarray | float,
    temperature_k: np.ndarray | float,
) -> np.ndarray:
    """Return the bulk electrolyte conductivity correlation used in V1."""
    concentration, temperature = np.broadcast_arrays(
        np.asarray(concentration_mol_m3, dtype=float),
        np.asarray(temperature_k, dtype=float),
    )
    if (
        not np.all(np.isfinite(concentration))
        or np.any(concentration <= 0.0)
        or not np.all(np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("Electrolyte concentration and temperature must be positive.")
    polynomial = (
        -10.5
        + 0.668e-3 * concentration
        + 0.494e-6 * concentration**2
        + (
            0.074
            - 1.78e-5 * concentration
            - 8.86e-10 * concentration**2
        )
        * temperature
        + (-6.96e-5 + 2.8e-8 * concentration) * temperature**2
    )
    conductivity = 1.0e-4 * concentration * polynomial**2
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("Conductivity correlation produced a nonphysical value.")
    return conductivity


@dataclass(frozen=True)
class ElectrolytePotentialState:
    potential_v: np.ndarray
    ionic_current_faces_a_m2: np.ndarray


@dataclass(frozen=True)
class ElectrolyteTransport:
    """Conservative 1D electrolyte concentration and charge transport."""

    grid: ElectrochemicalGrid1D
    porosity: np.ndarray
    bruggeman_exponent: np.ndarray
    cation_transference_number: float = 0.354
    maximum_picard_iterations: int = 100
    relative_tolerance: float = 1.0e-9
    charge_balance_tolerance_a_m2: float = 1.0e-8

    def __post_init__(self) -> None:
        cell_count = len(self.grid.cell_widths_m)
        porosity = self._expand_cell_values(self.porosity, cell_count, "porosity")
        exponent = self._expand_cell_values(
            self.bruggeman_exponent, cell_count, "bruggeman_exponent"
        )
        if np.any(porosity <= 0.0) or np.any(porosity > 1.0):
            raise ValueError("Electrolyte porosity must be in (0, 1].")
        if np.any(exponent < 0.0):
            raise ValueError("Bruggeman exponents must be nonnegative.")
        if (
            not np.isfinite(self.cation_transference_number)
            or not 0.0 <= self.cation_transference_number <= 1.0
        ):
            raise ValueError("Cation transference number must be in [0, 1].")
        if (
            not isinstance(self.maximum_picard_iterations, int)
            or self.maximum_picard_iterations < 1
        ):
            raise ValueError("maximum_picard_iterations must be a positive integer.")
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive and finite.")
        if (
            not np.isfinite(self.charge_balance_tolerance_a_m2)
            or self.charge_balance_tolerance_a_m2 < 0.0
        ):
            raise ValueError("charge_balance_tolerance_a_m2 must be nonnegative.")
        porosity.setflags(write=False)
        exponent.setflags(write=False)
        object.__setattr__(self, "porosity", porosity)
        object.__setattr__(self, "bruggeman_exponent", exponent)

    @staticmethod
    def _expand_cell_values(
        values: np.ndarray | float, cell_count: int, name: str
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = np.full(cell_count, float(array))
        if array.shape != (cell_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite and scalar or have shape ({cell_count},).")
        return array.copy()

    def _cell_state(
        self, values: np.ndarray | float, name: str, *, positive: bool = False
    ) -> np.ndarray:
        array = self._expand_cell_values(values, len(self.porosity), name)
        if positive and np.any(array <= 0.0):
            raise ValueError(f"{name} values must be positive.")
        return array

    def _face_diffusion_conductances(
        self, concentration: np.ndarray, temperature: np.ndarray
    ) -> np.ndarray:
        widths = self.grid.cell_widths_m
        faces = self.grid.face_positions_m
        centers = self.grid.cell_centers_m
        conductances = np.empty(len(widths) - 1, dtype=float)
        for face in range(1, len(widths)):
            left = face - 1
            right = face
            distance_left = faces[face] - centers[left]
            distance_right = centers[right] - faces[face]
            face_concentration = 0.5 * (concentration[left] + concentration[right])
            face_temperature = 0.5 * (temperature[left] + temperature[right])
            bulk_diffusivity = float(
                electrolyte_diffusivity_m2_s(face_concentration, face_temperature)
            )
            effective_left = (
                bulk_diffusivity
                * self.porosity[left] ** self.bruggeman_exponent[left]
            )
            effective_right = (
                bulk_diffusivity
                * self.porosity[right] ** self.bruggeman_exponent[right]
            )
            conductances[left] = 1.0 / (
                distance_left / effective_left + distance_right / effective_right
            )
        return conductances

    def advance_concentration(
        self,
        concentration_old_mol_m3: np.ndarray,
        time_step_s: float,
        volumetric_reaction_current_a_m3: np.ndarray | float = 0.0,
        temperature_k: np.ndarray | float = 298.15,
    ) -> np.ndarray:
        """Advance c_e using backward Euler, zero-flux outer boundaries, and source a*j."""
        old = self._cell_state(
            concentration_old_mol_m3,
            "concentration_old_mol_m3",
            positive=True,
        )
        temperature = self._cell_state(temperature_k, "temperature_k", positive=True)
        reaction_current = self._cell_state(
            volumetric_reaction_current_a_m3, "volumetric_reaction_current_a_m3"
        )
        if not np.isfinite(time_step_s) or time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive and finite.")

        widths = self.grid.cell_widths_m
        storage = self.porosity * widths
        concentration_source = (
            (1.0 - self.cation_transference_number)
            * reaction_current
            / FARADAY_CONSTANT_C_MOL
        )
        rhs = storage * old + time_step_s * widths * concentration_source
        guess = old.copy()
        concentration_scale = max(float(np.max(old)), 1.0)

        for _ in range(self.maximum_picard_iterations):
            conductances = self._face_diffusion_conductances(guess, temperature)
            matrix = np.diag(storage)
            for left, conductance in enumerate(conductances):
                right = left + 1
                term = time_step_s * conductance
                matrix[left, left] += term
                matrix[left, right] -= term
                matrix[right, right] += term
                matrix[right, left] -= term
            updated = np.linalg.solve(matrix, rhs)
            if not np.all(np.isfinite(updated)) or np.any(updated <= 0.0):
                raise RuntimeError(
                    "Electrolyte concentration became nonpositive or nonfinite; "
                    "reduce the time step or reaction current."
                )
            error = float(np.max(np.abs(updated - guess))) / concentration_scale
            if error <= self.relative_tolerance:
                return updated
            guess = updated

        raise RuntimeError(
            "Electrolyte diffusivity Picard iteration did not converge within "
            f"{self.maximum_picard_iterations} iterations."
        )

    def solve_potential(
        self,
        concentration_mol_m3: np.ndarray,
        temperature_k: np.ndarray | float,
        volumetric_reaction_current_a_m3: np.ndarray | float,
        left_boundary_ionic_current_a_m2: float = 0.0,
        right_boundary_ionic_current_a_m2: float = 0.0,
    ) -> ElectrolytePotentialState:
        """Integrate electrolyte charge conservation with a zero-mean gauge origin.

        Positive reaction current is defined from solid into electrolyte.
        Electrode/separator and separator/electrode interfaces conserve ionic
        flux through cumulative finite-volume current balance.
        """
        concentration = self._cell_state(
            concentration_mol_m3, "concentration_mol_m3", positive=True
        )
        temperature = self._cell_state(temperature_k, "temperature_k", positive=True)
        reaction_current = self._cell_state(
            volumetric_reaction_current_a_m3, "volumetric_reaction_current_a_m3"
        )
        if not np.isfinite(left_boundary_ionic_current_a_m2) or not np.isfinite(
            right_boundary_ionic_current_a_m2
        ):
            raise ValueError("Boundary ionic currents must be finite.")

        widths = self.grid.cell_widths_m
        current_faces = np.empty(len(widths) + 1, dtype=float)
        current_faces[0] = left_boundary_ionic_current_a_m2
        current_faces[1:] = left_boundary_ionic_current_a_m2 + np.cumsum(
            reaction_current * widths
        )
        if abs(current_faces[-1] - right_boundary_ionic_current_a_m2) > (
            self.charge_balance_tolerance_a_m2
        ):
            raise ValueError(
                "Reaction-current integral is incompatible with the specified "
                "boundary ionic currents."
            )
        current_faces[-1] = right_boundary_ionic_current_a_m2

        centers = self.grid.cell_centers_m
        potential = np.zeros(len(widths), dtype=float)
        migration_factor = (
            2.0
            * (1.0 - self.cation_transference_number)
            * GAS_CONSTANT_J_MOL_K
            / FARADAY_CONSTANT_C_MOL
        )
        for left in range(len(widths) - 1):
            right = left + 1
            distance_left = self.grid.face_positions_m[right] - centers[left]
            distance_right = centers[right] - self.grid.face_positions_m[right]
            face_concentration = 0.5 * (
                concentration[left] + concentration[right]
            )
            face_temperature = 0.5 * (temperature[left] + temperature[right])
            bulk_conductivity = float(
                electrolyte_conductivity_s_m(face_concentration, face_temperature)
            )
            effective_left = (
                bulk_conductivity
                * self.porosity[left] ** self.bruggeman_exponent[left]
            )
            effective_right = (
                bulk_conductivity
                * self.porosity[right] ** self.bruggeman_exponent[right]
            )
            face_conductivity = 1.0 / (
                distance_left / effective_left + distance_right / effective_right
            )
            log_concentration_gradient = (
                np.log(concentration[right]) - np.log(concentration[left])
            ) / (centers[right] - centers[left])
            potential_gradient = (
                migration_factor
                * face_temperature
                * log_concentration_gradient
                - current_faces[right] / face_conductivity
            )
            potential[right] = (
                potential[left]
                + potential_gradient * (centers[right] - centers[left])
            )

        return ElectrolytePotentialState(
            potential_v=potential,
            ionic_current_faces_a_m2=current_faces,
        )
