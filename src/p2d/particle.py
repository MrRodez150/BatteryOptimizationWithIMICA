from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FARADAY_CONSTANT_C_MOL = 96485.33212
GAS_CONSTANT_J_MOL_K = 8.314462618


@dataclass(frozen=True)
class SphericalParticleDiffusion:
    """Implicit finite-volume diffusion in a spherical active-material particle.

    The state stores shell-averaged solid concentration in mol/m^3. A positive
    surface molar flux is defined as outward from the particle and therefore
    decreases the particle's total lithium inventory.
    """

    radius_m: float
    radial_cells: int
    diffusivity_m2_s: float
    activation_energy_j_mol: float
    reference_temperature_k: float = 298.15
    maximum_concentration_mol_m3: float = 51554.0

    def __post_init__(self) -> None:
        for name in (
            "radius_m",
            "diffusivity_m2_s",
            "reference_temperature_k",
            "maximum_concentration_mol_m3",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if (
            not np.isfinite(self.activation_energy_j_mol)
            or self.activation_energy_j_mol < 0.0
        ):
            raise ValueError("activation_energy_j_mol must be finite and nonnegative.")
        if not isinstance(self.radial_cells, int) or self.radial_cells < 2:
            raise ValueError("radial_cells must be an integer >= 2.")

    @property
    def radial_faces_m(self) -> np.ndarray:
        return np.linspace(0.0, self.radius_m, self.radial_cells + 1)

    @property
    def radial_centers_m(self) -> np.ndarray:
        faces = self.radial_faces_m
        return 0.5 * (faces[:-1] + faces[1:])

    @property
    def shell_volumes_m3(self) -> np.ndarray:
        faces = self.radial_faces_m
        return (4.0 * np.pi / 3.0) * (faces[1:] ** 3 - faces[:-1] ** 3)

    def diffusivity_at(self, temperature_k: float) -> float:
        if not np.isfinite(temperature_k) or temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive and finite.")
        exponent = (
            self.activation_energy_j_mol
            / GAS_CONSTANT_J_MOL_K
            * (1.0 / self.reference_temperature_k - 1.0 / temperature_k)
        )
        return float(self.diffusivity_m2_s * np.exp(exponent))

    def step(
        self,
        concentration_mol_m3: np.ndarray,
        time_step_s: float,
        surface_molar_flux_out_mol_m2_s: float = 0.0,
        temperature_k: float | None = None,
    ) -> np.ndarray:
        """Advance one backward-Euler step with specified outward surface flux."""
        concentration = np.asarray(concentration_mol_m3, dtype=float)
        if concentration.shape != (self.radial_cells,):
            raise ValueError(
                f"Expected {self.radial_cells} shell concentrations; "
                f"received shape {concentration.shape}."
            )
        if not np.all(np.isfinite(concentration)):
            raise ValueError("Concentration must contain only finite values.")
        if np.any(concentration < 0.0) or np.any(
            concentration > self.maximum_concentration_mol_m3
        ):
            raise ValueError("Initial particle concentration is outside physical bounds.")
        if not np.isfinite(time_step_s) or time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive and finite.")
        if not np.isfinite(surface_molar_flux_out_mol_m2_s):
            raise ValueError("Surface molar flux must be finite.")

        temperature = (
            self.reference_temperature_k
            if temperature_k is None
            else float(temperature_k)
        )
        diffusivity = self.diffusivity_at(temperature)
        faces = self.radial_faces_m
        centers = self.radial_centers_m
        volumes = self.shell_volumes_m3
        areas = 4.0 * np.pi * faces**2

        matrix = np.eye(self.radial_cells, dtype=float)
        rhs = concentration.copy()
        for index in range(self.radial_cells):
            if index > 0:
                conductance_left = (
                    diffusivity
                    * areas[index]
                    / (centers[index] - centers[index - 1])
                )
                coefficient = time_step_s * conductance_left / volumes[index]
                matrix[index, index] += coefficient
                matrix[index, index - 1] -= coefficient

            if index < self.radial_cells - 1:
                conductance_right = (
                    diffusivity
                    * areas[index + 1]
                    / (centers[index + 1] - centers[index])
                )
                coefficient = time_step_s * conductance_right / volumes[index]
                matrix[index, index] += coefficient
                matrix[index, index + 1] -= coefficient

        particle_surface_area = areas[-1]
        rhs[-1] -= (
            time_step_s
            * particle_surface_area
            * surface_molar_flux_out_mol_m2_s
            / volumes[-1]
        )
        updated = np.linalg.solve(matrix, rhs)

        tolerance = 1e-10 * self.maximum_concentration_mol_m3
        if (
            not np.all(np.isfinite(updated))
            or np.any(updated < -tolerance)
            or np.any(updated > self.maximum_concentration_mol_m3 + tolerance)
        ):
            raise RuntimeError(
                "Particle diffusion step produced a concentration outside "
                "physical bounds; reduce the time step or surface flux."
            )
        return updated

    def inventory_mol(self, concentration_mol_m3: np.ndarray) -> float:
        concentration = np.asarray(concentration_mol_m3, dtype=float)
        if concentration.shape != (self.radial_cells,):
            raise ValueError(
                f"Expected {self.radial_cells} shell concentrations; "
                f"received shape {concentration.shape}."
            )
        if not np.all(np.isfinite(concentration)):
            raise ValueError("Concentration must contain only finite values.")
        return float(np.dot(concentration, self.shell_volumes_m3))

    def surface_concentration_mol_m3(
        self, concentration_mol_m3: np.ndarray
    ) -> float:
        concentration = np.asarray(concentration_mol_m3, dtype=float)
        if concentration.shape != (self.radial_cells,):
            raise ValueError(
                f"Expected {self.radial_cells} shell concentrations; "
                f"received shape {concentration.shape}."
            )
        if not np.all(np.isfinite(concentration)):
            raise ValueError("Concentration must contain only finite values.")
        return float(concentration[-1])

    def mean_concentration_mol_m3(
        self, concentration_mol_m3: np.ndarray
    ) -> float:
        return self.inventory_mol(concentration_mol_m3) / (
            4.0 * np.pi * self.radius_m**3 / 3.0
        )

    def current_density_to_surface_molar_flux(
        self, interfacial_current_density_a_m2: float
    ) -> float:
        if not np.isfinite(interfacial_current_density_a_m2):
            raise ValueError("Interfacial current density must be finite.")
        return float(interfacial_current_density_a_m2 / FARADAY_CONSTANT_C_MOL)
