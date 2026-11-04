from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .electrolyte import ElectrolyteTransport
from .kinetics import ButlerVolmerKinetics
from .ocp import TabulatedOpenCircuitPotential
from .particle import FARADAY_CONSTANT_C_MOL, SphericalParticleDiffusion
from .solid_potential import SolidPotentialState, solve_solid_potential


@dataclass(frozen=True)
class ElectrochemicalSpeciesState:
    electrolyte_concentration_mol_m3: np.ndarray
    positive_particle_concentration_mol_m3: np.ndarray
    negative_particle_concentration_mol_m3: np.ndarray


@dataclass(frozen=True)
class DischargeTrajectory:
    time_s: np.ndarray
    voltage_v: np.ndarray
    final_species_state: ElectrochemicalSpeciesState
    reached_cutoff: bool


@dataclass(frozen=True)
class GalvanostaticState:
    positive_reaction_current_a_m3: np.ndarray
    negative_reaction_current_a_m3: np.ndarray
    electrolyte_potential_v: np.ndarray
    positive_solid_potential_v: np.ndarray
    negative_solid_potential_v: np.ndarray
    electrolyte_current_faces_a_m2: np.ndarray
    positive_solid_current_faces_a_m2: np.ndarray
    negative_solid_current_faces_a_m2: np.ndarray
    positive_collector_potential_v: float
    negative_collector_potential_v: float
    cell_voltage_v: float
    nonlinear_residual_norm: float
    function_evaluations: int


@dataclass(frozen=True)
class GalvanostaticElectrochemistry:
    """Solve the instantaneous 1D charge/reaction state at imposed current.

    Positive reaction current is conventional current from solid to electrolyte.
    During discharge, its integral is negative in the positive electrode and
    positive in the negative electrode. Thermal and solid-diffusion evolution
    are not solved by this algebraic charge-balance closure.
    """

    electrolyte: ElectrolyteTransport
    positive_effective_conductivity_s_m: np.ndarray
    negative_effective_conductivity_s_m: np.ndarray
    positive_interfacial_area_m_inv: np.ndarray
    negative_interfacial_area_m_inv: np.ndarray
    positive_kinetics: ButlerVolmerKinetics
    negative_kinetics: ButlerVolmerKinetics
    positive_ocp: TabulatedOpenCircuitPotential
    negative_ocp: TabulatedOpenCircuitPotential
    current_balance_tolerance_a_m2: float = 1.0e-8
    nonlinear_tolerance: float = 1.0e-9
    maximum_function_evaluations: int = 2000

    def __post_init__(self) -> None:
        grid = self.electrolyte.grid
        positive_count = grid.positive_cells
        negative_count = grid.negative_cells
        for name, value, count in (
            (
                "positive_effective_conductivity_s_m",
                self.positive_effective_conductivity_s_m,
                positive_count,
            ),
            (
                "negative_effective_conductivity_s_m",
                self.negative_effective_conductivity_s_m,
                negative_count,
            ),
            (
                "positive_interfacial_area_m_inv",
                self.positive_interfacial_area_m_inv,
                positive_count,
            ),
            (
                "negative_interfacial_area_m_inv",
                self.negative_interfacial_area_m_inv,
                negative_count,
            ),
        ):
            array = np.asarray(value, dtype=float).copy()
            if array.shape != (count,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite array of shape ({count},).")
            if np.any(array <= 0.0):
                raise ValueError(f"{name} values must be positive.")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

        for name, value in (
            ("current_balance_tolerance_a_m2", self.current_balance_tolerance_a_m2),
            ("nonlinear_tolerance", self.nonlinear_tolerance),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if (
            not isinstance(self.maximum_function_evaluations, int)
            or self.maximum_function_evaluations < 1
        ):
            raise ValueError("maximum_function_evaluations must be a positive integer.")

    def solve(
        self,
        applied_current_a_m2: float,
        electrolyte_concentration_mol_m3: np.ndarray,
        positive_surface_concentration_mol_m3: np.ndarray,
        negative_surface_concentration_mol_m3: np.ndarray,
        temperature_k: np.ndarray | float = 298.15,
    ) -> GalvanostaticState:
        """Solve BV kinetics and both phase charge balances at fixed current."""
        if not np.isfinite(applied_current_a_m2) or applied_current_a_m2 < 0.0:
            raise ValueError("Discharge current density must be finite and nonnegative.")

        grid = self.electrolyte.grid
        widths = grid.cell_widths_m
        regions = grid.region_slices
        positive_slice = regions["positive_electrode"]
        negative_slice = regions["negative_electrode"]
        positive_widths = widths[positive_slice]
        negative_widths = widths[negative_slice]

        electrolyte_concentration = self.electrolyte._cell_state(
            electrolyte_concentration_mol_m3,
            "electrolyte_concentration_mol_m3",
            positive=True,
        )
        temperature = self.electrolyte._cell_state(
            temperature_k, "temperature_k", positive=True
        )
        positive_surface = self._validate_surface_state(
            positive_surface_concentration_mol_m3,
            self.positive_kinetics.maximum_concentration_mol_m3,
            "positive_surface_concentration_mol_m3",
            self.positive_ocp,
            self.positive_interfacial_area_m_inv.size,
        )
        negative_surface = self._validate_surface_state(
            negative_surface_concentration_mol_m3,
            self.negative_kinetics.maximum_concentration_mol_m3,
            "negative_surface_concentration_mol_m3",
            self.negative_ocp,
            self.negative_interfacial_area_m_inv.size,
        )
        positive_temperature = temperature[positive_slice]
        negative_temperature = temperature[negative_slice]
        positive_electrolyte = electrolyte_concentration[positive_slice]
        negative_electrolyte = electrolyte_concentration[negative_slice]
        positive_ocp = self.positive_ocp.voltage_at(
            positive_surface / self.positive_kinetics.maximum_concentration_mol_m3,
            positive_temperature,
        )
        negative_ocp = self.negative_ocp.voltage_at(
            negative_surface / self.negative_kinetics.maximum_concentration_mol_m3,
            negative_temperature,
        )
        current_scale = max(applied_current_a_m2, 1.0)
        positive_reaction_scale = current_scale / float(np.sum(positive_widths))
        negative_reaction_scale = current_scale / float(np.sum(negative_widths))

        positive_guess = np.full(
            self.positive_interfacial_area_m_inv.size,
            -applied_current_a_m2 / float(np.sum(positive_widths)),
        )
        negative_guess = np.full(
            self.negative_interfacial_area_m_inv.size,
            applied_current_a_m2 / float(np.sum(negative_widths)),
        )
        variables = np.concatenate(
            (
                positive_guess / positive_reaction_scale,
                negative_guess / negative_reaction_scale,
                positive_ocp[:1],
                negative_ocp[:1],
            )
        )
        variable_count_positive = positive_guess.size
        variable_count_negative = negative_guess.size

        def residual(values: np.ndarray) -> np.ndarray:
            positive_reaction = (
                values[:variable_count_positive] * positive_reaction_scale
            )
            negative_start = variable_count_positive
            negative_reaction = (
                values[
                    negative_start : negative_start + variable_count_negative
                ]
                * negative_reaction_scale
            )
            positive_collector_potential = values[-2]
            negative_collector_potential = values[-1]

            volumetric_reaction = np.zeros(widths.size, dtype=float)
            volumetric_reaction[positive_slice] = positive_reaction
            volumetric_reaction[negative_slice] = negative_reaction
            electrolyte_right_current = float(np.dot(widths, volumetric_reaction))
            electrolyte_state = self.electrolyte.solve_potential(
                concentration_mol_m3=electrolyte_concentration,
                temperature_k=temperature,
                volumetric_reaction_current_a_m3=volumetric_reaction,
                left_boundary_ionic_current_a_m2=0.0,
                right_boundary_ionic_current_a_m2=electrolyte_right_current,
            )
            positive_solid = self._positive_solid_state(
                positive_reaction,
                positive_collector_potential,
                applied_current_a_m2,
            )
            negative_solid = self._negative_solid_state(
                negative_reaction,
                negative_collector_potential,
            )

            positive_eta = (
                positive_solid.potential_v
                - electrolyte_state.potential_v[positive_slice]
                - positive_ocp
            )
            negative_eta = (
                negative_solid.potential_v
                - electrolyte_state.potential_v[negative_slice]
                - negative_ocp
            )
            predicted_positive = self.positive_interfacial_area_m_inv * (
                self.positive_kinetics.current_density_a_m2(
                    positive_eta,
                    positive_electrolyte,
                    positive_surface,
                    positive_temperature,
                )
            )
            predicted_negative = self.negative_interfacial_area_m_inv * (
                self.negative_kinetics.current_density_a_m2(
                    negative_eta,
                    negative_electrolyte,
                    negative_surface,
                    negative_temperature,
                )
            )
            positive_constraint = (
                float(np.dot(positive_widths, positive_reaction))
                + applied_current_a_m2
            ) / current_scale
            negative_constraint = (
                float(np.dot(negative_widths, negative_reaction))
                - applied_current_a_m2
            ) / current_scale
            return np.concatenate(
                (
                    (positive_reaction - predicted_positive)
                    / positive_reaction_scale,
                    (negative_reaction - predicted_negative)
                    / negative_reaction_scale,
                    [positive_constraint, negative_constraint],
                )
            )

        solution = least_squares(
            residual,
            variables,
            method="lm",
            xtol=self.nonlinear_tolerance,
            ftol=self.nonlinear_tolerance,
            gtol=self.nonlinear_tolerance,
            max_nfev=self.maximum_function_evaluations,
        )
        residual_values = residual(solution.x)
        residual_norm = float(np.linalg.norm(residual_values, ord=np.inf))
        if not solution.success or not np.isfinite(residual_norm):
            raise RuntimeError(
                "Galvanostatic electrochemical solve did not converge: "
                f"{solution.message}; scaled residual infinity norm={residual_norm:.3e}."
            )
        if residual_norm > max(10.0 * self.nonlinear_tolerance, 1.0e-7):
            raise RuntimeError(
                "Galvanostatic solve stopped without satisfying the residual "
                f"tolerance; scaled residual infinity norm={residual_norm:.3e}."
            )

        positive_reaction = (
            solution.x[:variable_count_positive] * positive_reaction_scale
        )
        negative_start = variable_count_positive
        negative_reaction = (
            solution.x[
                negative_start : negative_start + variable_count_negative
            ]
            * negative_reaction_scale
        )
        positive_collector_potential = float(solution.x[-2])
        negative_collector_potential = float(solution.x[-1])

        volumetric_reaction = np.zeros(widths.size, dtype=float)
        volumetric_reaction[positive_slice] = positive_reaction
        volumetric_reaction[negative_slice] = negative_reaction
        net_ionic_current = float(np.dot(widths, volumetric_reaction))
        positive_current_error = abs(
            float(np.dot(positive_widths, positive_reaction))
            + applied_current_a_m2
        )
        negative_current_error = abs(
            float(np.dot(negative_widths, negative_reaction))
            - applied_current_a_m2
        )
        if max(positive_current_error, negative_current_error, abs(net_ionic_current)) > (
            self.current_balance_tolerance_a_m2
        ):
            raise RuntimeError(
                "Galvanostatic solution violates electrode current balance: "
                f"positive={positive_current_error:.3e}, "
                f"negative={negative_current_error:.3e}, "
                f"electrolyte net={abs(net_ionic_current):.3e} A/m^2."
            )
        electrolyte_state = self.electrolyte.solve_potential(
            concentration_mol_m3=electrolyte_concentration,
            temperature_k=temperature,
            volumetric_reaction_current_a_m3=volumetric_reaction,
            left_boundary_ionic_current_a_m2=0.0,
            right_boundary_ionic_current_a_m2=0.0,
        )
        positive_solid = self._positive_solid_state(
            positive_reaction,
            positive_collector_potential,
            applied_current_a_m2,
        )
        negative_solid = self._negative_solid_state(
            negative_reaction,
            negative_collector_potential,
        )
        positive_collector = self._positive_collector_value(
            positive_solid, positive_collector_potential
        )
        negative_collector = self._negative_collector_value(negative_solid)

        return GalvanostaticState(
            positive_reaction_current_a_m3=positive_reaction,
            negative_reaction_current_a_m3=negative_reaction,
            electrolyte_potential_v=electrolyte_state.potential_v,
            positive_solid_potential_v=positive_solid.potential_v,
            negative_solid_potential_v=negative_solid.potential_v,
            electrolyte_current_faces_a_m2=electrolyte_state.ionic_current_faces_a_m2,
            positive_solid_current_faces_a_m2=positive_solid.electronic_current_faces_a_m2,
            negative_solid_current_faces_a_m2=negative_solid.electronic_current_faces_a_m2,
            positive_collector_potential_v=positive_collector,
            negative_collector_potential_v=negative_collector,
            cell_voltage_v=positive_collector - negative_collector,
            nonlinear_residual_norm=residual_norm,
            function_evaluations=int(solution.nfev),
        )

    def simulate_constant_current(
        self,
        applied_current_a_m2: float,
        electrolyte_concentration_initial_mol_m3: np.ndarray,
        positive_particle_concentration_initial_mol_m3: np.ndarray,
        negative_particle_concentration_initial_mol_m3: np.ndarray,
        positive_particles: tuple[SphericalParticleDiffusion, ...],
        negative_particles: tuple[SphericalParticleDiffusion, ...],
        time_step_s: float,
        maximum_steps: int,
        cutoff_voltage_v: float = 2.5,
        temperature_k: np.ndarray | float = 298.15,
    ) -> DischargeTrajectory:
        """Run a bounded galvanostatic discharge using split species updates."""
        if not isinstance(maximum_steps, int) or maximum_steps < 1:
            raise ValueError("maximum_steps must be a positive integer.")
        if not np.isfinite(cutoff_voltage_v) or cutoff_voltage_v <= 0.0:
            raise ValueError("cutoff_voltage_v must be positive and finite.")
        if not np.isfinite(time_step_s) or time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive and finite.")

        species = ElectrochemicalSpeciesState(
            electrolyte_concentration_mol_m3=np.asarray(
                electrolyte_concentration_initial_mol_m3, dtype=float
            ).copy(),
            positive_particle_concentration_mol_m3=np.asarray(
                positive_particle_concentration_initial_mol_m3, dtype=float
            ).copy(),
            negative_particle_concentration_mol_m3=np.asarray(
                negative_particle_concentration_initial_mol_m3, dtype=float
            ).copy(),
        )
        times = [0.0]
        voltages: list[float] = []
        reached_cutoff = False

        for step_index in range(maximum_steps + 1):
            positive_surface = np.asarray(
                [
                    particle.surface_concentration_mol_m3(profile)
                    for particle, profile in zip(
                        positive_particles,
                        species.positive_particle_concentration_mol_m3,
                    )
                ]
            )
            negative_surface = np.asarray(
                [
                    particle.surface_concentration_mol_m3(profile)
                    for particle, profile in zip(
                        negative_particles,
                        species.negative_particle_concentration_mol_m3,
                    )
                ]
            )
            state = self.solve(
                applied_current_a_m2=applied_current_a_m2,
                electrolyte_concentration_mol_m3=species.electrolyte_concentration_mol_m3,
                positive_surface_concentration_mol_m3=positive_surface,
                negative_surface_concentration_mol_m3=negative_surface,
                temperature_k=temperature_k,
            )
            voltages.append(state.cell_voltage_v)
            if state.cell_voltage_v <= cutoff_voltage_v:
                reached_cutoff = True
                break
            if step_index == maximum_steps:
                break

            species = self.advance_species(
                electrochemical_state=state,
                electrolyte_concentration_old_mol_m3=species.electrolyte_concentration_mol_m3,
                positive_particle_concentration_old_mol_m3=species.positive_particle_concentration_mol_m3,
                negative_particle_concentration_old_mol_m3=species.negative_particle_concentration_mol_m3,
                positive_particles=positive_particles,
                negative_particles=negative_particles,
                time_step_s=time_step_s,
                temperature_k=temperature_k,
            )
            times.append((step_index + 1) * time_step_s)

        return DischargeTrajectory(
            time_s=np.asarray(times, dtype=float),
            voltage_v=np.asarray(voltages, dtype=float),
            final_species_state=species,
            reached_cutoff=reached_cutoff,
        )

    def advance_species(
        self,
        electrochemical_state: GalvanostaticState,
        electrolyte_concentration_old_mol_m3: np.ndarray,
        positive_particle_concentration_old_mol_m3: np.ndarray,
        negative_particle_concentration_old_mol_m3: np.ndarray,
        positive_particles: tuple[SphericalParticleDiffusion, ...],
        negative_particles: tuple[SphericalParticleDiffusion, ...],
        time_step_s: float,
        temperature_k: np.ndarray | float = 298.15,
    ) -> ElectrochemicalSpeciesState:
        """Advance electrolyte salt and spherical particle states one split step.

        The reaction field comes from ``solve`` at the old state. This is a
        sequential (operator-split) update, not a fully implicit coupled P2D
        time step.
        """
        if not np.isfinite(time_step_s) or time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive and finite.")
        positive_particles = tuple(positive_particles)
        negative_particles = tuple(negative_particles)
        if len(positive_particles) != self.electrolyte.grid.positive_cells:
            raise ValueError("One positive particle model is required per cathode cell.")
        if len(negative_particles) != self.electrolyte.grid.negative_cells:
            raise ValueError("One negative particle model is required per anode cell.")

        positive_old = np.asarray(
            positive_particle_concentration_old_mol_m3, dtype=float
        )
        negative_old = np.asarray(
            negative_particle_concentration_old_mol_m3, dtype=float
        )
        if positive_old.ndim != 2 or positive_old.shape[0] != len(positive_particles):
            raise ValueError(
                "Positive particle state must have shape (positive_cells, radial_cells)."
            )
        if negative_old.ndim != 2 or negative_old.shape[0] != len(negative_particles):
            raise ValueError(
                "Negative particle state must have shape (negative_cells, radial_cells)."
            )
        if not np.all(np.isfinite(positive_old)) or not np.all(
            np.isfinite(negative_old)
        ):
            raise ValueError("Particle concentrations must be finite.")
        for index, particle in enumerate(positive_particles):
            if positive_old.shape[1] != particle.radial_cells:
                raise ValueError("Positive particle radial state shape is inconsistent.")
            if not np.isclose(
                particle.maximum_concentration_mol_m3,
                self.positive_kinetics.maximum_concentration_mol_m3,
            ):
                raise ValueError("Positive particle and kinetic c_max values differ.")
        for index, particle in enumerate(negative_particles):
            if negative_old.shape[1] != particle.radial_cells:
                raise ValueError("Negative particle radial state shape is inconsistent.")
            if not np.isclose(
                particle.maximum_concentration_mol_m3,
                self.negative_kinetics.maximum_concentration_mol_m3,
            ):
                raise ValueError("Negative particle and kinetic c_max values differ.")

        temperature = self.electrolyte._cell_state(
            temperature_k, "temperature_k", positive=True
        )
        positive_slice = self.electrolyte.grid.region_slices["positive_electrode"]
        negative_slice = self.electrolyte.grid.region_slices["negative_electrode"]
        positive_temperature = temperature[positive_slice]
        negative_temperature = temperature[negative_slice]
        positive_updated = np.empty_like(positive_old)
        negative_updated = np.empty_like(negative_old)
        for index, particle in enumerate(positive_particles):
            interfacial_current = (
                electrochemical_state.positive_reaction_current_a_m3[index]
                / self.positive_interfacial_area_m_inv[index]
            )
            positive_updated[index] = particle.step(
                positive_old[index],
                time_step_s,
                surface_molar_flux_out_mol_m2_s=(
                    interfacial_current / FARADAY_CONSTANT_C_MOL
                ),
                temperature_k=float(positive_temperature[index]),
            )
        for index, particle in enumerate(negative_particles):
            interfacial_current = (
                electrochemical_state.negative_reaction_current_a_m3[index]
                / self.negative_interfacial_area_m_inv[index]
            )
            negative_updated[index] = particle.step(
                negative_old[index],
                time_step_s,
                surface_molar_flux_out_mol_m2_s=(
                    interfacial_current / FARADAY_CONSTANT_C_MOL
                ),
                temperature_k=float(negative_temperature[index]),
            )

        electrolyte_reaction = np.zeros(
            len(self.electrolyte.grid.cell_widths_m), dtype=float
        )
        electrolyte_reaction[positive_slice] = (
            electrochemical_state.positive_reaction_current_a_m3
        )
        electrolyte_reaction[negative_slice] = (
            electrochemical_state.negative_reaction_current_a_m3
        )
        electrolyte_updated = self.electrolyte.advance_concentration(
            electrolyte_concentration_old_mol_m3,
            time_step_s,
            volumetric_reaction_current_a_m3=electrolyte_reaction,
            temperature_k=temperature,
        )
        return ElectrochemicalSpeciesState(
            electrolyte_concentration_mol_m3=electrolyte_updated,
            positive_particle_concentration_mol_m3=positive_updated,
            negative_particle_concentration_mol_m3=negative_updated,
        )

    def _validate_surface_state(
        self,
        concentration: np.ndarray,
        maximum_concentration: float,
        name: str,
        ocp: TabulatedOpenCircuitPotential,
        expected_count: int,
    ) -> np.ndarray:
        values = np.asarray(concentration, dtype=float)
        if values.shape != (expected_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite with shape ({expected_count},).")
        if np.any(values <= 0.0) or np.any(values >= maximum_concentration):
            raise ValueError(f"{name} must lie strictly inside (0, c_max).")
        stoichiometry = values / maximum_concentration
        if (
            np.any(stoichiometry < ocp.stoichiometry[0])
            or np.any(stoichiometry > ocp.stoichiometry[-1])
        ):
            raise ValueError(f"{name} is outside the OCP table's stoichiometry range.")
        return values

    def _positive_solid_state(
        self, reaction: np.ndarray, collector_potential: float, applied_current: float
    ) -> SolidPotentialState:
        widths = self.electrolyte.grid.cell_widths_m[
            self.electrolyte.grid.region_slices["positive_electrode"]
        ]
        right_current = -applied_current - float(np.dot(widths, reaction))
        return solve_solid_potential(
            widths,
            self.positive_effective_conductivity_s_m,
            reaction,
            left_boundary_current_a_m2=-applied_current,
            right_boundary_current_a_m2=right_current,
            left_boundary_potential_v=collector_potential,
        )

    def _negative_solid_state(
        self, reaction: np.ndarray, collector_potential: float
    ) -> SolidPotentialState:
        widths = self.electrolyte.grid.cell_widths_m[
            self.electrolyte.grid.region_slices["negative_electrode"]
        ]
        right_current = -float(np.dot(widths, reaction))
        return solve_solid_potential(
            widths,
            self.negative_effective_conductivity_s_m,
            reaction,
            left_boundary_current_a_m2=0.0,
            right_boundary_current_a_m2=right_current,
            left_boundary_potential_v=collector_potential,
        )

    def _positive_collector_value(
        self, state: SolidPotentialState, first_center_potential: float
    ) -> float:
        width = self.electrolyte.grid.cell_widths_m[
            self.electrolyte.grid.region_slices["positive_electrode"]
        ][0]
        conductivity = self.positive_effective_conductivity_s_m[0]
        return float(
            first_center_potential
            + state.electronic_current_faces_a_m2[0] * width / (2.0 * conductivity)
        )

    def _negative_collector_value(
        self, state: SolidPotentialState
    ) -> float:
        widths = self.electrolyte.grid.cell_widths_m[
            self.electrolyte.grid.region_slices["negative_electrode"]
        ]
        conductivity = self.negative_effective_conductivity_s_m[-1]
        return float(
            state.potential_v[-1]
            - state.electronic_current_faces_a_m2[-1]
            * widths[-1]
            / (2.0 * conductivity)
        )
