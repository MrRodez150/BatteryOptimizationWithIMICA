from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

import numpy as np

from .materials import (
    CURRENT_COLLECTOR_ALUMINUM,
    CURRENT_COLLECTOR_COPPER,
    ELECTROLYTE_MATERIAL,
    GRAPHITE_DENSITY_KG_M3,
    GRAPHITE_PRICE_USD_KG,
    LCO_MATERIAL,
    SEPARATOR_MATERIAL,
)
from .geometry import electrode_area
from ..p2d import (
    AI2020_SEI_PARAMETERS,
    Ai2020ReferenceModel,
    ElectrochemicalGrid1D,
    ElectrochemicalSpeciesState,
    SEIGrowthParameters,
    build_ai2020_reference_model,
)
from ..p2d.particle import FARADAY_CONSTANT_C_MOL


_REQUIRED_DESIGN_FIELDS = (
    "C",
    "lp",
    "lo",
    "ln",
    "Rp",
    "Rn",
    "efp",
    "efo",
    "efn",
    "la",
    "lz",
    "Lh",
    "Rcell",
    "Ns",
    "Np",
)


@dataclass(frozen=True)
class P2DEvaluator:
    """Map an EV/LCO design to isothermal P2D objectives and constraints.

    The design fractions ``efp`` and ``efn`` are treated as active-material
    volume fractions; ``efo`` is the separator solid fraction. The remaining
    volume in each layer is electrolyte-filled porosity. Other P2D properties
    use the documented Ai2020 reference values.

    By default, evaluation also runs the configured 20-cycle CC/SEI aging
    protocol on an 8/3/8 axial mesh with 12 radial particle cells. Set
    ``aging_cycles=None`` to evaluate without that costly phase.
    """

    positive_cells: int = 8
    separator_cells: int = 3
    negative_cells: int = 8
    radial_cells: int = 12
    time_step_s: float = 10.0
    maximum_steps: int | None = None
    cutoff_voltage_v: float = 2.5
    maximum_voltage_v: float = 4.3
    aging_discharge_cutoff_voltage_v: float = 3.4
    charge_cutoff_voltage_v: float = 4.0
    aging_cycles: int | None = 20
    sei_parameters: SEIGrowthParameters = AI2020_SEI_PARAMETERS

    def __post_init__(self) -> None:
        for name in (
            "positive_cells",
            "separator_cells",
            "negative_cells",
            "radial_cells",
        ):
            value = getattr(self, name)
            minimum = 2 if name == "radial_cells" else 1
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.maximum_steps is not None and (
            not isinstance(self.maximum_steps, int) or self.maximum_steps < 1
        ):
            raise ValueError("maximum_steps must be None or a positive integer.")
        for name in (
            "time_step_s",
            "cutoff_voltage_v",
            "maximum_voltage_v",
            "aging_discharge_cutoff_voltage_v",
            "charge_cutoff_voltage_v",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.maximum_voltage_v <= self.cutoff_voltage_v:
            raise ValueError("maximum_voltage_v must exceed cutoff_voltage_v.")
        if (
            self.aging_discharge_cutoff_voltage_v <= self.cutoff_voltage_v
            or self.aging_discharge_cutoff_voltage_v >= self.maximum_voltage_v
        ):
            raise ValueError(
                "aging_discharge_cutoff_voltage_v must be between the energy "
                "cutoff and maximum_voltage_v."
            )
        if self.charge_cutoff_voltage_v <= self.aging_discharge_cutoff_voltage_v:
            raise ValueError(
                "charge_cutoff_voltage_v must exceed the aging discharge cutoff."
            )
        if self.charge_cutoff_voltage_v > self.maximum_voltage_v:
            raise ValueError(
                "charge_cutoff_voltage_v must not exceed maximum_voltage_v."
            )
        if self.aging_cycles is not None and (
            not isinstance(self.aging_cycles, int) or self.aging_cycles < 1
        ):
            raise ValueError("aging_cycles must be None or a positive integer.")

    def evaluate(self, design: Mapping[str, float]) -> dict[str, Any]:
        """Return specific energy, electrochemical constraints, and trajectory."""
        missing = set(_REQUIRED_DESIGN_FIELDS) - design.keys()
        if missing:
            raise ValueError(f"Design is missing variables: {sorted(missing)}")
        values = {name: float(design[name]) for name in _REQUIRED_DESIGN_FIELDS}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("P2D design variables must be finite.")
        for name in (
            "C",
            "lp",
            "lo",
            "ln",
            "Rp",
            "Rn",
            "la",
            "lz",
            "Lh",
            "Rcell",
            "Ns",
            "Np",
        ):
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in ("Ns", "Np"):
            if not values[name].is_integer():
                raise ValueError(f"{name} must be an integer.")
        for name in ("efp", "efo", "efn"):
            if not 0.0 < values[name] < 1.0:
                raise ValueError(f"{name} must be in (0, 1).")

        grid = ElectrochemicalGrid1D(
            positive_thickness_m=values["lp"],
            separator_thickness_m=values["lo"],
            negative_thickness_m=values["ln"],
            positive_cells=self.positive_cells,
            separator_cells=self.separator_cells,
            negative_cells=self.negative_cells,
        )
        reference = build_ai2020_reference_model(
            grid,
            radial_cells=self.radial_cells,
            positive_electrolyte_porosity=1.0 - values["efp"],
            separator_electrolyte_porosity=1.0 - values["efo"],
            negative_electrolyte_porosity=1.0 - values["efn"],
            positive_active_material_volume_fraction=values["efp"],
            negative_active_material_volume_fraction=values["efn"],
            positive_particle_radius_m=values["Rp"],
            negative_particle_radius_m=values["Rn"],
        )
        positive_ocp = reference.electrochemistry.positive_ocp
        negative_ocp = reference.electrochemistry.negative_ocp
        positive_ocp_range = float(
            positive_ocp.stoichiometry[-1] - positive_ocp.stoichiometry[0]
        )
        negative_ocp_range = float(
            negative_ocp.stoichiometry[-1] - negative_ocp.stoichiometry[0]
        )
        positive_theta_initial = float(
            positive_ocp.stoichiometry[0] + 1.0e-4 * positive_ocp_range
        )
        negative_theta_initial = float(
            negative_ocp.stoichiometry[-1] - 1.0e-4 * negative_ocp_range
        )
        positive_initial, negative_initial = reference.initial_particle_states(
            positive_theta_initial, negative_theta_initial
        )
        electrolyte_initial = np.full(
            len(grid.cell_widths_m),
            reference.initial_electrolyte_concentration_mol_m3,
        )

        positive_capacity_ah_m2 = (
            FARADAY_CONSTANT_C_MOL
            * values["lp"]
            * values["efp"]
            * reference.electrochemistry.positive_kinetics.maximum_concentration_mol_m3
            * (positive_ocp.stoichiometry[-1] - positive_theta_initial)
            / 3600.0
        )
        negative_capacity_ah_m2 = (
            FARADAY_CONSTANT_C_MOL
            * values["ln"]
            * values["efn"]
            * reference.electrochemistry.negative_kinetics.maximum_concentration_mol_m3
            * (negative_theta_initial - negative_ocp.stoichiometry[0])
            / 3600.0
        )
        capacity_ah_m2 = min(positive_capacity_ah_m2, negative_capacity_ah_m2)
        applied_current_a_m2 = values["C"] * capacity_ah_m2
        maximum_steps = self.maximum_steps
        if maximum_steps is None:
            maximum_steps = max(1, ceil(3600.0 / (values["C"] * self.time_step_s)))
        trajectory = reference.electrochemistry.simulate_constant_current(
            applied_current_a_m2=applied_current_a_m2,
            electrolyte_concentration_initial_mol_m3=electrolyte_initial,
            positive_particle_concentration_initial_mol_m3=positive_initial,
            negative_particle_concentration_initial_mol_m3=negative_initial,
            positive_particles=reference.positive_particles,
            negative_particles=reference.negative_particles,
            time_step_s=self.time_step_s,
            maximum_steps=maximum_steps,
            cutoff_voltage_v=self.cutoff_voltage_v,
        )
        aging = (
            self._evaluate_aging(
                reference=reference,
                electrolyte_initial=electrolyte_initial,
                capacity_ah_m2=capacity_ah_m2,
                cycles=self.aging_cycles,
                maximum_steps=(
                    self.maximum_steps
                    if self.maximum_steps is not None
                    else max(1, ceil(3600.0 / self.time_step_s))
                ),
            )
            if self.aging_cycles is not None
            else None
        )

        mass_kg_m2 = self._areal_mass_kg_m2(values)
        voltage_time_integral_vs = float(
            np.sum(
                0.5
                * (trajectory.voltage_v[1:] + trajectory.voltage_v[:-1])
                * np.diff(trajectory.time_s)
            )
        )
        energy_wh_m2 = (
            applied_current_a_m2
            * voltage_time_integral_vs
            / 3600.0
        )
        specific_energy = energy_wh_m2 / mass_kg_m2
        cell_area_m2 = electrode_area(
            values["Lh"],
            values["la"] + values["lp"] + values["lo"] + values["ln"] + values["lz"],
            values["Rcell"],
        )
        material_costs_usd, material_masses_kg = self._pack_material_costs(
            values, cell_area_m2
        )
        pack_cell_count = int(values["Ns"] * values["Np"])
        material_costs_usd = {
            name: cost * pack_cell_count
            for name, cost in material_costs_usd.items()
        }
        material_masses_kg = {
            name: mass * pack_cell_count
            for name, mass in material_masses_kg.items()
        }
        pack_material_cost_usd = float(sum(material_costs_usd.values()))
        ohmic_power_time_integral_j_m2 = float(
            np.sum(
                0.5
                * (
                    trajectory.ohmic_power_loss_w_m2[1:]
                    + trajectory.ohmic_power_loss_w_m2[:-1]
                )
                * np.diff(trajectory.time_s)
            )
        )
        ohmic_energy_loss_j_m2 = ohmic_power_time_integral_j_m2
        delivered_energy_j_m2 = energy_wh_m2 * 3600.0
        ohmic_energy_loss_fraction = (
            ohmic_energy_loss_j_m2
            / (ohmic_energy_loss_j_m2 + delivered_energy_j_m2)
            if ohmic_energy_loss_j_m2 + delivered_energy_j_m2 > 0.0
            else 0.0
        )
        p_range = float(positive_ocp.stoichiometry[-1] - positive_ocp.stoichiometry[0])
        n_range = float(negative_ocp.stoichiometry[-1] - negative_ocp.stoichiometry[0])
        constraints = {
            "positive_stoichiometry_lower": (
                positive_ocp.stoichiometry[0]
                - trajectory.minimum_positive_surface_stoichiometry
            )
            / p_range,
            "positive_stoichiometry_upper": (
                trajectory.maximum_positive_surface_stoichiometry
                - positive_ocp.stoichiometry[-1]
            )
            / p_range,
            "negative_stoichiometry_lower": (
                negative_ocp.stoichiometry[0]
                - trajectory.minimum_negative_surface_stoichiometry
            )
            / n_range,
            "negative_stoichiometry_upper": (
                trajectory.maximum_negative_surface_stoichiometry
                - negative_ocp.stoichiometry[-1]
            )
            / n_range,
            "electrolyte_concentration_nonnegative": (
                -trajectory.minimum_electrolyte_concentration_mol_m3
                / reference.initial_electrolyte_concentration_mol_m3
            ),
            "maximum_voltage": (
                float(np.max(trajectory.voltage_v)) / self.maximum_voltage_v - 1.0
            ),
            "discharge_completion": (
                0.0
                if trajectory.reached_cutoff or trajectory.reached_stoichiometry_limit
                else 1.0
            ),
        }
        objectives = {
            "specific_energy": float(specific_energy),
            "ohmic_energy_loss_j_m2": float(ohmic_energy_loss_j_m2),
            "price": pack_material_cost_usd,
        }
        metrics = {
            "applied_current_a_m2": float(applied_current_a_m2),
            "theoretical_capacity_ah_m2": float(capacity_ah_m2),
            "mass_kg_m2": float(mass_kg_m2),
            "cell_area_m2": float(cell_area_m2),
            "pack_cell_count": pack_cell_count,
            "pack_material_cost_usd": pack_material_cost_usd,
            "material_costs_usd": material_costs_usd,
            "material_masses_kg": material_masses_kg,
            "energy_wh_m2": float(energy_wh_m2),
            "ohmic_energy_loss_j_m2": float(ohmic_energy_loss_j_m2),
            "ohmic_energy_loss_fraction": float(ohmic_energy_loss_fraction),
            "maximum_ohmic_power_loss_w_m2": float(
                np.max(trajectory.ohmic_power_loss_w_m2)
            ),
            "discharge_time_s": float(trajectory.time_s[-1]),
            "reached_cutoff": trajectory.reached_cutoff,
            "reached_stoichiometry_limit": (
                trajectory.reached_stoichiometry_limit
            ),
            "stop_reason": (
                "voltage_cutoff"
                if trajectory.reached_cutoff
                else "stoichiometry_limit"
                if trajectory.reached_stoichiometry_limit
                else "maximum_steps"
            ),
            "minimum_electrolyte_concentration_mol_m3": (
                trajectory.minimum_electrolyte_concentration_mol_m3
            ),
            "maximum_voltage_v": float(np.max(trajectory.voltage_v)),
            "parameter_source": reference.source,
            "time_step_s": self.time_step_s,
            "maximum_steps": maximum_steps,
            "aging_model_enabled": aging is not None,
        }
        if aging is not None:
            objectives["capacity_fade_fraction"] = aging["capacity_fade_fraction"]
            metrics.update(aging["metrics"])
        result = {
            "design": dict(design),
            "objectives": objectives,
            "constraints": {
                name: float(value) for name, value in constraints.items()
            },
            "metrics": metrics,
            "mapped_parameters": {
                "electrolyte_porosity": {
                    "positive": 1.0 - values["efp"],
                    "separator": 1.0 - values["efo"],
                    "negative": 1.0 - values["efn"],
                },
                "active_material_volume_fraction": {
                    "positive": values["efp"],
                    "negative": values["efn"],
                },
                "particle_radius_m": {
                    "positive": values["Rp"],
                    "negative": values["Rn"],
                },
                "layer_thickness_m": {
                    "positive": values["lp"],
                    "separator": values["lo"],
                    "negative": values["ln"],
                },
                "initial_stoichiometry": {
                    "positive": positive_theta_initial,
                    "negative": negative_theta_initial,
                },
            },
            "trajectory": {
                "time_s": trajectory.time_s.copy(),
                "voltage_v": trajectory.voltage_v.copy(),
                "ohmic_power_loss_w_m2": (
                    trajectory.ohmic_power_loss_w_m2.copy()
                ),
            },
        }
        if aging is not None:
            result["aging"] = aging["trajectory"]
        return result

    def _evaluate_aging(
        self,
        reference: Ai2020ReferenceModel,
        electrolyte_initial: np.ndarray,
        capacity_ah_m2: float,
        cycles: int,
        maximum_steps: int,
    ) -> dict[str, Any]:
        electrochemistry = reference.electrochemistry
        positive_ocp = electrochemistry.positive_ocp
        negative_ocp = electrochemistry.negative_ocp
        positive_discharged_theta = float(
            positive_ocp.stoichiometry[-1]
            - 1.0e-4
            * (positive_ocp.stoichiometry[-1] - positive_ocp.stoichiometry[0])
        )
        negative_discharged_theta = float(
            negative_ocp.stoichiometry[0]
            + 1.0e-4
            * (negative_ocp.stoichiometry[-1] - negative_ocp.stoichiometry[0])
        )
        positive_discharged, negative_discharged = (
            reference.initial_particle_states(
                positive_discharged_theta, negative_discharged_theta
            )
        )
        species = ElectrochemicalSpeciesState(
            electrolyte_concentration_mol_m3=electrolyte_initial.copy(),
            positive_particle_concentration_mol_m3=positive_discharged,
            negative_particle_concentration_mol_m3=negative_discharged,
        )
        sei_thickness = np.full(
            self.negative_cells,
            self.sei_parameters.initial_thickness_m,
            dtype=float,
        )
        initial_sei_thickness = sei_thickness.copy()
        cycle_capacities_ah_m2: list[float] = []
        total_lithium_loss_mol_m2 = 0.0
        for cycle_index in range(cycles):
            charge = electrochemistry.simulate_constant_current(
                applied_current_a_m2=-capacity_ah_m2,
                electrolyte_concentration_initial_mol_m3=(
                    species.electrolyte_concentration_mol_m3
                ),
                positive_particle_concentration_initial_mol_m3=(
                    species.positive_particle_concentration_mol_m3
                ),
                negative_particle_concentration_initial_mol_m3=(
                    species.negative_particle_concentration_mol_m3
                ),
                positive_particles=reference.positive_particles,
                negative_particles=reference.negative_particles,
                time_step_s=self.time_step_s,
                maximum_steps=maximum_steps,
                cutoff_voltage_v=self.aging_discharge_cutoff_voltage_v,
                upper_cutoff_voltage_v=self.charge_cutoff_voltage_v,
                sei_parameters=self.sei_parameters,
                sei_thickness_initial_m=sei_thickness,
            )
            if not charge.reached_cutoff or charge.reached_stoichiometry_limit:
                raise RuntimeError(
                    f"SEI aging cycle {cycle_index + 1} did not complete its "
                    "1C charge at the configured voltage cutoff."
                )
            total_lithium_loss_mol_m2 += charge.lithium_loss_to_sei_mol_m2
            species = charge.final_species_state
            sei_thickness = charge.final_sei_thickness_m

            discharge = electrochemistry.simulate_constant_current(
                applied_current_a_m2=capacity_ah_m2,
                electrolyte_concentration_initial_mol_m3=(
                    species.electrolyte_concentration_mol_m3
                ),
                positive_particle_concentration_initial_mol_m3=(
                    species.positive_particle_concentration_mol_m3
                ),
                negative_particle_concentration_initial_mol_m3=(
                    species.negative_particle_concentration_mol_m3
                ),
                positive_particles=reference.positive_particles,
                negative_particles=reference.negative_particles,
                time_step_s=self.time_step_s,
                maximum_steps=maximum_steps,
                cutoff_voltage_v=self.aging_discharge_cutoff_voltage_v,
                sei_parameters=self.sei_parameters,
                sei_thickness_initial_m=sei_thickness,
            )
            if not discharge.reached_cutoff or discharge.reached_stoichiometry_limit:
                raise RuntimeError(
                    f"SEI aging cycle {cycle_index + 1} did not complete its "
                    "1C discharge at the configured voltage cutoff."
                )
            discharge_capacity = capacity_ah_m2 * discharge.time_s[-1] / 3600.0
            if not np.isfinite(discharge_capacity) or discharge_capacity <= 0.0:
                raise RuntimeError(
                    f"SEI aging cycle {cycle_index + 1} delivered no discharge capacity."
                )
            cycle_capacities_ah_m2.append(float(discharge_capacity))
            total_lithium_loss_mol_m2 += discharge.lithium_loss_to_sei_mol_m2
            species = discharge.final_species_state
            sei_thickness = discharge.final_sei_thickness_m

        initial_capacity = cycle_capacities_ah_m2[0]
        final_capacity = cycle_capacities_ah_m2[-1]
        capacity_fade_fraction = (initial_capacity - final_capacity) / initial_capacity
        sei_growth = float(np.mean(sei_thickness - initial_sei_thickness))
        return {
            "capacity_fade_fraction": float(capacity_fade_fraction),
            "metrics": {
                "aging_cycles_requested": cycles,
                "aging_cycles_completed": len(cycle_capacities_ah_m2),
                "aging_protocol_current_a_m2": float(capacity_ah_m2),
                "aging_protocol_c_rate": 1.0,
                "aging_charge_cutoff_voltage_v": self.charge_cutoff_voltage_v,
                "aging_discharge_cutoff_voltage_v": (
                    self.aging_discharge_cutoff_voltage_v
                ),
                "capacity_after_first_cycle_ah_m2": initial_capacity,
                "capacity_after_final_cycle_ah_m2": final_capacity,
                "capacity_fade_percent": float(100.0 * capacity_fade_fraction),
                "sei_growth_mean_m": sei_growth,
                "sei_thickness_final_mean_m": float(np.mean(sei_thickness)),
                "lithium_loss_to_sei_mol_m2": float(total_lithium_loss_mol_m2),
                "sei_parameter_source": (
                    "PyBaMM Ai2020 parameter set; reaction-limited SEI model"
                ),
            },
            "trajectory": {
                "cycle_index": np.arange(1, cycles + 1, dtype=int),
                "discharge_capacity_ah_m2": np.asarray(
                    cycle_capacities_ah_m2, dtype=float
                ),
                "sei_thickness_m": sei_thickness.copy(),
            },
        }

    @staticmethod
    def _areal_mass_kg_m2(design: Mapping[str, float]) -> float:
        electrolyte_porosity = (
            (1.0 - design["efp"]) * design["lp"]
            + (1.0 - design["efo"]) * design["lo"]
            + (1.0 - design["efn"]) * design["ln"]
        )
        mass = (
            design["efp"] * design["lp"] * LCO_MATERIAL.density_kg_m3
            + design["efo"] * design["lo"] * SEPARATOR_MATERIAL.density_kg_m3
            + design["efn"] * design["ln"] * GRAPHITE_DENSITY_KG_M3
            + electrolyte_porosity * ELECTROLYTE_MATERIAL.density_kg_m3
            + design["la"] * CURRENT_COLLECTOR_ALUMINUM["density_kg_m3"]
            + design["lz"] * CURRENT_COLLECTOR_COPPER["density_kg_m3"]
        )
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("Computed cell areal mass must be positive and finite.")
        return float(mass)

    @staticmethod
    def _pack_material_costs(
        design: Mapping[str, float], cell_area_m2: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return material costs and masses for one wound cell."""
        areal_masses_kg_m2 = {
            "LCO": design["efp"] * design["lp"] * LCO_MATERIAL.density_kg_m3,
            "graphite": (
                design["efn"] * design["ln"] * GRAPHITE_DENSITY_KG_M3
            ),
            "separator": (
                design["efo"] * design["lo"] * SEPARATOR_MATERIAL.density_kg_m3
            ),
            "electrolyte": (
                (
                    (1.0 - design["efp"]) * design["lp"]
                    + (1.0 - design["efo"]) * design["lo"]
                    + (1.0 - design["efn"]) * design["ln"]
                )
                * ELECTROLYTE_MATERIAL.density_kg_m3
            ),
            "aluminum": (
                design["la"] * CURRENT_COLLECTOR_ALUMINUM["density_kg_m3"]
            ),
            "copper": design["lz"] * CURRENT_COLLECTOR_COPPER["density_kg_m3"],
        }
        prices_usd_kg = {
            "LCO": LCO_MATERIAL.price_usd_kg,
            "graphite": GRAPHITE_PRICE_USD_KG,
            "separator": SEPARATOR_MATERIAL.price_usd_kg,
            "electrolyte": ELECTROLYTE_MATERIAL.price_usd_kg,
            "aluminum": CURRENT_COLLECTOR_ALUMINUM["price_usd_kg"],
            "copper": CURRENT_COLLECTOR_COPPER["price_usd_kg"],
        }
        masses_kg = {
            name: float(areal_mass * cell_area_m2)
            for name, areal_mass in areal_masses_kg_m2.items()
        }
        costs_usd = {
            name: float(masses_kg[name] * prices_usd_kg[name])
            for name in masses_kg
        }
        if not all(np.isfinite(value) and value >= 0.0 for value in costs_usd.values()):
            raise RuntimeError("Computed material cost is non-finite or negative.")
        return costs_usd, masses_kg
