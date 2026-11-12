from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .electrolyte import ElectrolyteTransport
from .galvanostatic import GalvanostaticElectrochemistry
from .grid import ElectrochemicalGrid1D
from .kinetics import ButlerVolmerKinetics
from .ocp import TabulatedOpenCircuitPotential
from .particle import SphericalParticleDiffusion


REFERENCE_TEMPERATURE_K = 298.15
AI2020_SOURCE_COMMIT = "b4b6089a65f736fe2444ecf4536a297cbfbd3420"
AI2020_PARAMETER_SOURCE = (
    "PyBaMM Ai2020 Enertech parameter set; see Ai et al. (2020), "
    "DOI:10.1149/2.0122001JES, and the referenced Rieger et al. (2016) "
    "LCO OCP data."
)


@dataclass(frozen=True)
class ElectrodeReferenceParameters:
    maximum_concentration_mol_m3: float
    particle_radius_m: float
    particle_diffusivity_m2_s: float
    particle_diffusivity_activation_energy_j_mol: float
    kinetic_rate_constant_reference: float
    kinetic_activation_energy_j_mol: float
    solid_conductivity_s_m: float
    electrolyte_porosity: float
    active_material_volume_fraction: float
    electrolyte_bruggeman_exponent: float
    interfacial_area_m_inv: float


@dataclass(frozen=True)
class Ai2020ReferenceModel:
    electrochemistry: GalvanostaticElectrochemistry
    positive_particles: tuple[SphericalParticleDiffusion, ...]
    negative_particles: tuple[SphericalParticleDiffusion, ...]
    initial_electrolyte_concentration_mol_m3: float
    source: str = AI2020_PARAMETER_SOURCE

    def initial_particle_states(
        self, positive_stoichiometry: float, negative_stoichiometry: float
    ) -> tuple[np.ndarray, np.ndarray]:
        for name, value, ocp, kinetics in (
            (
                "positive_stoichiometry",
                positive_stoichiometry,
                self.electrochemistry.positive_ocp,
                self.electrochemistry.positive_kinetics,
            ),
            (
                "negative_stoichiometry",
                negative_stoichiometry,
                self.electrochemistry.negative_ocp,
                self.electrochemistry.negative_kinetics,
            ),
        ):
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1).")
            if value < ocp.stoichiometry[0] or value > ocp.stoichiometry[-1]:
                raise ValueError(f"{name} is outside the tabulated OCP range.")
            if not 0.0 < value * kinetics.maximum_concentration_mol_m3 < (
                kinetics.maximum_concentration_mol_m3
            ):
                raise ValueError(f"{name} produces an invalid solid concentration.")

        positive = np.asarray(
            [
                np.full(
                    particle.radial_cells,
                    positive_stoichiometry
                    * particle.maximum_concentration_mol_m3,
                )
                for particle in self.positive_particles
            ]
        )
        negative = np.asarray(
            [
                np.full(
                    particle.radial_cells,
                    negative_stoichiometry
                    * particle.maximum_concentration_mol_m3,
                )
                for particle in self.negative_particles
            ]
        )
        return positive, negative


POSITIVE_ELECTRODE = ElectrodeReferenceParameters(
    maximum_concentration_mol_m3=49943.0,
    particle_radius_m=3.0e-6,
    particle_diffusivity_m2_s=5.387e-15,
    particle_diffusivity_activation_energy_j_mol=5000.0,
    kinetic_rate_constant_reference=1.0e-11,
    kinetic_activation_energy_j_mol=5000.0,
    solid_conductivity_s_m=10.0,
    electrolyte_porosity=0.32,
    active_material_volume_fraction=0.62,
    electrolyte_bruggeman_exponent=1.83,
    interfacial_area_m_inv=620000.0,
)

NEGATIVE_ELECTRODE = ElectrodeReferenceParameters(
    maximum_concentration_mol_m3=28700.0,
    particle_radius_m=5.0e-6,
    particle_diffusivity_m2_s=3.9e-14,
    particle_diffusivity_activation_energy_j_mol=42770.0,
    kinetic_rate_constant_reference=1.0e-11,
    kinetic_activation_energy_j_mol=5000.0,
    solid_conductivity_s_m=100.0,
    electrolyte_porosity=0.33,
    active_material_volume_fraction=0.61,
    electrolyte_bruggeman_exponent=2.914,
    interfacial_area_m_inv=3.0 * 0.61 / 5.0e-6,
)

SEPARATOR_POROSITY = 0.5
SEPARATOR_BRUGGEMAN_EXPONENT = 1.5
CATION_TRANSFERENCE_NUMBER = 0.38
INITIAL_ELECTROLYTE_CONCENTRATION_MOL_M3 = 1000.0


def _data_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "p2d_parameters"


def _read_ocp_table(path: Path) -> np.ndarray:
    try:
        table = np.loadtxt(path, delimiter=",", comments="#", dtype=float)
    except OSError as error:
        raise FileNotFoundError(f"Could not read OCP data file: {path}") from error
    except ValueError as error:
        raise ValueError(f"OCP data file is malformed: {path}") from error
    if (
        table.ndim != 2
        or table.shape[1] != 2
        or table.shape[0] < 2
        or not np.all(np.isfinite(table))
        or np.any(np.diff(table[:, 0]) <= 0.0)
    ):
        raise ValueError(f"OCP table must have two finite columns and increasing x: {path}")
    return table


def load_ai2020_reference_ocps(
    data_directory: Path | None = None,
) -> tuple[TabulatedOpenCircuitPotential, TabulatedOpenCircuitPotential]:
    """Load source-distributed Ai2020/Rieger LCO and Enertech graphite OCPs.

    The LCO table is limited to the overlap with the Ai2020 entropy fit's
    published validity interval. This reference loader supplies an isothermal
    OCP at 298.15 K; graphite entropy is not included.
    """
    directory = _data_directory() if data_directory is None else Path(data_directory)
    lco_table = _read_ocp_table(directory / "lico2_ocp_Ai2020.csv")
    graphite_table = _read_ocp_table(
        directory / "graphite_ocp_Enertech_Ai2020.csv"
    )

    lco_mask = (lco_table[:, 0] >= 0.43) & (lco_table[:, 0] <= 0.9936)
    lco_table = lco_table[lco_mask]
    graphite_table = graphite_table[
        (graphite_table[:, 0] > 0.0) & (graphite_table[:, 0] < 1.0)
    ]
    if lco_table.shape[0] < 2 or graphite_table.shape[0] < 2:
        raise ValueError("Reference OCP tables do not cover the required operating range.")

    lco_entropy = _lico2_entropy_change_ai2020(lco_table[:, 0])
    lco = TabulatedOpenCircuitPotential(
        stoichiometry=lco_table[:, 0],
        voltage_v=lco_table[:, 1],
        source=(
            "Rieger et al. (2016) LCO OCP, distributed by PyBaMM Ai2020 "
            f"parameter set at commit {AI2020_SOURCE_COMMIT}"
        ),
        reference_temperature_k=REFERENCE_TEMPERATURE_K,
        entropy_coefficient_v_k=lco_entropy,
    )
    graphite = TabulatedOpenCircuitPotential(
        stoichiometry=graphite_table[:, 0],
        voltage_v=graphite_table[:, 1],
        source=(
            "Enertech graphite OCP data distributed by PyBaMM Ai2020 "
            f"parameter set at commit {AI2020_SOURCE_COMMIT}"
        ),
        reference_temperature_k=REFERENCE_TEMPERATURE_K,
    )
    return lco, graphite


def _lico2_entropy_change_ai2020(stoichiometry: np.ndarray) -> np.ndarray:
    """Ai et al. LCO entropy fit, stated valid for 0.43 < sto < 0.9936."""
    theta = np.asarray(stoichiometry, dtype=float)
    return (
        -3.20392657 * theta**7
        + 14.5719049 * theta**6
        - 27.9047599 * theta**5
        + 29.1744564 * theta**4
        - 17.992018 * theta**3
        + 6.54799331 * theta**2
        - 1.30382445 * theta
        + 0.109667298
    )


def build_ai2020_reference_model(
    grid: ElectrochemicalGrid1D,
    radial_cells: int = 20,
    data_directory: Path | None = None,
    *,
    positive_electrolyte_porosity: float | None = None,
    separator_electrolyte_porosity: float | None = None,
    negative_electrolyte_porosity: float | None = None,
    positive_active_material_volume_fraction: float | None = None,
    negative_active_material_volume_fraction: float | None = None,
    positive_particle_radius_m: float | None = None,
    negative_particle_radius_m: float | None = None,
) -> Ai2020ReferenceModel:
    """Build the isothermal Ai2020 reference electrochemical model.

    This reference is not asserted to represent the user's experimental cell.
    Its OCP tables, material parameters, and source commit are documented in
    ``data/p2d_parameters/README.md``.
    """
    if not isinstance(radial_cells, int) or radial_cells < 2:
        raise ValueError("radial_cells must be an integer >= 2.")
    positive_porosity = (
        POSITIVE_ELECTRODE.electrolyte_porosity
        if positive_electrolyte_porosity is None
        else positive_electrolyte_porosity
    )
    separator_porosity = (
        SEPARATOR_POROSITY
        if separator_electrolyte_porosity is None
        else separator_electrolyte_porosity
    )
    negative_porosity = (
        NEGATIVE_ELECTRODE.electrolyte_porosity
        if negative_electrolyte_porosity is None
        else negative_electrolyte_porosity
    )
    positive_active_fraction = (
        POSITIVE_ELECTRODE.active_material_volume_fraction
        if positive_active_material_volume_fraction is None
        else positive_active_material_volume_fraction
    )
    negative_active_fraction = (
        NEGATIVE_ELECTRODE.active_material_volume_fraction
        if negative_active_material_volume_fraction is None
        else negative_active_material_volume_fraction
    )
    positive_radius = (
        POSITIVE_ELECTRODE.particle_radius_m
        if positive_particle_radius_m is None
        else positive_particle_radius_m
    )
    negative_radius = (
        NEGATIVE_ELECTRODE.particle_radius_m
        if negative_particle_radius_m is None
        else negative_particle_radius_m
    )
    for name, value in (
        ("positive_active_material_volume_fraction", positive_active_fraction),
        ("negative_active_material_volume_fraction", negative_active_fraction),
    ):
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1).")
    for name, value in (
        ("positive_particle_radius_m", positive_radius),
        ("negative_particle_radius_m", negative_radius),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    lco_ocp, graphite_ocp = load_ai2020_reference_ocps(data_directory)
    porosity = np.concatenate(
        (
            np.full(grid.positive_cells, positive_porosity),
            np.full(grid.separator_cells, separator_porosity),
            np.full(grid.negative_cells, negative_porosity),
        )
    )
    bruggeman = np.concatenate(
        (
            np.full(
                grid.positive_cells,
                POSITIVE_ELECTRODE.electrolyte_bruggeman_exponent,
            ),
            np.full(grid.separator_cells, SEPARATOR_BRUGGEMAN_EXPONENT),
            np.full(
                grid.negative_cells,
                NEGATIVE_ELECTRODE.electrolyte_bruggeman_exponent,
            ),
        )
    )
    electrolyte = ElectrolyteTransport(
        grid=grid,
        porosity=porosity,
        bruggeman_exponent=bruggeman,
        cation_transference_number=CATION_TRANSFERENCE_NUMBER,
    )
    positive_kinetics = ButlerVolmerKinetics(
        rate_constant_reference=POSITIVE_ELECTRODE.kinetic_rate_constant_reference,
        activation_energy_j_mol=POSITIVE_ELECTRODE.kinetic_activation_energy_j_mol,
        maximum_concentration_mol_m3=POSITIVE_ELECTRODE.maximum_concentration_mol_m3,
    )
    negative_kinetics = ButlerVolmerKinetics(
        rate_constant_reference=NEGATIVE_ELECTRODE.kinetic_rate_constant_reference,
        activation_energy_j_mol=NEGATIVE_ELECTRODE.kinetic_activation_energy_j_mol,
        maximum_concentration_mol_m3=NEGATIVE_ELECTRODE.maximum_concentration_mol_m3,
    )
    electrochemistry = GalvanostaticElectrochemistry(
        electrolyte=electrolyte,
        positive_effective_conductivity_s_m=np.full(
            grid.positive_cells, POSITIVE_ELECTRODE.solid_conductivity_s_m
        ),
        negative_effective_conductivity_s_m=np.full(
            grid.negative_cells, NEGATIVE_ELECTRODE.solid_conductivity_s_m
        ),
        positive_interfacial_area_m_inv=np.full(
            grid.positive_cells,
            3.0 * positive_active_fraction / positive_radius,
        ),
        negative_interfacial_area_m_inv=np.full(
            grid.negative_cells,
            3.0 * negative_active_fraction / negative_radius,
        ),
        positive_kinetics=positive_kinetics,
        negative_kinetics=negative_kinetics,
        positive_ocp=lco_ocp,
        negative_ocp=graphite_ocp,
    )
    positive_particles = tuple(
        SphericalParticleDiffusion(
            radius_m=positive_radius,
            radial_cells=radial_cells,
            diffusivity_m2_s=POSITIVE_ELECTRODE.particle_diffusivity_m2_s,
            activation_energy_j_mol=(
                POSITIVE_ELECTRODE.particle_diffusivity_activation_energy_j_mol
            ),
            reference_temperature_k=REFERENCE_TEMPERATURE_K,
            maximum_concentration_mol_m3=(
                POSITIVE_ELECTRODE.maximum_concentration_mol_m3
            ),
        )
        for _ in range(grid.positive_cells)
    )
    negative_particles = tuple(
        SphericalParticleDiffusion(
            radius_m=negative_radius,
            radial_cells=radial_cells,
            diffusivity_m2_s=NEGATIVE_ELECTRODE.particle_diffusivity_m2_s,
            activation_energy_j_mol=(
                NEGATIVE_ELECTRODE.particle_diffusivity_activation_energy_j_mol
            ),
            reference_temperature_k=REFERENCE_TEMPERATURE_K,
            maximum_concentration_mol_m3=(
                NEGATIVE_ELECTRODE.maximum_concentration_mol_m3
            ),
        )
        for _ in range(grid.negative_cells)
    )
    return Ai2020ReferenceModel(
        electrochemistry=electrochemistry,
        positive_particles=positive_particles,
        negative_particles=negative_particles,
        initial_electrolyte_concentration_mol_m3=(
            INITIAL_ELECTROLYTE_CONCENTRATION_MOL_M3
        ),
    )
