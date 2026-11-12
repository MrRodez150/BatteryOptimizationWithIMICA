"""Foundational discretization components for the V3 P2D simulator."""

from .galvanostatic import (
    DischargeTrajectory,
    ElectrochemicalSpeciesState,
    GalvanostaticElectrochemistry,
    GalvanostaticState,
)
from .electrolyte import (
    ElectrolytePotentialState,
    ElectrolyteTransport,
    electrolyte_conductivity_s_m,
    electrolyte_diffusivity_m2_s,
)
from .grid import ElectrochemicalGrid1D
from .kinetics import ButlerVolmerKinetics, reaction_overpotential_v
from .ocp import TabulatedOpenCircuitPotential
from .particle import SphericalParticleDiffusion
from .sei import AI2020_SEI_PARAMETERS, SEIGrowthParameters
from .reference_parameters import (
    AI2020_PARAMETER_SOURCE,
    AI2020_SOURCE_COMMIT,
    Ai2020ReferenceModel,
    ElectrodeReferenceParameters,
    build_ai2020_reference_model,
    load_ai2020_reference_ocps,
)
from .solid_potential import SolidPotentialState, solve_solid_potential

__all__ = [
    "ButlerVolmerKinetics",
    "DischargeTrajectory",
    "AI2020_PARAMETER_SOURCE",
    "AI2020_SOURCE_COMMIT",
    "ElectrochemicalGrid1D",
    "ElectrochemicalSpeciesState",
    "ElectrodeReferenceParameters",
    "GalvanostaticElectrochemistry",
    "GalvanostaticState",
    "ElectrolytePotentialState",
    "ElectrolyteTransport",
    "SphericalParticleDiffusion",
    "SEIGrowthParameters",
    "SolidPotentialState",
    "TabulatedOpenCircuitPotential",
    "Ai2020ReferenceModel",
    "AI2020_SEI_PARAMETERS",
    "build_ai2020_reference_model",
    "electrolyte_conductivity_s_m",
    "electrolyte_diffusivity_m2_s",
    "load_ai2020_reference_ocps",
    "reaction_overpotential_v",
    "solve_solid_potential",
]
