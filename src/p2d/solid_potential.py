from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolidPotentialState:
    potential_v: np.ndarray
    electronic_current_faces_a_m2: np.ndarray


def solve_solid_potential(
    cell_widths_m: np.ndarray,
    effective_conductivity_s_m: np.ndarray,
    volumetric_reaction_current_a_m3: np.ndarray,
    left_boundary_current_a_m2: float,
    right_boundary_current_a_m2: float,
    left_boundary_potential_v: float = 0.0,
    current_balance_tolerance_a_m2: float = 1.0e-8,
) -> SolidPotentialState:
    """Solve 1D solid-phase charge conservation in one electrode.

    Positive reaction current is defined from solid to electrolyte. Therefore
    the positive-x solid current decreases by the integrated reaction current:
    ``i_s,right - i_s,left = -integral(a*j_F dx)``. The first cell-center
    potential supplies the gauge/reference value.
    """
    widths = np.asarray(cell_widths_m, dtype=float)
    conductivity = np.asarray(effective_conductivity_s_m, dtype=float)
    reaction_current = np.asarray(volumetric_reaction_current_a_m3, dtype=float)
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError("cell_widths_m must be a nonempty one-dimensional array.")
    if conductivity.shape != widths.shape or reaction_current.shape != widths.shape:
        raise ValueError("Conductivity and reaction current must match the cell grid.")
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("Cell widths must be positive and finite.")
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("Effective solid conductivity must be positive and finite.")
    if not np.all(np.isfinite(reaction_current)):
        raise ValueError("Volumetric reaction current must be finite.")
    for name, value in (
        ("left_boundary_current_a_m2", left_boundary_current_a_m2),
        ("right_boundary_current_a_m2", right_boundary_current_a_m2),
        ("left_boundary_potential_v", left_boundary_potential_v),
        ("current_balance_tolerance_a_m2", current_balance_tolerance_a_m2),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if current_balance_tolerance_a_m2 < 0.0:
        raise ValueError("current_balance_tolerance_a_m2 must be nonnegative.")

    current_faces = np.empty(widths.size + 1, dtype=float)
    current_faces[0] = left_boundary_current_a_m2
    current_faces[1:] = left_boundary_current_a_m2 - np.cumsum(
        reaction_current * widths
    )
    if abs(current_faces[-1] - right_boundary_current_a_m2) > (
        current_balance_tolerance_a_m2
    ):
        raise ValueError(
            "Reaction-current integral is incompatible with the specified "
            "solid-phase boundary currents."
        )
    current_faces[-1] = right_boundary_current_a_m2

    centers = np.cumsum(widths) - 0.5 * widths
    potential = np.empty(widths.size, dtype=float)
    potential[0] = left_boundary_potential_v
    for left in range(widths.size - 1):
        right = left + 1
        distance_left = 0.5 * widths[left]
        distance_right = 0.5 * widths[right]
        face_conductivity = 1.0 / (
            distance_left / conductivity[left]
            + distance_right / conductivity[right]
        )
        potential_gradient = -current_faces[right] / face_conductivity
        potential[right] = potential[left] + potential_gradient * (
            centers[right] - centers[left]
        )

    return SolidPotentialState(
        potential_v=potential,
        electronic_current_faces_a_m2=current_faces,
    )
