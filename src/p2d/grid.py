from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElectrochemicalGrid1D:
    """Finite-volume axial grid for positive electrode, separator, and anode."""

    positive_thickness_m: float
    separator_thickness_m: float
    negative_thickness_m: float
    positive_cells: int = 30
    separator_cells: int = 10
    negative_cells: int = 30

    def __post_init__(self) -> None:
        for name in (
            "positive_thickness_m",
            "separator_thickness_m",
            "negative_thickness_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be a positive finite length.")
        for name in ("positive_cells", "separator_cells", "negative_cells"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def cell_widths_m(self) -> np.ndarray:
        return np.concatenate(
            (
                np.full(
                    self.positive_cells,
                    self.positive_thickness_m / self.positive_cells,
                ),
                np.full(
                    self.separator_cells,
                    self.separator_thickness_m / self.separator_cells,
                ),
                np.full(
                    self.negative_cells,
                    self.negative_thickness_m / self.negative_cells,
                ),
            )
        )

    @property
    def face_positions_m(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.cell_widths_m)))

    @property
    def cell_centers_m(self) -> np.ndarray:
        faces = self.face_positions_m
        return 0.5 * (faces[:-1] + faces[1:])

    @property
    def region_slices(self) -> dict[str, slice]:
        p_end = self.positive_cells
        separator_end = p_end + self.separator_cells
        return {
            "positive_electrode": slice(0, p_end),
            "separator": slice(p_end, separator_end),
            "negative_electrode": slice(separator_end, len(self.cell_widths_m)),
        }

    @property
    def interface_positions_m(self) -> tuple[float, float]:
        return (
            self.positive_thickness_m,
            self.positive_thickness_m + self.separator_thickness_m,
        )
