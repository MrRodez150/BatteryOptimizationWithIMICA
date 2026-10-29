import numpy as np

from .geometry import electrode_area, interfacial_area, internal_resistance, mass_of_battery
from .materials import (
    CURRENT_COLLECTOR_ALUMINUM,
    CURRENT_COLLECTOR_COPPER,
    ELECTROLYTE_MATERIAL,
    LCO_MATERIAL,
    SEPARATOR_MATERIAL,
)


class EVLCOBattery:
    """Minimal EV + LCO battery simulator used as the V3.0 baseline."""

    def __init__(self, v_pack=48.0, current_app=80.0):
        self.v_pack = float(v_pack)
        self.current_app = float(current_app)

    @staticmethod
    def _build_components(design):
        positive = {
            "l": float(design["lp"]),
            "rho": float(LCO_MATERIAL.density_kg_m3),
            "epsf": float(design["efp"]),
            "sigma": float(LCO_MATERIAL.conductivity_s_m),
            "mu": 0.09787,
            "price": float(LCO_MATERIAL.price_usd_kg),
        }

        separator = {
            "l": float(design["lo"]),
            "rho": float(SEPARATOR_MATERIAL.density_kg_m3),
            "epsf": float(design["efo"]),
            "sigma": 0.0,
            "mu": 0.0,
            "price": float(SEPARATOR_MATERIAL.price_usd_kg),
        }

        negative = {
            "l": float(design["ln"]),
            "rho": 2160.0,
            "epsf": float(design["efn"]),
            "sigma": 1000.0,
            "mu": 0.01201,
            "price": 60.0,
        }

        electrolyte = {
            "l": float(design["lp"] + design["lo"] + design["ln"]),
            "rho": float(ELECTROLYTE_MATERIAL.density_kg_m3),
            "epsf": float(1.0 - design["efp"] + 1.0 - design["efo"] + 1.0 - design["efn"]),
            "sigma": float(ELECTROLYTE_MATERIAL.conductivity_s_m),
            "mu": 0.0,
            "price": float(ELECTROLYTE_MATERIAL.price_usd_kg),
        }

        current_a = {
            "l": float(design["la"]),
            "rho": float(CURRENT_COLLECTOR_ALUMINUM["density_kg_m3"]),
            "sigma": float(CURRENT_COLLECTOR_ALUMINUM["conductivity_s_m"]),
        }

        current_z = {
            "l": float(design["lz"]),
            "rho": float(CURRENT_COLLECTOR_COPPER["density_kg_m3"]),
            "sigma": float(CURRENT_COLLECTOR_COPPER["conductivity_s_m"]),
        }

        return positive, separator, negative, electrolyte, current_a, current_z

    def evaluate(self, design):
        C = float(design["C"])
        la = float(design["la"])
        lp = float(design["lp"])
        lo = float(design["lo"])
        ln = float(design["ln"])
        lz = float(design["lz"])
        Lh = float(design["Lh"])
        Rcell = float(design["Rcell"])
        Rp = float(design["Rp"])
        Rn = float(design["Rn"])
        efp = float(design["efp"])
        efo = float(design["efo"])
        efn = float(design["efn"])
        Ns = float(design["Ns"])
        Np = float(design["Np"])

        positive, separator, negative, electrolyte, current_a, current_z = self._build_components(design)

        total_layer = la + lp + lo + ln + lz
        area_cell = electrode_area(Lh, total_layer, Rcell)
        rx = internal_resistance(current_a, positive, negative, current_z)

        mass = mass_of_battery(current_a, positive, separator, negative, current_z, electrolyte)
        if mass <= 0:
            raise ValueError("Battery mass must be positive.")

        current = np.abs(self.current_app)
        times = np.linspace(0.0, 1.0, 256)
        voltage = np.linspace(3.0, 4.0, len(times))
        current_profile = np.full_like(times, current, dtype=float)

        specific_energy = area_cell * np.trapz(current_profile * (voltage - (rx * current_profile)), times) / mass
        sei_growth = (0.5 * 96485.33212 / (8.314462618 * 298.15)) * np.mean(np.abs(voltage - 3.7))
        temp_increase = float(np.mean(voltage) - 3.7)

        price = Ns * Np * area_cell * (
            positive["price"] * positive["l"] * positive["rho"] * positive["epsf"]
            + separator["price"] * separator["l"] * separator["rho"] * separator["epsf"]
            + negative["price"] * negative["l"] * negative["rho"] * negative["epsf"]
            + electrolyte["price"] * electrolyte["l"] * electrolyte["rho"] * electrolyte["epsf"]
            + current_a["rho"] * current_a["l"] * 100.0
            + current_z["rho"] * current_z["l"] * 100.0
        )

        v_cell = voltage[-1]
        v_upper = v_cell * Ns - 1.05 * self.v_pack
        v_lower = 0.95 * self.v_pack - v_cell * Ns
        volume_violation = efp + efo + efn - 0.97

        objectives = {
            "specific_energy": float(specific_energy),
            "sei_growth": float(sei_growth),
            "temp_increase": float(temp_increase),
            "price": float(price),
        }

        constraints = {
            "upper_violation": float(v_upper),
            "lower_violation": float(v_lower),
            "volume_fraction_violation": float(volume_violation),
        }

        return {
            "design": design,
            "objectives": objectives,
            "constraints": constraints,
            "cell_area_m2": float(area_cell),
            "mass_kg": float(mass),
            "voltage_cell_v": float(v_cell),
        }
