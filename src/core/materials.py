from dataclasses import dataclass


@dataclass(frozen=True)
class ElectrodeMaterial:
    name: str
    cmax_mol_m3: float
    diffusivity_m2_s: float
    reaction_rate_m_s: float
    density_kg_m3: float
    thermal_conductivity_w_mk: float
    specific_heat_j_kgk: float
    conductivity_s_m: float
    price_usd_kg: float
    average_particle_radius_m: float
    porosity: float


@dataclass(frozen=True)
class SeparatorMaterial:
    name: str
    porosity: float
    density_kg_m3: float
    thermal_conductivity_w_mk: float
    specific_heat_j_kgk: float
    price_usd_kg: float


@dataclass(frozen=True)
class ElectrolyteMaterial:
    name: str
    density_kg_m3: float
    thermal_conductivity_w_mk: float
    specific_heat_j_kgk: float
    price_usd_kg: float
    conductivity_s_m: float


LCO_MATERIAL = ElectrodeMaterial(
    name="LCO",
    cmax_mol_m3=51554.0,
    diffusivity_m2_s=1.806e-14,
    reaction_rate_m_s=7.898e-12,
    density_kg_m3=3282.0,
    thermal_conductivity_w_mk=3.4,
    specific_heat_j_kgk=1269.0,
    conductivity_s_m=1.1901,
    price_usd_kg=140.0,
    average_particle_radius_m=2.0e-6,
    porosity=0.30,
)

SEPARATOR_MATERIAL = SeparatorMaterial(
    name="separator",
    porosity=0.30,
    density_kg_m3=900.0,
    thermal_conductivity_w_mk=0.2,
    specific_heat_j_kgk=1700.0,
    price_usd_kg=223.0,
)

ELECTROLYTE_MATERIAL = ElectrolyteMaterial(
    name="electrolyte",
    density_kg_m3=1220.0,
    thermal_conductivity_w_mk=0.62,
    specific_heat_j_kgk=2000.0,
    price_usd_kg=820.0,
    conductivity_s_m=0.62,
)

GRAPHITE_DENSITY_KG_M3 = 2160.0
GRAPHITE_PRICE_USD_KG = 60.0

CURRENT_COLLECTOR_ALUMINUM = {
    "density_kg_m3": 2700.0,
    "thermal_conductivity_w_mk": 247.0,
    "specific_heat_j_kgk": 910.0,
    "price_usd_kg": 100.0,
    "conductivity_s_m": 3.55e7,
}

CURRENT_COLLECTOR_COPPER = {
    "density_kg_m3": 8960.0,
    "thermal_conductivity_w_mk": 371.0,
    "specific_heat_j_kgk": 390.0,
    "price_usd_kg": 100.0,
    "conductivity_s_m": 5.96e7,
}
