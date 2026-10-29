import numpy as np


def turns(radius_cell_m, total_layer_thickness_m):
    return 2.0 * np.pi * radius_cell_m / total_layer_thickness_m


def electrode_area(cell_height_m, total_layer_thickness_m, radius_cell_m):
    t = turns(radius_cell_m, total_layer_thickness_m)
    return (
        (cell_height_m * total_layer_thickness_m)
        * (t * np.sqrt(t**2 + 1.0) + np.log(t + np.sqrt(t**2 + 1.0)))
        / (4.0 * np.pi)
    )


def volume_fraction_from_porosity(eps, l, total_length):
    return (1.0 - eps) * l / total_length


def interfacial_area(eps_f, particle_radius_m):
    return 3.0 * (1.0 - eps_f) / particle_radius_m


def internal_resistance(current_collector_a, positive_electrode, negative_electrode, current_collector_z):
    la = current_collector_a["l"]
    sig_a = current_collector_a["sigma"]
    lp = positive_electrode["l"]
    sig_p = positive_electrode["sigma"]
    ln = negative_electrode["l"]
    sig_n = negative_electrode["sigma"]
    lz = current_collector_z["l"]
    sig_z = current_collector_z["sigma"]
    denom = la * sig_a + lp * sig_p + ln * sig_n + lz * sig_z
    if denom == 0:
        raise ValueError("Internal resistance denominator is zero.")
    return 1.0 / denom


def mass_of_battery(current_collector_a, positive_electrode, separator, negative_electrode, current_collector_z, electrolyte):
    total = (
        (positive_electrode["l"] * positive_electrode["rho"] * positive_electrode["epsf"])
        + (separator["l"] * separator["rho"] * separator["epsf"])
        + (negative_electrode["l"] * negative_electrode["rho"] * negative_electrode["epsf"])
        + (electrolyte["l"] * electrolyte["rho"] * electrolyte["epsf"])
        + (current_collector_a["l"] * current_collector_a["rho"])
        + (current_collector_z["l"] * current_collector_z["rho"])
    )
    return total
