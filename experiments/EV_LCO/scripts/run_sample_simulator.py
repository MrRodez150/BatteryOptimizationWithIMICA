import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.battery import EVLCOBattery


def main():
    simulator = EVLCOBattery(v_pack=48.0, current_app=80.0)

    design = {
        "C": 1.0,
        "la": 12e-6,
        "lp": 80e-6,
        "lo": 20e-6,
        "ln": 90e-6,
        "lz": 15e-6,
        "Lh": 0.08,
        "Rcell": 0.01,
        "Rp": 1e-6,
        "Rn": 2e-6,
        "efp": 0.25,
        "efo": 0.30,
        "efn": 0.35,
        "Ns": 20,
        "Np": 6,
    }

    result = simulator.evaluate(design)
    print("Objectives:")
    for key, value in result["objectives"].items():
        print(f"  {key}: {value}")

    print("\nConstraints:")
    for key, value in result["constraints"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
