from __future__ import annotations

# Hidden specimen: fixed composition and geometry (angstrom). Neutral singlet.
_ATOMS = ["Bi", "H", "H", "H"]
_COORDS = [
    [0.000000, 0.000000, 0.000000],
    [1.459600, 0.000000, -1.019000],
    [-0.729800, 1.264000, -1.019000],
    [-0.729800, -1.264000, -1.019000],
]

_KB = 0.0019872041  # kcal/mol/K (provided for convenience; not needed here)


def handle_query(mode: str, parameters: dict | None = None) -> dict:
    parameters = parameters or {}

    if mode == "help":
        return {
            "description": "A sealed cell holding one neutral, closed-shell molecule at a fixed "
                           "nuclear geometry. The instrument reports the specimen's composition and "
                           "geometry only; it computes no properties. The molecule is a singlet.",
            "modes": {
                "structure": "{} -> {atoms: [element symbols], coordinates_angstrom: [[x, y, z], ...], "
                             "unit: 'angstrom', charge: 0, spin: 0}  the element and position of every "
                             "atom in the specimen",
            },
            "units": {"length": "angstrom", "energy_conversion": "1 hartree = 27.211386 eV"},
        }

    if mode == "structure":
        return {
            "atoms": list(_ATOMS),
            "coordinates_angstrom": [list(c) for c in _COORDS],
            "unit": "angstrom",
            "charge": 0,
            "spin": 0,
        }

    return {"error": f"Unknown mode: {mode}. Use mode='help' to see available modes."}


query_oracle = handle_query
