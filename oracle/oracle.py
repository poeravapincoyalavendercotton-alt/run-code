"""Oracle for the FFAG chromatic-amplitude inverse problem.

Hidden parameters (thin-octupole integrated strengths):
    (K_3 L)_F = ... (before each F magnet)
    (K_3 L)_D = ... (before each D magnet)

Returns on-momentum alpha_yy(T) precomputed on a dense KE grid over a
stable kinetic-energy window, linearly interpolated between grid
points. At two or more distinct KE values alpha_yy gives independent
constraints on (K_3 L)_F and (K_3 L)_D through their differing beta-
function weights at the octupole locations.
"""
from __future__ import annotations

from bisect import bisect_left


_KE_MIN = 145.0
_KE_MAX = 160.0


# 16-point grid, 1-MeV spacing, computed at the hidden octupole
# strengths with the reference pipeline
# (pyffag sector_4d + thin octupole + thin RF + danf 0.4.0 NormalForm).
_KE_GRID = [
    145.0, 146.0, 147.0, 148.0, 149.0,
    150.0, 151.0, 152.0, 153.0, 154.0,
    155.0, 156.0, 157.0, 158.0, 159.0,
    160.0,
]

_ALPHA_YY_GRID = [
    1.2096524564, 1.2138595119, 1.2180608268, 1.2222564476, 1.2264464202,
    1.2306307906, 1.2348096040, 1.2389829058, 1.2431507407, 1.2473131532,
    1.2514701874, 1.2556218872, 1.2597682961, 1.2639094571, 1.2680454132,
    1.2721762067,
]

assert len(_KE_GRID) == len(_ALPHA_YY_GRID)


def _interp(KE):
    if KE <= _KE_GRID[0]:
        return _ALPHA_YY_GRID[0]
    if KE >= _KE_GRID[-1]:
        return _ALPHA_YY_GRID[-1]
    i = bisect_left(_KE_GRID, KE)
    x0, x1 = _KE_GRID[i - 1], _KE_GRID[i]
    y0, y1 = _ALPHA_YY_GRID[i - 1], _ALPHA_YY_GRID[i]
    t = (KE - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


# Per-session query budget.
_MAX_QUERIES = 5
_query_count = 0


def handle_query(mode, parameters):
    """Oracle entry point.

    Modes:
        measure_alpha_yy: Measure the on-momentum first-order vertical
            amplitude-detuning coefficient alpha_yy at a given beam
            kinetic energy.
            Parameters: {"KE_MeV": float in [145.0, 160.0]}
            Returns: {"KE_MeV": float, "alpha_yy": float,
                      "queries_used": int, "queries_remaining": int}
    """
    global _query_count

    if mode != "measure_alpha_yy":
        return {
            "error": f"Unknown mode: {mode!r}. Supported modes: "
                     f"'measure_alpha_yy'.",
        }

    if not isinstance(parameters, dict):
        return {"error": "parameters must be a dict."}
    if "KE_MeV" not in parameters:
        return {"error": "parameters must contain 'KE_MeV'."}
    try:
        KE = float(parameters["KE_MeV"])
    except (TypeError, ValueError):
        return {"error": "parameters['KE_MeV'] must be numeric."}

    if not (_KE_MIN <= KE <= _KE_MAX):
        return {
            "error": f"KE_MeV out of stable window "
                     f"[{_KE_MIN}, {_KE_MAX}]: got {KE}.",
        }

    if _query_count >= _MAX_QUERIES:
        return {
            "error": f"query budget exhausted ({_MAX_QUERIES} successful "
                     f"calls). No further measurements available.",
        }

    _query_count += 1
    alpha = _interp(KE)
    return {
        "KE_MeV": KE,
        "alpha_yy": alpha,
        "queries_used": _query_count,
        "queries_remaining": _MAX_QUERIES - _query_count,
    }
