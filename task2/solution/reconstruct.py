"""Reconstruct F-magnet field coefficients (including cubic a3_F) and
compute 2D first-order amplitude-dependent tune-shift tensor."""
import os, subprocess, sys
for pkg in ["daceypy", "pyffag", "danf>=0.2.1", "numpy", "scipy"]:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

import numpy as np
from scipy.optimize import least_squares
from daceypy import DA
from pyffag import sector_map_4d, compose_sequence, compose_n
from pyffag.constants import kinetic_to_brho, M_PROTON
from danf import NormalForm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle"))
from oracle import handle_query as query_oracle

B0 = 1.2
KE_ref = 150.0
Brho_ref = kinetic_to_brho(KE_ref, M_PROTON)
r0 = Brho_ref / B0
angle_F = np.radians(12.0)
angle_D = np.radians(6.0)
N_cells = 12
a1_D, a2_D = -5.5, -1.0   # D-magnet coefficients (known, disclosed in problem)

measured = []
for KE in [135.0, 145.0, 155.0]:
    r = query_oracle("measure_tune", {"KE_MeV": float(KE)})
    measured.append((KE, r["nu_x"], r["nu_y"]))


def B_local(Brho, a1, a2, a3):
    dr = Brho / B0 - r0
    return [B0 + a1 * dr + a2 * dr ** 2 + a3 * dr ** 3,
            a1 + 2 * a2 * dr + 3 * a3 * dr ** 2,
            a2 + 3 * a3 * dr,
            a3]


def model_tunes(KE, a1F, a2F, a3F):
    Brho = kinetic_to_brho(KE, M_PROTON)
    DA.init(1, 4)
    try:
        F = sector_map_4d(B_local(Brho, a1F, a2F, a3F), Brho, angle=angle_F, max_y_order=2)
        D = sector_map_4d(B_local(Brho, a1_D, a2_D, 0.0), Brho, angle=angle_D, max_y_order=2)
        ring = compose_n(compose_sequence([F, D, F]), N_cells)
    except Exception:
        return None
    M = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            e = [0, 0, 0, 0]; e[j] = 1
            M[i, j] = ring[i].getCoefficient(e)
    trx = M[0, 0] + M[1, 1]
    try_y = M[2, 2] + M[3, 3]
    if not (abs(trx) < 2 and abs(try_y) < 2):
        return None
    return (np.arccos(np.clip(trx / 2, -1, 1)) / (2 * np.pi),
            np.arccos(np.clip(try_y / 2, -1, 1)) / (2 * np.pi))


def residuals(p):
    r = []
    for KE, nux_m, nuy_m in measured:
        t = model_tunes(KE, *p)
        if t is None:
            r += [1e3, 1e3]
        else:
            r += [t[0] - nux_m, t[1] - nuy_m]
    return r


res = least_squares(residuals, [1.0, 2.0, 0.0], method="lm", xtol=1e-14, ftol=1e-14)
a1_F, a2_F, a3_F = res.x

DA.init(4, 4)
F = sector_map_4d(B_local(Brho_ref, a1_F, a2_F, a3_F), Brho_ref, angle=angle_F)
D = sector_map_4d(B_local(Brho_ref, a1_D, a2_D, 0.0), Brho_ref, angle=angle_D)
ring = compose_n(compose_sequence([F, D, F]), N_cells)

nf = NormalForm(ring)
nf.compute()
d = nf.detuning

alpha_x  = d["dnux_dJx"] / 2
alpha_xy = d["dnux_dJy"] / 2
alpha_y  = d["dnuy_dJy"] / 2

print(f"{a1_F:.3f},{a2_F:.3f},{a3_F:.3f},{alpha_x:.3f},{alpha_xy:.3f},{alpha_y:.3f}")
