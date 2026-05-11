"""Plug in maintainer-provided (K3L_F, K3L_D) and compute the rest."""
from __future__ import annotations
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_DIR))

from solution.reference_solution import (
    compute_C,
    compute_alpha_yy,
    compute_xi_y_taylor_and_dispersion,
    compute_n_T,
)

K3L_F, K3L_D = 0.193, -2.775

print(f"Using (K3L_F, K3L_D) = ({K3L_F}, {K3L_D})")

# sanity: what does the model predict for alpha_yy at the two oracle KEs?
a1 = compute_alpha_yy(146.0, K3L_F, K3L_D)
a2 = compute_alpha_yy(158.0, K3L_F, K3L_D)
print(f"model alpha_yy(146) = {a1:.6f}   (oracle = 1.213860)")
print(f"model alpha_yy(158) = {a2:.6f}   (oracle = 1.263909)")

C = compute_C(K3L_F, K3L_D)
print(f"C = {C:.6f}")
xi_y, xi_y2, xi_y3, nu_y_anchor, eta1, eta2, eta3, L0 = compute_xi_y_taylor_and_dispersion(K3L_F, K3L_D)
print(f"xi_y  = {xi_y:.6f}")
print(f"xi_y2 = {xi_y2:.6f}")
print(f"xi_y3 = {xi_y3:.6f}")
print(f"eta1 = {eta1}")
print(f"eta2 = {eta2}")
n_T_list, truth, d2nx, d2ny = compute_n_T(K3L_F, K3L_D, eta1, eta2, xi_y2, xi_y3)
print(f"truth = {truth}, n_T candidates = {n_T_list}")
print(f"d2nx = {d2nx:.6f}, d2ny = {d2ny:.6f}")

n_T = n_T_list[0] if len(n_T_list) == 1 else float('nan')
print()
print(f"{C:.3f},{xi_y2:.3f},{xi_y3:.3f},{n_T:.3f},{K3L_F:.3f},{K3L_D:.3f}")
print("expected: 0.873,1.524,-0.387,-1.474,0.193,-2.775")
