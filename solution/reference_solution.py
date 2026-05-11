"""Reference solution for the inverse FFAG chromatic-amplitude task (Tier 5, 6-tuple)."""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
from daceypy import DA, array, integrator
from daceypy.RK import RK78
from pyffag.sector_4d import _build_field_4d
from danf import NormalForm
from scipy.optimize import fsolve


# ----- Fixed kinematics + lattice -----
MP = 938.272
THETA_F = np.radians(9.0)
THETA_D = np.radians(12.0)
COEFS_F = [1.2, +2.0, +4.0]
COEFS_D = [1.2, -4.0, -5.0]
V_RF = 50e-3
H_RF = 6
PHI_S = np.pi


class _Int6D(integrator):
    _coeffs = []; _max_y_order = 4
    _H = 0.0; _BRHO = 0.0
    @staticmethod
    def f(z, s):
        x, px, y, py, ell, delta = z[0], z[1], z[2], z[3], z[4], z[5]
        By, Bx = _build_field_4d(x, y, _Int6D._H, _Int6D._coeffs, _Int6D._max_y_order)
        ohx = 1.0 + _Int6D._H * x
        ps = DA.sqrt((1.0 + delta)**2 - px*px - py*py)
        return array([
            ohx * px / ps, _Int6D._H * ps - ohx * By / _Int6D._BRHO,
            ohx * py / ps, ohx * Bx / _Int6D._BRHO,
            ohx * (1.0 + delta) / ps - 1.0, DA(0.0),
        ])


def _build_ring(KE, K3L_F, K3L_D, order):
    E0 = MP + KE
    P0 = np.sqrt(E0**2 - MP**2)
    BETA0 = P0 / E0
    BRHO = P0 / 299.792458
    B0 = COEFS_F[0]
    RHO = BRHO / B0
    H = 1.0 / RHO
    L_F = RHO * THETA_F
    L_D = RHO * THETA_D
    C_RING = 12 * (2 * L_F + L_D)
    _Int6D._H = H
    _Int6D._BRHO = BRHO

    DA.init(order, 6)
    st = [DA(1), DA(2), DA(3), DA(4), DA(5), DA(6)]

    def track(state, L, coeffs):
        _Int6D._coeffs = coeffs
        prop = _Int6D(RKcoeff=RK78(), stateType=array)
        prop.loadTime(0.0, L); prop.loadTol(1e-14, 1e-14); prop.loadStepSize()
        return [prop.propagate(array(state), 0.0, L)[i] for i in range(6)]

    def oct_kick(s, K):
        x, px, y, py, ell, delta = s
        dpx = -(K/6.0) * (x*x*x - 3*x*y*y)
        dpy = +(K/6.0) * (3*x*x*y - y*y*y)
        return [x, px + dpx, y, py + dpy, ell, delta]

    def rf_kick(s):
        x, px, y, py, ell, delta = s
        phase = H_RF * 2*np.pi*ell / C_RING + PHI_S
        d_delta = (V_RF / (BETA0**2 * E0)) * (DA.sin(phase) - np.sin(PHI_S))
        return [x, px, y, py, ell, delta + d_delta]

    def one_cell(s):
        s = oct_kick(s, K3L_F); s = track(s, L_F, COEFS_F)
        s = oct_kick(s, K3L_D); s = track(s, L_D, COEFS_D)
        s = oct_kick(s, K3L_F); s = track(s, L_F, COEFS_F)
        return s

    cell = one_cell(st)
    ring = cell
    for _ in range(11):
        ring = [cell[i].eval(ring) for i in range(6)]
    return rf_kick(ring)


def compute_alpha_yy(KE, K3L_F, K3L_D):
    ring = _build_ring(KE, K3L_F, K3L_D, order=4)
    nf = NormalForm(ring); nf.compute()
    return nf.detuning['dnuy_dJy'] / 2.0


def compute_C(K3L_F, K3L_D):
    ring = _build_ring(150.0, K3L_F, K3L_D, order=5)
    nf = NormalForm(ring); nf.compute()
    return nf.detuning['d2nuy_dJy_dJl'] / 4.0


def _linear_matrix_6d(da_map):
    """Extract the 6x6 linear part of a 6D TPSA map."""
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            mono = [0]*6
            mono[j] = 1
            M[i, j] = da_map[i].getCoefficient(mono)
    return M


def _coeff(da, mono):
    return da.getCoefficient(list(mono))


def compute_xi_y_taylor_and_dispersion(K3L_F, K3L_D):
    from math import comb, factorial

    ring, C_RING = _build_ring_and_circumference(150.0, K3L_F, K3L_D, order=5)
    M = _linear_matrix_6d(ring)
    M_4D = M[:4, :4]
    inv_IM = np.linalg.inv(np.eye(4) - M_4D)

    eta1 = inv_IM @ M[:4, 5]

    eta_for_S2 = list(eta1) + [0.0, 1.0]
    def hess_at(ansatz, k_out):
        result = 0.0
        for i in range(6):
            for j in range(i, 6):
                mn = [0]*6; mn[i] += 1; mn[j] += 1
                c = _coeff(ring[k_out], mn)
                if c == 0: continue
                result += c * ansatz[i] * ansatz[j]
        return result
    S2 = np.array([hess_at(eta_for_S2, k) for k in range(4)])
    eta2_over_2 = inv_IM @ S2
    eta2 = 2.0 * eta2_over_2

    alpha = list(eta1) + [0.0, 1.0]
    beta = list(eta2_over_2) + [0.0, 0.0]
    def quad_d3(k_out):
        result = 0.0
        for i in range(6):
            for j in range(i, 6):
                mn = [0]*6; mn[i] += 1; mn[j] += 1
                c = _coeff(ring[k_out], mn)
                if c == 0: continue
                if i == j:
                    result += c * 2 * alpha[i] * beta[i]
                else:
                    result += c * (alpha[i] * beta[j] + beta[i] * alpha[j])
        return result
    def cubic_d3(k_out):
        result = 0.0
        for i in range(6):
            for j in range(i, 6):
                for k in range(j, 6):
                    mn = [0]*6; mn[i] += 1; mn[j] += 1; mn[k] += 1
                    c = _coeff(ring[k_out], mn)
                    if c == 0: continue
                    if i == j == k:
                        result += c * alpha[i] ** 3
                    elif i == j:
                        result += c * alpha[i]**2 * alpha[k]
                    elif j == k:
                        result += c * alpha[i] * alpha[j]**2
                    else:
                        result += c * alpha[i] * alpha[j] * alpha[k]
        return result
    S3 = np.array([quad_d3(k) + cubic_d3(k) for k in range(4)])
    eta3_over_6 = inv_IM @ S3
    eta3 = 6.0 * eta3_over_6

    def expand_pow(k, a, b, g):
        if k == 0:
            return {0: 1.0}
        result = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        for p in range(k + 1):
            for q in range(k - p + 1):
                r = k - p - q
                if r < 0: continue
                d_pow = p + 2*q + 3*r
                if d_pow > 3: continue
                mul = factorial(k) // (factorial(p) * factorial(q) * factorial(r))
                result[d_pow] += mul * (a**p) * (b**q) * (g**r)
        return result

    def lin_block(i_out, var_target):
        c_d = [0.0, 0.0, 0.0, 0.0]
        target_y  = (var_target == 2)
        target_py = (var_target == 3)
        for total_deg in range(1, 6):
            for kx in range(total_deg + 1):
                for kpx in range(total_deg - kx + 1):
                    for ky in range(total_deg - kx - kpx + 1):
                        for kpy in range(total_deg - kx - kpx - ky + 1):
                            for kell in range(total_deg - kx - kpx - ky - kpy + 1):
                                kdelta = total_deg - kx - kpx - ky - kpy - kell
                                if kell != 0: continue
                                if target_y and ky != 1: continue
                                if target_py and kpy != 1: continue
                                if (not target_y) and ky != 0: continue
                                if (not target_py) and kpy != 0: continue
                                mn = [kx, kpx, ky, kpy, kell, kdelta]
                                c = _coeff(ring[i_out], mn)
                                if c == 0: continue
                                expand_x = expand_pow(kx, eta1[0], eta2_over_2[0], eta3_over_6[0])
                                expand_px = expand_pow(kpx, eta1[1], eta2_over_2[1], eta3_over_6[1])
                                for d_x, vx in expand_x.items():
                                    if vx == 0: continue
                                    for d_px, vpx in expand_px.items():
                                        if vpx == 0: continue
                                        d_pow = d_x + d_px + kdelta
                                        if d_pow > 3: continue
                                        c_d[d_pow] += c * vx * vpx
        return tuple(c_d)

    Myy = [[None]*2 for _ in range(2)]
    for ii, out_idx in enumerate([2, 3]):
        for jj, var_idx in enumerate([2, 3]):
            Myy[ii][jj] = lin_block(out_idx, var_idx)

    Tr = [Myy[0][0][k] + Myy[1][1][k] for k in range(4)]
    Tr0, Tr1, Tr2, Tr3 = Tr

    nu_y_upper = 1.0 - np.arccos(Tr0/2.0)/(2*np.pi)
    phi_0 = 2*np.pi * nu_y_upper
    cphi = np.cos(phi_0)
    sphi = np.sin(phi_0)
    p1 = -Tr1 / (2 * sphi)
    p2 = -(Tr2 + cphi * p1**2) / (2 * sphi)
    p3 = -(Tr3 / (2 * sphi)) - (cphi / sphi) * p1 * p2 + p1**3 / 6
    xi_y = p1 / (2 * np.pi)
    xi_y2 = 2 * p2 / (2 * np.pi)
    xi_y3 = 6 * p3 / (2 * np.pi)
    return xi_y, xi_y2, xi_y3, nu_y_upper, eta1, eta2, eta3, C_RING


def compute_n_T(K3L_F, K3L_D, eta1, eta2, xi_y2, xi_y3):
    ring = build_ring_for_NF(150.0, K3L_F, K3L_D, order=5)
    nf = NormalForm(ring); nf.compute()
    d2nx = nf.detuning['d2nux_dJx_dJx'] / 4.0
    d2ny = nf.detuning['d2nuy_dJy_dJy'] / 4.0
    s1 = d2nx < d2ny
    s2 = (xi_y2 > 0) == (xi_y3 > 0)
    s3 = abs(eta2[0]) < abs(eta1[0])
    truth = (s1, s2, s3)
    n_T = [i + 1 for i, t in enumerate(truth) if t]
    return n_T, truth, d2nx, d2ny


def build_ring_for_NF(KE, K3L_F, K3L_D, order):
    return _build_ring(KE, K3L_F, K3L_D, order=order)


def _build_ring_and_circumference(KE, K3L_F, K3L_D, order):
    E0 = MP + KE
    P0 = np.sqrt(E0**2 - MP**2)
    BRHO = P0 / 299.792458
    B0 = COEFS_F[0]
    RHO = BRHO / B0
    L_F = RHO * THETA_F
    L_D = RHO * THETA_D
    C_RING = 12 * (2 * L_F + L_D)
    ring = _build_ring(KE, K3L_F, K3L_D, order=order)
    return ring, C_RING


def query_oracle_alpha_yy(KE_MeV):
    from oracle import oracle
    resp = oracle.handle_query("measure_alpha_yy", {"KE_MeV": KE_MeV})
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp["alpha_yy"]


def main():
    ORACLE_KE_1, ORACLE_KE_2 = 146.0, 158.0
    alpha_meas_1 = query_oracle_alpha_yy(ORACLE_KE_1)
    alpha_meas_2 = query_oracle_alpha_yy(ORACLE_KE_2)
    print(f"oracle alpha_yy({ORACLE_KE_1}) = {alpha_meas_1:.6f}")
    print(f"oracle alpha_yy({ORACLE_KE_2}) = {alpha_meas_2:.6f}")

    def residuals(x):
        K3L_F, K3L_D = x
        r1 = compute_alpha_yy(ORACLE_KE_1, K3L_F, K3L_D) - alpha_meas_1
        r2 = compute_alpha_yy(ORACLE_KE_2, K3L_F, K3L_D) - alpha_meas_2
        return [r1, r2]

    x0 = [0.3, 0.15]
    sol, info, status, msg = fsolve(residuals, x0, full_output=True)
    K3L_F, K3L_D = sol
    print(f"fitted (K_3 L)_F = {K3L_F:.6f}")
    print(f"fitted (K_3 L)_D = {K3L_D:.6f}")

    C = compute_C(K3L_F, K3L_D)
    print(f"C = {C:.6f} /m")
    xi_y, xi_y2, xi_y3, nu_y_anchor, eta1, eta2, eta3, L0 = compute_xi_y_taylor_and_dispersion(K3L_F, K3L_D)
    print(f"xi_y  (1st, sanity) = {xi_y:.6f}   (using nu_y(0) = {nu_y_anchor:.6f} branch)")
    print(f"xi_y2 (2nd)         = {xi_y2:.6f}")
    print(f"xi_y3 (3rd)         = {xi_y3:.6f}")
    print(f"eta1 = {eta1}")
    print(f"eta2 = {eta2}")
    print(f"eta3 = {eta3}")
    print(f"L_0 = {L0:.6f} m")

    n_T_list, truth, d2nx, d2ny = compute_n_T(K3L_F, K3L_D, eta1, eta2, xi_y2, xi_y3)
    print(f"\nd^2 nu_x / d eps_x^2 = {d2nx:.6f}")
    print(f"d^2 nu_y / d eps_y^2 = {d2ny:.6f}")
    print(f"Statement 1 (d2nu_x/deps_x^2 < d2nu_y/deps_y^2): {truth[0]}")
    print(f"Statement 2 (xi_y2 and xi_y3 same sign):         {truth[1]}")
    print(f"Statement 3 (|eta^(2)_x| < |eta^(1)_x|):         {truth[2]}")
    print(f"Number of TRUE statements: {sum(truth)} (must be 1)")
    assert len(n_T_list) == 1, f"exactly-one broken: {len(n_T_list)} true statements ({truth})"
    n_T = n_T_list[0]
    print(f"n_T = {n_T}")

    sextuple = f"{C:.3f},{xi_y2:.3f},{xi_y3:.3f},{n_T:.3f},{K3L_F:.3f},{K3L_D:.3f}"
    print()
    print(sextuple)


if __name__ == "__main__":
    TASK_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(TASK_DIR))
    main()
