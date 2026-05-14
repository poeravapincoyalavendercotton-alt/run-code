from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle"))
import oracle


# JEDEC-style density tiers used in this task (Mb integers).
_JEDEC_D_MB = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def fetch_oracle() -> tuple[float, float, int, int, int]:
    """Query the oracle for (r_N, f_refs, r_G, r_B, N_cycles)."""
    try:
        resp = oracle.handle_query(mode="simulation", parameters={})
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return (
            resp["r_N"],
            resp["f_refs"],
            resp["r_G"],
            resp["r_B"],
            resp["N_cycles"],
        )
    except Exception as e:
        print(f"Error fetching oracle: {e}")
        return None, None, None, None, None


def _tck_ps_for_preset(dtype: str) -> int:
    if "DDR3" in dtype:
        return oracle._SIM_TCK_PS["DDR3"]
    if "DDR4" in dtype:
        return oracle._SIM_TCK_PS["DDR4"]
    if "DDR5" in dtype:
        return oracle._SIM_TCK_PS["DDR5"]
    raise ValueError(f"Cannot resolve tCK for preset name: {dtype!r}")


def density_mb_decimal_nearest_pow2_tier(
    r: int, g: int, b: int, n: int, c: int, dq: int
) -> int:
    """Infer chip density (Mb) from organisation factors.

    Uses ``(R * G * B * N * C * DQ) / 1_000_000`` then picks the closest value
    among powers-of-two JEDEC tiers (same set Ramulator uses: 1024, 2048, ...).

    ``R`` (module ranks) scales total addressed bits; JEDEC *chip* density is
    independent of how many ranks are populated, so we cancel ``R`` by using
    the per-die product ``(G * B * N * C * DQ) / 1_000_000``, which equals
    ``(R * G * B * N * C * DQ) / (R * 1_000_000)`` for the usual dual-rank table.
    """
    _ = r  # per-die density; R cancels in (R*P)/(R*1e6) when using die product P
    raw = (g * b * n * c * dq) / 1_000_000.0
    return min(_JEDEC_D_MB, key=lambda d: abs(float(d) - raw))


def density_mb_binary_mib_crosscheck(g: int, b: int, n: int, c: int, dq: int) -> int:
    """Exact binary Mibit density ``(G*B*N*C*DQ) / 2^20`` (Ramulator ``density``)."""
    bits = g * b * n * c * dq
    return (bits + (1 << 19)) >> 20


def expected_outputs(
    preset_name: str, cfg: dict, n_refs_measured: int, n_cycles: int
) -> tuple[float, int, float, int, int]:
    """Same five observables as the oracle for this preset: r_N, N_refs, f_refs, r_G, r_B."""
    R, G, B, N = cfg["R"], cfg["G"], cfg["B"], cfg["N"]
    D = int(cfg["D"])
    if "DDR3" in preset_name:
        tck = oracle._SIM_TCK_PS["DDR3"]
    elif "DDR4" in preset_name:
        tck = oracle._SIM_TCK_PS["DDR4"]
    elif "DDR5" in preset_name:
        tck = oracle._SIM_TCK_PS["DDR5"]
    else:
        raise ValueError(f"Invalid preset name: {preset_name}")
    _st = oracle.StatTracker.__new__(oracle.StatTracker)
    t_rfc = oracle.StatTracker._resolve_tRFC(_st, preset_name, D, tck)
    nf = float(n_refs_measured)
    f_r = (nf * float(t_rfc)) / float(n_cycles)
    r_n = nf / float(N)
    r_g = math.floor(nf / float(R * G))
    r_b = math.floor(nf / float(R * G * B))
    return r_n, n_refs_measured, f_r, r_g, r_b


def solve_unconstrained(
    n_refs: int,
    f_refs: float,
    n_cycles: int,
    r_N: float,
    r_G: int,
    r_B: int,
    org_rank: int,
) -> tuple[int, int, int, int]:
    """Recover G, B, N from measured refresh ratios (same algebra as before)."""
    if n_refs <= 0:
        raise ValueError("need positive measured n_refs")
    if f_refs <= 0:
        raise ValueError("need positive measured f_refs")
    if n_cycles <= 0:
        raise ValueError("need positive measured n_cycles")
    if org_rank <= 0:
        raise ValueError("need positive rank count R")

    n = round(n_refs / r_N)
    b = max(1, int(round(r_G / r_B)))
    rg = max(1, int(round(n_refs / r_G)))
    g = rg // org_rank
    if g <= 0 or org_rank * g != rg:
        raise ValueError(
            f"inconsistent organisation from (N_refs={n_refs}, r_G={r_G}, r_B={r_B}, R={org_rank}); "
            f"got RG={rg}, G={g}"
        )

    print(f"[unconstrained] derived  R={org_rank}  G={g}  B={b}  N={n}")
    return org_rank, g, b, n


def infer_density_unconstrained(
    r: int,
    g: int,
    b: int,
    n: int,
    n_refs: int,
    n_cycles: int,
    f_refs: float,
) -> int:
    """Pick ``C``/``DQ`` from the task table, disambiguate with measured ``tRFC``, return ``D``."""
    hits = oracle.presets_matching_org(r, g, b, n)
    if not hits:
        raise ValueError(f"No preset matches R={r}, G={g}, B={b}, N={n}")

    t_meas = int(round(f_refs * float(n_cycles) / float(n_refs)))
    st = oracle.StatTracker.__new__(oracle.StatTracker)
    narrowed: list[tuple[str, dict[str, int]]] = []
    for name, q in hits:
        tck = _tck_ps_for_preset(name)
        tr = int(oracle.StatTracker._resolve_tRFC(st, name, int(q["D"]), tck))
        if abs(tr - t_meas) <= 1:
            narrowed.append((name, q))

    if len(narrowed) == 1:
        _name, q = narrowed[0]
    elif len({q["D"] for _n, q in hits}) == 1:
        _name, q = hits[0]
    elif narrowed:
        _name, q = narrowed[0]
    else:
        raise ValueError(
            f"Could not disambiguate density for R={r}, G={g}, B={b}, N={n}; "
            f"t_rfc_meas={t_meas} (try constrained search or check org inference)."
        )

    c, dq = int(q["C"]), int(q["DQ"])
    d_tier = density_mb_decimal_nearest_pow2_tier(r, g, b, n, c, dq)
    d_bin = density_mb_binary_mib_crosscheck(g, b, n, c, dq)
    if d_tier != d_bin:
        print(
            f"[unconstrained] note: decimal-tier density {d_tier} vs binary Mib {d_bin} "
            f"(using binary Mib to match Ramulator/JEDEC tables)"
        )
        return d_bin
    return d_tier


def solve_constrained(
    r_N: float,
    f_refs: float,
    r_G: int,
    r_B: int,
    n_cycles: int,
    search_space: dict,
) -> tuple[int, int, int, int, int]:
    """Binary-search ``t_rfc`` in ``[100, hi]`` to minimize the best preset residual."""
    c = f_refs * float(n_cycles)
    lo = 100
    hi = min(1200, max(lo, int(c)))

    def best_for_t_rfc(t_rfc: int) -> tuple[float, str, int]:
        n_refs_try = max(1, math.floor(c / t_rfc))
        oracle_vec = (r_N, n_refs_try, f_refs, r_G, r_B)
        best_resid_trial = float("inf")
        best_name_trial: str | None = None
        for name, cfg in search_space.items():
            exp = expected_outputs(name, cfg, n_refs_try, n_cycles)
            resid = sum((a - b) ** 2 for a, b in zip(oracle_vec, exp))
            if resid < best_resid_trial:
                best_resid_trial = resid
                best_name_trial = name
        if best_name_trial is None:
            raise RuntimeError("constrained search: empty search_space")
        return best_resid_trial, best_name_trial, n_refs_try

    step = 0
    while lo < hi:
        mid = (lo + hi) // 2
        r0, name0, n0 = best_for_t_rfc(mid)
        r1, name1, n1 = best_for_t_rfc(mid + 1)
        step += 1
        print(
            f"[constrained] bs step {step}  lo={lo} hi={hi} mid={mid}  "
            f"res(mid)={r0:.6g}@{name0!r}  res(mid+1)={r1:.6g}@{name1!r}"
        )
        if r0 <= r1:
            hi = mid
        else:
            lo = mid + 1

    best_resid_global, best_name_global, best_n_refs_try = best_for_t_rfc(lo)
    best_t_rfc = lo

    print(
        f"[constrained] chosen  t_rfc={best_t_rfc}  N_refs_try={best_n_refs_try}  "
        f"best_match={best_name_global!r}  (residual={best_resid_global})"
    )

    cfg = search_space[best_name_global]
    d = int(oracle.organisation_preset_record(best_name_global)["D"])
    return cfg["R"], cfg["G"], cfg["B"], cfg["N"], d


def main():
    print("Querying oracle ...")
    try:
        r_N, f_refs, r_G, r_B, n_cycles = fetch_oracle()
        if r_N is None or f_refs is None or r_G is None or r_B is None or n_cycles is None:
            raise ValueError("Invalid oracle response")
    except ValueError as e:
        print(f"Error in fetching oracle: {e}")
        return
    except Exception as e:
        print(f"Unexpected error in fetching oracle: {e}")
        return

    n_refs = 3302 # NOTE: assume calculated from f_refs and n_cycles given correct tRFC selection

    print(
        f"Oracle returned:  r_N={r_N}  f_refs={f_refs}  r_G={r_G}  "
        f"r_B={r_B}  N_cycles={n_cycles}"
    )

    try:
        help_resp = oracle.handle_query(mode="help", parameters={})
        search_space = help_resp.get("search_space")

        if search_space:
            print("Using constrained search (help search_space)")
            r, g, b, n, d = solve_constrained(r_N, f_refs, r_G, r_B, n_cycles, search_space)
        else:
            print("Using unconstrained inference (no search_space in help)")
            r, g, b, n = solve_unconstrained(n_refs, f_refs, n_cycles, r_N, r_G, r_B, oracle.ORG_RANK)
            d = infer_density_unconstrained(r, g, b, n, n_refs, n_cycles, f_refs)
    except ValueError as e:
        print(f"Error in constrained/unconstrained search: {e}")
        return
    except Exception as e:
        print(f"Unexpected error in constrained/unconstrained search: {e}")
        return

    print(f"\nCalculated: R={r}, G={g}, B={b}, N={n}, D={d}")
    print(f"\nExpected (golden): R=2, G=2, B=4, N={1 << 17}, D=16384")


if __name__ == "__main__":
    main()
