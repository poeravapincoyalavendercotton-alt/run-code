from __future__ import annotations

import os
import re
import math
import shutil
import sys
import tempfile


# ============================================================
# REVEAL_SEARCH_SPACE flag
#   False --> handle_query("help") returns only output key names
#           and descriptions.  The AI must solve unconstrained
#           (invert the four formulae algebraically).
#   True  --> handle_query("help") additionally lists all K known
#           DRAM configurations (org presets with R, G, B, N).
#           The AI can then enumerate candidates and match by
#           residual - a constrained search strategy.
# ============================================================
REVEAL_SEARCH_SPACE = False
_DUAL_RANK = True
# Rank count in Ramulator and in r_G/r_B formulas must match (same ``R`` in __INPUTS_KV).
_ORG_RANK = 2 if _DUAL_RANK else 1
ORG_RANK = _ORG_RANK


# -- Ramulator import --------------------------------------------------------
# 1) Plain ``import ramulator`` - works when the image installed it (pip) or
#    the process was started with PYTHONPATH including ``<ramulator2>/python``.
# 2) Walk up from this file for ``<ancestor>/ramulator2/python`` (stem-software layout).
# 3) Optional override: RAMULATOR2_PYTHON_PATH=directory_that_contains_package_ramulator

_ramulator_mod = None


def _import_ramulator():
    global _ramulator_mod
    if _ramulator_mod is not None:
        return _ramulator_mod

    try:
        import ramulator  # type: ignore
    except ImportError:
        pass
    else:
        _ramulator_mod = ramulator
        return ramulator

    extra: list[str] = []
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(12):
        guess = os.path.join(here, "ramulator2", "python")
        if os.path.isdir(os.path.join(guess, "ramulator")):
            extra.append(guess)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent

    loc = os.environ.get("RAMULATOR2_PYTHON_PATH")
    if loc:
        p = os.path.abspath(os.path.expanduser(loc.strip()))
        if os.path.isdir(os.path.join(p, "ramulator")):
            extra.append(p)

    for d in extra:
        sys.path.insert(0, d)

    try:
        import ramulator  # type: ignore
    except ImportError as err:
        raise ImportError(
            "Cannot import ``ramulator``. Use a Python where it is installed, set "
            "PYTHONPATH to `<ramulator2>/python`, or set RAMULATOR2_PYTHON_PATH to "
            "that directory."
        ) from err
    _ramulator_mod = ramulator
    return ramulator


# -- Simulation knobs --------------------------------------------------------

_SIM_TIMING_PRESET: dict = {
    "DDR3": "DDR3_800D",
    "DDR4": "DDR4_2400R",
    "DDR5": "DDR5_3200C",
}
_SIM_TCK_PS: dict = {
    "DDR3": 2500,
    "DDR4": 833,
    "DDR5": 625,
}
_TREFW_PS: dict = {
    "DDR3": 64_000_000_000,
    "DDR4": 64_000_000_000,
    "DDR5": 32_000_000_000,
}
_TREFW_CYCLES_NUM: dict = {
    "DDR3": _TREFW_PS["DDR3"] / _SIM_TCK_PS["DDR3"],
    "DDR4": _TREFW_PS["DDR4"] / _SIM_TCK_PS["DDR4"],
    "DDR5": _TREFW_PS["DDR5"] / _SIM_TCK_PS["DDR5"],
}
_SIM_NUM_REQUESTS: int = 1_000_000
_SIM_CLOCK_RATIO: int = 1                # memory-system / frontend clock ratio


def jedec_density_mb_from_dtype(dtype: str) -> int:
    """Chip density in Mb from the JEDEC preset label (``..._8Gb_...`` -> 8192)."""
    m = re.search(r"_(\d+)Gb_", dtype)
    if not m:
        raise ValueError(f"Cannot parse JEDEC density from dtype: {dtype!r}")
    return int(m.group(1)) * 1024


def _controller_cycles_from_stats(stats: object) -> int:
    """Return positive DRAM-controller ``cycles`` from Ramulator ``get_stats()`` dict."""

    def from_block(block: object) -> int:
        if not isinstance(block, dict):
            return 0
        v = block.get("cycles")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            i = int(v)
            return i if i > 0 else 0
        return 0

    if not isinstance(stats, dict):
        return 0
    ms = stats.get("memory_system")
    if not isinstance(ms, dict):
        return 0

    ctrl = ms.get("controller")
    if isinstance(ctrl, dict):
        c = from_block(ctrl)
        if c:
            return c
    if isinstance(ctrl, list):
        best = max((from_block(x) for x in ctrl), default=0)
        if best:
            return best

    found: list[int] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "cycles" in node:
                v = node["cycles"]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    i = int(v)
                    if i > 0:
                        found.append(i)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(ms)
    return max(found) if found else 0


class StatTracker:
    # Represents the search space (DRAM organisation).  The DRAM configuration
    # is organised into:
    #   ranks R, bankgroups per rank G, banks per bankgroup B, rows per bank N
    # where total rows K_rows = R * G * B * N.
    # NOTE: concrete implementation repurposed from
    #       ``ramulator2/python/ramulator/dram/ddr4.py``
    # ``D`` = JEDEC DDR4 device density in Mb, matching ``ramulator.dram.ddr4.DDR4.org_presets``.
    __INPUTS_KV = {
        # NOTE: DDR3
        "DDR3_1Gb_x4":  {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 14, "C": 1 << 11, "DQ": 4},
        "DDR3_1Gb_x16": {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 13, "C": 1 << 10, "DQ": 16},
        "DDR3_2Gb_x8":  {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 15, "C": 1 << 11, "DQ": 8},
        "DDR3_2Gb_x16": {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 14, "C": 1 << 10, "DQ": 16},
        "DDR3_4Gb_x4":  {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 16, "C": 1 << 11, "DQ": 4},
        "DDR3_4Gb_x16": {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 15, "C": 1 << 10, "DQ": 16},
        "DDR3_8Gb_x8":  {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 17, "C": 1 << 10, "DQ": 8},
        "DDR3_8Gb_x16": {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 16, "C": 1 << 10, "DQ": 16},

        # NOTE: DDR4
        "DDR4_2Gb_x4": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 15, "C": 1 << 10, "DQ": 4},
        "DDR4_2Gb_x8": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 14, "C": 1 << 10, "DQ": 8},
        "DDR4_2Gb_x16": {"R": _ORG_RANK, "G": 2, "B": 4, "N": 1 << 14, "C": 1 << 10, "DQ": 16},
        "DDR4_4Gb_x4": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 16, "C": 1 << 10, "DQ": 4},
        "DDR4_4Gb_x8": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 15, "C": 1 << 10, "DQ": 8},
        "DDR4_4Gb_x16": {"R": _ORG_RANK, "G": 2, "B": 4, "N": 1 << 15, "C": 1 << 10, "DQ": 16},
        "DDR4_8Gb_x4": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 17, "C": 1 << 10, "DQ": 4},
        "DDR4_8Gb_x8": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 16, "C": 1 << 10, "DQ": 8},
        "DDR4_8Gb_x16": {"R": _ORG_RANK, "G": 2, "B": 4, "N": 1 << 16, "C": 1 << 10, "DQ": 16},
        "DDR4_16Gb_x4": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 18, "C": 1 << 10, "DQ": 4},
        "DDR4_16Gb_x8": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 17, "C": 1 << 10, "DQ": 8},
        "DDR4_16Gb_x16": {"R": _ORG_RANK, "G": 2, "B": 4, "N": 1 << 17, "C": 1 << 10, "DQ": 16},

        # NOTE: DDR5
        "DDR5_8Gb_x8":   {"R": _ORG_RANK, "G": 8, "B": 2, "N": 1 << 16, "C": 1 << 10, "DQ": 8},
        "DDR5_8Gb_x16":  {"R": _ORG_RANK, "G": 4, "B": 2, "N": 1 << 16, "C": 1 << 10, "DQ": 16},
        "DDR5_16Gb_x4":  {"R": _ORG_RANK, "G": 8, "B": 4, "N": 1 << 16, "C": 1 << 11, "DQ": 4},
        "DDR5_16Gb_x16": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 16, "C": 1 << 10, "DQ": 16},
        "DDR5_32Gb_x4":  {"R": _ORG_RANK, "G": 8, "B": 4, "N": 1 << 17, "C": 1 << 11, "DQ": 4},
        "DDR5_32Gb_x16": {"R": _ORG_RANK, "G": 4, "B": 4, "N": 1 << 17, "C": 1 << 10, "DQ": 16},
    }

    #   N_refs = measured REFab cmd count
    #   r_N  = N_refs / N        (rows **per bank**, not R*G*B*N because all-bank REF)
    #   r_G  = N_refs / (R*G)
    #   r_B  = N_refs / (R*G*B)
    # ``N_cycles`` is elapsed simulated time in cycles.
    _OUTPUT_KEYS = ("r_N", "N_refs", "f_refs", "r_G", "r_B")

    def __init__(self, dtype: str = None) -> None:
        self._outputs: dict = {k: 0.0 for k in self._OUTPUT_KEYS}

        if not dtype:
            raise ValueError(f"{type(self).__name__}: 'dtype' is required")

        if dtype not in self.__INPUTS_KV:
            raise ValueError(f"{type(self).__name__}: invalid parameter 'dtype'")

        if not self.__parse_outputs(dtype):
            self.__execute_simulation(dtype)

    def _resolve_tRFC(self, dtype: str, density: int, tCK_ps: int) -> int:
        # Repurposed from ramulator2/python/ramulator/dram/ddr4.py
        if "DDR3" in dtype:
            if density <= 1024: tRFC_ns = 110
            elif density <= 2048: tRFC_ns = 160
            elif density <= 4096: tRFC_ns = 260
            elif density <= 8192: tRFC_ns = 350
            else: return -1
            return math.ceil(tRFC_ns * 1000 / tCK_ps)
        elif "DDR4" in dtype:
            if density <= 2048: tRFC_ns = 160
            elif density <= 4096: tRFC_ns = 260
            elif density <= 8192: tRFC_ns = 360
            elif density <= 16384: tRFC_ns = 550
            else: return -1
            return math.ceil(tRFC_ns * 1000 / tCK_ps)
        elif "DDR5" in dtype:
            if density <= 8192: tRFC_ns = 195
            elif density <= 16384: tRFC_ns = 295
            elif density <= 32768: tRFC_ns = 410
            else: return -1
            return math.ceil(tRFC_ns * 1000 / tCK_ps)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")


    def _populate_from_refab_measured(self, dtype: str, refab: int) -> None:
        """Set r_N, N_refs, f_refs, r_G, r_B from counted REFab and ground-truth org."""
        if refab <= 0:
            raise ValueError("refab must be positive")

        p = self.__INPUTS_KV[dtype]
        R, G, B, N = p["R"], p["G"], p["B"], p["N"]
        D = jedec_density_mb_from_dtype(dtype)

        # N_refs is Ramulator all-bank refresh command count for given # of input requests.
        n_refs_f = float(refab)
        self._outputs["N_refs"] = n_refs_f
        self._outputs["r_N"]  = n_refs_f / float(N)
        self._outputs["r_G"]  = n_refs_f / float(R * G)
        self._outputs["r_B"]  = n_refs_f / float(R * G * B)

        total_cycles = int(self._outputs["_N_cycles"])
        if "DDR3" in dtype:
            tCK_ps = _SIM_TCK_PS["DDR3"]
        elif "DDR4" in dtype:
            tCK_ps = _SIM_TCK_PS["DDR4"]
        elif "DDR5" in dtype:
            tCK_ps = _SIM_TCK_PS["DDR5"]
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
        tRFC = self._resolve_tRFC(dtype, D, tCK_ps)
        # problem.md: f_refs = (N_refs * tRFC) / N_cycles  (tRFC in DRAM cycles)
        self._outputs["f_refs"] = (n_refs_f * float(tRFC)) / float(total_cycles)

    # -- Output file helpers ----------------------------------------------

    def __parse_outputs(self, dtype: str = None) -> bool:
        """Check for persistent cached output files.

        Returns False unconditionally - output files are ephemeral (generated
        in a tmp folder and cleaned up after each simulation run).  Override
        this method to enable a persistent cache if desired.
        """
        return False

    def __parse_cmd_count(self, path: str) -> dict:
        """Parse a CommandCounter CSV file into {cmd_name: count} dict."""
        counts: dict = {}
        with open(path, "r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    counts[parts[0].strip()] = int(parts[1].strip())
        return counts

    # -- Layout helper (mirrors ramulator2/tests/utils.py) -----------------

    def __extract_layout(self, dram_obj) -> dict:
        """Extract LatencyThroughputTrace layout parameters from a DDR4 object."""
        cls = type(dram_obj)
        level_names = list(cls.levels.keys())
        org_dict, _ = dram_obj.resolve()
        org_counts = [org_dict.get(name.lower(), 1) for name in level_names]

        row_idx = level_names.index("Row")
        col_idx = level_names.index("Column")

        bank_positions = list(range(1, row_idx))
        bank_counts = [org_counts[i] for i in bank_positions]

        # BankGroup cycles fastest --> preferential nCCDS (inter-BG) timing.
        if "BankGroup" in level_names:
            bg_idx = level_names.index("BankGroup") - 1  # offset by Channel
            if bg_idx < len(bank_positions) - 1:
                pos = bank_positions.pop(bg_idx)
                cnt = bank_counts.pop(bg_idx)
                bank_positions.append(pos)
                bank_counts.append(cnt)

        total_bank_units = math.prod(bank_counts)
        num_cols = org_counts[col_idx]
        ips = cls.internal_prefetch_size

        return {
            "addr_vec_size":          len(level_names),
            "bank_positions":         bank_positions,
            "bank_counts":            bank_counts,
            "total_bank_units":       total_bank_units,
            "row_pos":                row_idx,
            "col_pos":                col_idx,
            "num_rows":               org_counts[row_idx],
            "num_cols":               num_cols,
            "internal_prefetch_size": ips,
            "num_cls":                num_cols // ips,
        }

    # -- Simulation --------------------------------------------------------

    def __execute_simulation(self, dtype: str) -> None:
        """Run a Ramulator 2.1 simulation for *dtype* without CPU traces."""
        ramulator = _import_ramulator()

        # Create tmp dir inside the task (project) folder.
        # oracle.py lives in  <task>/oracle/  -->  parent is <task>.
        task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tmp_dir = tempfile.mkdtemp(dir=task_dir, prefix="_ram_tmp_")
        cmd_count_path = os.path.join(tmp_dir, "cmd.count")
        stats_yaml_path = os.path.join(tmp_dir, "stats.yaml")

        try:
            p_org = self.__INPUTS_KV[dtype]
            dram_obj = None

            # Check if the dtype is DDR3 or DDR4 or DDR5
            if "DDR3" in dtype:
                dram_obj = ramulator.dram.DDR3(
                    org_preset=dtype,
                    timing_preset=_SIM_TIMING_PRESET["DDR3"],
                    rank=p_org["R"],
                )
                layout = self.__extract_layout(dram_obj)
            elif "DDR4" in dtype:
                dram_obj = ramulator.dram.DDR4(
                    org_preset=dtype,
                    timing_preset=_SIM_TIMING_PRESET["DDR4"],
                    rank=p_org["R"],
                )
                layout = self.__extract_layout(dram_obj)
            elif "DDR5" in dtype:
                dram_obj = ramulator.dram.DDR5(
                    org_preset=dtype,
                    timing_preset=_SIM_TIMING_PRESET["DDR5"],
                    rank=p_org["R"],
                )
                layout = self.__extract_layout(dram_obj)
            else:
                raise ValueError(f"Invalid dtype: {dtype}")


            # -- Controller ----------------------------------------------
            # Basic GenericDDR with all-bank periodic refresh (no RFM, no PRAC).
            ctrl = ramulator.controller.GenericDDR(
                dram=dram_obj,
                scheduler=ramulator.scheduler.FRFCFS(),
                refresh_manager=ramulator.refresh_manager.AllBank(scope="Rank"),
                row_policy=ramulator.row_policy.Open(),
                addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
                controller_plugins=[
                    ramulator.controller_plugin.CommandCounter(
                        commands_to_count=["REFab"],
                        path=cmd_count_path,
                    ),
                ],
            )

            # -- Frontend - streaming only, no CPU traces ----------------
            frontend = ramulator.frontend.LatencyThroughputTrace(
                clock_ratio=_SIM_CLOCK_RATIO,
                nop_counter=1,
                num_probe_requests=0,
                streaming_only=True,
                num_streaming_requests=_SIM_NUM_REQUESTS,
                warmup_cycles=0,
                **layout,
            )

            # -- Memory system -------------------------------------------
            mem = ramulator.memory_system.GenericDRAM(
                clock_ratio=_SIM_CLOCK_RATIO,
                controllers=[ctrl],
                channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
            )

            # -- Run -----------------------------------------------------
            sim = ramulator.Simulation(frontend, mem)
            sim.run()
            stats_yaml_text = sim.stats_yaml
            stats = sim.stats

            with open(stats_yaml_path, "w") as fh:
                fh.write(stats_yaml_text)

            if not os.path.isfile(cmd_count_path):
                raise RuntimeError(f"Missing CommandCounter file after simulation: {cmd_count_path}")
            cmd_counts = self.__parse_cmd_count(cmd_count_path)
            refab_sim = cmd_counts.get("REFab", 0)

            actual_cycles = _controller_cycles_from_stats(stats)

            if actual_cycles <= 0:
                raise RuntimeError(
                    f"No DRAM-controller cycles reported in Ramulator stats (got {actual_cycles})."
                )
            if refab_sim <= 0:
                raise RuntimeError(
                    f"No all-bank refresh commands counted (got {refab_sim}). "
                    "Check CommandCounter wiring."
                )

            if "DDR3" in dtype:
                t_ref = _TREFW_CYCLES_NUM["DDR3"]
            elif "DDR4" in dtype:
                t_ref = _TREFW_CYCLES_NUM["DDR4"]
            elif "DDR5" in dtype:
                t_ref = _TREFW_CYCLES_NUM["DDR5"]
            else:
                raise ValueError(f"Invalid dtype: {dtype}")

            span_note = ""
            if actual_cycles < t_ref:
                span_note = f" (dram time < one tREFW={t_ref:.0f}c)"
            print(
                f"[oracle] Ramulator done."
            )

            self._outputs["_N_cycles"] = int(actual_cycles)
            self._populate_from_refab_measured(dtype, refab_sim)

        except RuntimeError:
            raise
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Ramulator simulation failed: {exc}") from exc

        finally:
            # Always remove the tmp directory and its contents.
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- Public API --------------------------------------------------------

    def lookup_input(self, dtype: str, key: str) -> int:
        """Retrieve an organisation value (R, G, B, N, C, DQ, or D) for *dtype*."""
        if dtype not in self.__INPUTS_KV:
            raise ValueError(f"{type(self).__name__}: invalid parameter 'dtype'")
        org = self.__INPUTS_KV[dtype]
        if key == "D":
            return jedec_density_mb_from_dtype(dtype)
        allowed = ("R", "G", "B", "N", "C", "DQ")
        if key not in allowed or key not in org:
            raise ValueError(f"{type(self).__name__}: invalid parameter 'key'")
        return math.floor(org[key])

    def lookup_output(self, key: str) -> int | float:
        """Retrieve a simulation output value (r_N, N_refs, r_G, or r_B).

        ``r_N`` and ``f_refs`` are **floats**; ``N_refs``, ``r_G``, and ``r_B`` are floored.
        """
        if key not in self._outputs:
            raise ValueError(f"{type(self).__name__}: invalid parameter 'key'")
        v = self._outputs[key]
        if key in ("r_N", "f_refs"):
            return float(v)
        return math.floor(v)


def organisation_preset_record(dtype: str) -> dict[str, int]:
    """Return organisation fields for *dtype*, including computed ``D`` (Mb)."""
    if dtype not in StatTracker._StatTracker__INPUTS_KV:
        raise ValueError(f"Unknown preset dtype: {dtype!r}")
    p = dict(StatTracker._StatTracker__INPUTS_KV[dtype])
    p["D"] = jedec_density_mb_from_dtype(dtype)
    return p


def presets_matching_org(R: int, G: int, B: int, N: int) -> list[tuple[str, dict[str, int]]]:
    """All table presets with this ``(R, G, B, N)``, each dict including ``D``."""
    hits: list[tuple[str, dict[str, int]]] = []
    for name, p in StatTracker._StatTracker__INPUTS_KV.items():
        if p["R"] == R and p["G"] == G and p["B"] == B and p["N"] == N:
            q = dict(p)
            q["D"] = jedec_density_mb_from_dtype(name)
            hits.append((name, q))
    return hits


# -- Oracle query handler ----------------------------------------------------

def handle_query(mode: str = "simulation", parameters: dict = None) -> dict:
    if mode == "help":
        response = {
            "modes": {
                "simulation": {
                    "returns": {
                        "r_N":    "float - N_refs / N (rows per bank; not floored)",
                        "f_refs": "float - (N_refs * tRFC_cycles) / N_cycles",
                        "r_G":    "int - per-bankgroup refresh count",
                        "r_B":    "int - per-bank refresh count",
                        "N_cycles": "int - elapsed simulated time in cycles",
                    },
                }
            }
        }
        if REVEAL_SEARCH_SPACE:
            response["search_space"] = {
                name: {
                    "R": p["R"],
                    "G": p["G"],
                    "B": p["B"],
                    "N": p["N"],
                    "C": p["C"],
                    "DQ": p["DQ"],
                    "D": jedec_density_mb_from_dtype(name),
                }
                for name, p in StatTracker._StatTracker__INPUTS_KV.items()
            }
        return response

    if mode == "simulation":
        try:
            # -- Hidden ground-truth DRAM configuration ----------------
            dtype_true = "DDR4_16Gb_x16"
            stat_track = StatTracker(dtype_true)

            r_N = stat_track.lookup_output("r_N")
            N_refs = stat_track.lookup_output("N_refs")
            f_refs = stat_track.lookup_output("f_refs")
            r_G = stat_track.lookup_output("r_G")
            r_B = stat_track.lookup_output("r_B")

            return {
                "r_N": r_N,
                "f_refs": f_refs,
                "r_G": r_G,
                "r_B": r_B,
                "N_cycles": int(stat_track._outputs["_N_cycles"]),
            }
        except (ImportError, RuntimeError, ValueError) as exc:
            return {"error": f"Ramulator oracle: {exc}"}

    return {"error": f"Unknown mode: {mode}"}
