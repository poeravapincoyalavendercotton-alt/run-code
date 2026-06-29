import json
import os
import subprocess
import sys
from math import gcd
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
SWEEP_CSV = HERE / "sweep.csv"
STEP = 250

CLK_A_PERIOD_PS = int(os.environ.get("CLK_A_PERIOD_PS", "10000"))
CLK_B_PERIOD_PS = int(os.environ.get("CLK_B_PERIOD_PS", "13000"))
N_PULSES = int(os.environ.get("N_PULSES", "32"))
GAP_A = int(os.environ.get("GAP_A", "3"))
SEED = int(os.environ.get("SEED", "1"))


def lcm(a, b):
    return a * b // gcd(a, b)


def run_one_phase(phase_ps):
    env = dict(os.environ)
    env.update({
        "CLK_A_PERIOD_PS": str(CLK_A_PERIOD_PS),
        "CLK_B_PERIOD_PS": str(CLK_B_PERIOD_PS),
        "PHASE_PS": str(phase_ps),
        "N_PULSES": str(N_PULSES),
        "GAP_A": str(GAP_A),
        "SEED": str(SEED),
        "COCOTB_HDL_TIMEUNIT": "1ns",
        "COCOTB_HDL_TIMEPRECISION": "1ps",
    })
    if RESULTS.exists():
        RESULTS.unlink()
    proc = subprocess.run(["make"], cwd=HERE, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        raise RuntimeError("make failed at PHASE_PS=%d" % phase_ps)
    if not RESULTS.exists():
        raise RuntimeError("No results.json after PHASE_PS=%d" % phase_ps)
    data = json.loads(RESULTS.read_text())
    return int(data["answer"]), int(data.get("received", -1)), int(data.get("sent", -1))


def main():
    phi = lcm(CLK_A_PERIOD_PS, CLK_B_PERIOD_PS)
    phases = list(range(0, phi, STEP))
    print("fundamental period: %d ps  (%d phase points at %d ps)" % (phi, len(phases), STEP))
    total = 0
    rows = []
    for i, ph in enumerate(phases, 1):
        corrupted, received, sent = run_one_phase(ph)
        total += corrupted
        rows.append((ph, corrupted, received, sent))
        print("[%3d/%3d] PHASE_PS=%6d  corrupted=%3d  received=%3d"
              % (i, len(phases), ph, corrupted, received))
    SWEEP_CSV.write_text(
        "phase_ps,corrupted,received,sent\n"
        + "\n".join("%d,%d,%d,%d" % r for r in rows) + "\n"
    )
    print("-" * 56)
    print("GOLDEN ANSWER (sum of corrupted over [0,%d)) : %d" % (phi, total))


if __name__ == "__main__":
    main()
