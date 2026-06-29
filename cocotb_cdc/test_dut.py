import json
import os
import random
from collections import Counter
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer

CLK_A_PERIOD_PS = int(os.environ.get("CLK_A_PERIOD_PS", "10000"))
CLK_B_PERIOD_PS = int(os.environ.get("CLK_B_PERIOD_PS", "13000"))
PHASE_PS        = int(os.environ.get("PHASE_PS", "0"))
N_PULSES        = int(os.environ.get("N_PULSES", "32"))
GAP_A           = int(os.environ.get("GAP_A", "3"))
SEED            = int(os.environ.get("SEED", "1"))
RESULTS = Path(__file__).parent / "results.json"


async def start_clocks(dut):
    cocotb.start_soon(Clock(dut.clk_a, CLK_A_PERIOD_PS, units="ps").start())
    if PHASE_PS:
        await Timer(PHASE_PS, units="ps")
    cocotb.start_soon(Clock(dut.clk_b, CLK_B_PERIOD_PS, units="ps").start())


async def reset_dut(dut, cycles=8):
    dut.rst_a.value = 1
    dut.rst_b.value = 1
    dut.data_in.value = 0
    dut.valid_in.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk_b)
    dut.rst_a.value = 0
    dut.rst_b.value = 0
    await RisingEdge(dut.clk_b)


async def driver(dut, payloads):
    for p in payloads:
        await RisingEdge(dut.clk_a)
        dut.data_in.value = p
        dut.valid_in.value = 1
        await RisingEdge(dut.clk_a)
        dut.valid_in.value = 0
        dut.data_in.value = 0
        for _ in range(GAP_A):
            await RisingEdge(dut.clk_a)


async def monitor(dut, received, stop):
    while not stop.is_set():
        await RisingEdge(dut.clk_b)
        await ReadOnly()
        if dut.valid_out.value == 1:
            received.append(int(dut.data_out.value))


@cocotb.test()
async def measure(dut):
    random.seed(SEED)
    await start_clocks(dut)
    await reset_dut(dut)

    payloads = [i + 1 for i in range(N_PULSES)]
    received = []
    stop = cocotb.triggers.Event()

    mon = cocotb.start_soon(monitor(dut, received, stop))
    await driver(dut, payloads)

    for _ in range(2 * N_PULSES + 16):
        await RisingEdge(dut.clk_b)
    stop.set()
    await mon

    counts = Counter(received)
    corrupted = sum(1 for p in payloads if counts.get(p, 0) != 1)
    corrupted += sum(c for p, c in counts.items() if p not in set(payloads))

    RESULTS.write_text(json.dumps({
        "phase_ps": PHASE_PS,
        "answer": corrupted,
        "sent": len(payloads),
        "received": len(received),
    }))
