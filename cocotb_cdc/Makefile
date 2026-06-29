# cocotb Makefile for the CDC stretch/synchronizer DUT.
# Invoked once per phase point by main.py.

SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(PWD)/dut.v

# Top-level RTL module (in dut.v) and the Python test module (test_dut.py)
TOPLEVEL = dut
MODULE   = test_dut

include $(shell cocotb-config --makefiles)/Makefile.sim
