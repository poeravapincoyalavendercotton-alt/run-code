"""
Reference solution -- binding energy of the least-bound electron.

The specimen is a heavy p-block hydride whose HOMO is the heavy atom's ns lone
pair. The prompt fixes the recipe -- PBE0, ano-rcc on the heavy atom + def2-TZVP on
H, the exact reported geometry -- but not the Hamiltonian. ano-rcc is a
relativistically recontracted all-electron basis; used with the default
non-relativistic Hamiltonian the lone-pair orbital energy is wrong by ~1.6 eV, yet
the SCF converges cleanly and looks entirely healthy. The ns shell of a sixth-row
element is strongly stabilised by scalar relativity, so the defensible route applies
the scalar-relativistic X2C Hamiltonian (mf.x2c()). The graded quantity is the
binding energy -epsilon(HOMO) in eV.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pyscf import gto, dft

HARTREE2EV = 27.211386


def solve(query_oracle):
    spec = query_oracle("structure", {})
    atoms = spec["atoms"]
    coords = spec["coordinates_angstrom"]
    atom = [(el, tuple(xyz)) for el, xyz in zip(atoms, coords)]

    # ano-rcc on the heavier element(s), def2-TZVP on hydrogen
    basis = {el: ("def2-tzvp" if el == "H" else "ano-rcc") for el in set(atoms)}

    mol = gto.M(atom=atom, basis=basis,
                charge=spec.get("charge", 0), spin=spec.get("spin", 0), verbose=0)

    mf = dft.RKS(mol)
    mf.xc = "pbe0"
    mf = mf.x2c()                      # scalar-relativistic Hamiltonian (the discipline)
    mf.conv_tol = 1e-10
    mf.kernel()

    mo_e, mo_occ = mf.mo_energy, mf.mo_occ
    homo = mo_e[mo_occ > 0].max()
    return round(-homo * HARTREE2EV, 2)   # binding energy = -epsilon(HOMO), eV


def _load_oracle():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
    try:
        from oracle import handle_query  # sandbox: oracle.py
    except ImportError:
        from setup import handle_query   # local: oracle/setup.py
    return handle_query


def main():
    print(solve(_load_oracle()))


if __name__ == "__main__":
    main()
