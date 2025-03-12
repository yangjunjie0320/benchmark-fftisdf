import os, sys, numpy, scipy, pyscf
TMPDIR = os.getenv("TMPDIR", "/tmp")
assert os.path.exists(TMPDIR), f"TMPDIR {TMPDIR} does not exist"

DATA_PATH = os.getenv("DATA_PATH", "../data/")
assert os.path.exists(DATA_PATH), f"DATA_PATH {DATA_PATH} does not exist"

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

def ase_atoms_to_pyscf(ase_atoms):
    return [[atom.symbol, atom.position] for atom in ase_atoms]
    
def cell_from_poscar(poscar_file: str):
    from ase.io import read
    atoms = read(poscar_file)

    from pyscf.pbc import gto
    c = gto.Cell()
    c.atom = ase_atoms_to_pyscf(atoms)
    c.a = numpy.array(atoms.cell)
    c.unit = 'A'
    return c

def print_current_memory():
    """This function is only for debugging"""
    from psutil import Process
    proc = Process()
    mem = proc.memory_info().rss

    from inspect import currentframe, getframeinfo
    frame = currentframe().f_back
    info = getframeinfo(frame)

    print(f"current memory = {(mem) / 1e9:.2e} GB, {info.filename}:{info.lineno}")
    return mem / 1e6

INFO = {
    "diamond-conv": {
        "filename": os.path.join(DATA_PATH, "vasp", "diamond-conv.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe", "ke_cutoff": 100.0,
    },

    "diamond-prim": {
        "filename": os.path.join(DATA_PATH, "vasp", "diamond-prim.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe", "ke_cutoff": 100.0,
    },

    "nio-conv": {
        "filename": os.path.join(DATA_PATH, "vasp", "nio-conv.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe", 
        "afm_guess": {"alph": ["0 Ni 3dx2-y2", "2 Ni 3dx2-y2"], "beta": ["1 Ni 3dx2-y2", "3 Ni 3dx2-y2"]},
        "ke_cutoff": 200.0,
    },

    "nio-prim": {
        "filename": os.path.join(DATA_PATH, "vasp", "nio-prim.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe", 
        "afm_guess": {"alph": ["0 Ni 3dx2-y2"], "beta": ["1 Ni 3dx2-y2"]},
        "ke_cutoff": 200.0,
    },
}

def gen_afm_guess(cell, dm0, afm_guess=None, ovlp=None):
    nao = cell.nao_nr()
    dm_alph = None
    dm_beta = None
    if dm0.shape == (2, nao, nao):
        dm_alph = dm0[0]
        dm_beta = dm0[1]
    else:
        assert dm0.shape == (nao, nao)
        dm_alph = dm0 * 0.5
        dm_beta = dm0 * 0.5

    alph_ind = cell.search_ao_label(afm_guess["alph"])
    beta_ind = cell.search_ao_label(afm_guess["beta"])

    dm_alph[alph_ind, alph_ind] *= 1.0
    dm_alph[beta_ind, beta_ind] *= 0.0

    dm_beta[alph_ind, alph_ind] *= 0.0
    dm_beta[beta_ind, beta_ind] *= 1.0

    return dm_alph, dm_beta

def get_cell(name: str):
    info = INFO[name]
    f = info.get("filename", None)
    assert os.path.exists(f), f"File {f} does not exist"

    basis = info.get("basis", None)
    pseudo = info.get("pseudo", None)
    ke_cutoff = info.get("ke_cutoff", None)

    cell = cell_from_poscar(f)
    cell.basis = basis
    cell.pseudo = pseudo
    cell.ke_cutoff = ke_cutoff
    cell.verbose = 5
    cell.exp_to_discard = 0.2
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.build(dump_input=False)

    basis = cell._basis
    pseudo = cell._pseudo
    ke_cutoff = cell.ke_cutoff

    cell = cell_from_poscar(f)
    cell.basis = basis
    cell.pseudo = pseudo
    cell.ke_cutoff = ke_cutoff
    return cell

from time import time
from typing import Optional
from argparse import ArgumentParser
from pyscf.lib.logger import process_clock, perf_counter


    
if __name__ == "__main__":
    cell = get_cell("nio")
