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
    "hg1212": {
        "filename": os.path.join(DATA_PATH, "vasp", "hg1212-conv.vasp"),
        "basis": os.path.join(DATA_PATH, "basis", "cc-pvdz.dat"),
        "pseudo": "gth-pbe", "afm_guess": {"alph": ["3 Cu 3dx2-dy2"], "beta": ["4 Cu 3dx2-dy2"]},
    },
    "cco": {
        "filename": os.path.join(DATA_PATH, "vasp", "cco-conv-2x2x1.vasp"),
        "basis": os.path.join(DATA_PATH, "basis", "cc-pvdz.dat"),
        "pseudo": "gth-pbe", "afm_guess": {"alph": ["1 Cu 3dx2-dy2", "13 Cu 3dx2-dy2"], "beta": ["5 Cu 3dx2-dy2", "9 Cu 3dx2-dy2"]},
    },
    "nio": {
        "filename": os.path.join(DATA_PATH, "vasp", "nio-conv.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe", "afm_guess": {"alph": ["0 Ni 3dx2-y2", "2 Ni 3dx2-y2"], "beta": ["1 Ni 3dx2-y2", "3 Ni 3dx2-y2"]},
    },
    "diamond": {
        "filename": os.path.join(DATA_PATH, "vasp", "diamond-conv.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe"
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
    cell.exp_to_discard = 0.1
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

def save_vjk_from_1e_dm0(isdf_obj=None, exxdiv="ewald", log=None):
    cell = isdf_obj.cell.copy(deep=True)

    from pyscf.pbc.scf import RHF
    scf_obj = RHF(cell)
    scf_obj.exxdiv = exxdiv
    scf_obj.verbose = 10
    scf_obj.with_df = isdf_obj
    scf_obj._is_mem_enough = lambda : False

    h1e = scf_obj.get_hcore()
    s1e = scf_obj.get_ovlp()
    e0_mo, c0_mo = scf_obj.eig(h1e, s1e)
    n0_mo = scf_obj.get_occ(e0_mo, c0_mo)
    dm0 = scf_obj.make_rdm1(c0_mo, n0_mo)

    from pyscf.lib import tag_array
    dm0 = tag_array(dm0, mo_coeff=c0_mo, mo_occ=n0_mo)
    # vj, vk = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv=exxdiv, with_k=True, with_j=True)
    # vj = vj.reshape(h1e.shape)
    # vk = vk.reshape(h1e.shape)
    # vjk = vj - 0.5 * vk
    # f1e = h1e + vjk
    # f1e = f1e.reshape(h1e.shape)
    # log.timer("vjk build", *t0)

    t0 = (process_clock(), perf_counter())
    vj = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv=exxdiv, with_k=False, with_j=True)[0]
    log.timer("vj", *t0)

    t0 = (process_clock(), perf_counter())
    vk = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv=exxdiv, with_k=True, with_j=False)[1]
    log.timer("vk", *t0)

    vjk = vj - 0.5 * vk
    f1e = h1e + vjk
    f1e = f1e.reshape(h1e.shape)
    # e_ref = scf_obj.energy_elec(dm=dm0)[0]
    # e_sol = numpy.einsum('ij,ji->', 0.5 * (f1e + h1e), dm0)
    # assert numpy.allclose(e_ref, e_sol), f"e_ref = {e_ref}, e_sol = {e_sol}"
    e_tot = numpy.einsum('ij,ji->', 0.5 * (f1e + h1e), dm0) + cell.energy_nuc()

    chk_path = os.path.join(TMPDIR, f"isdf.chk")
    from pyscf.lib.chkfile import dump
    dump(chk_path, "natm", cell.natm)
    dump(chk_path, "ke_cutoff", cell.ke_cutoff)
    dump(chk_path, "basis", cell._basis)
    dump(chk_path, "pseudo", cell._pseudo)
    
    dump(chk_path, "h1e", h1e)
    dump(chk_path, "s1e", s1e)

    dump(chk_path, "dm0", dm0)
    dump(chk_path, "c0_mo", c0_mo)
    dump(chk_path, "n0_mo", n0_mo)
    dump(chk_path, "e0_mo", e0_mo)

    dump(chk_path, "vj", vj)
    dump(chk_path, "vk", vk)
    dump(chk_path, "vjk", vjk)
    dump(chk_path, "f1e", f1e)
    dump(chk_path, "e_tot", e_tot)
    return e_tot, chk_path
    
if __name__ == "__main__":
    cell = get_cell("nio")
