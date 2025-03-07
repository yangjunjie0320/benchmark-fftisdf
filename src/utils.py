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
def save_vjk_from_1e_dm0(
        isdf_obj=None, config: Optional[ArgumentParser] = None,
        time_table: Optional[dict] = None
    ):
    ke_cutoff = config.ke_cutoff
    
    cell = get_cell(config.name)
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.verbose = 10
    cell.ke_cutoff = ke_cutoff
    cell.build(dump_input=False)

    from pyscf.pbc.scf import RHF
    scf_obj = RHF(cell)
    scf_obj.exxdiv = config.exxdiv
    scf_obj.verbose = 10
    scf_obj._is_mem_enough = lambda : False

    h1e = scf_obj.get_hcore()
    s1e = scf_obj.get_ovlp()
    e0_mo, c0_mo = scf_obj.eig(h1e, s1e)
    n0_mo = scf_obj.get_occ(e0_mo, c0_mo)
    dm0 = scf_obj.make_rdm1(c0_mo, n0_mo)

    t0 = time()
    scf_obj.with_df = isdf_obj
    scf_obj.build()
    time_table["ISDF build"] = time() - t0

    t0 = time()
    vj, vk = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv="ewald")
    vj = vj.reshape(h1e.shape)
    vk = vk.reshape(h1e.shape)
    vjk = vj - 0.5 * vk
    f1e = h1e + vjk
    f1e = f1e.reshape(h1e.shape)
    time_table["VJK build"] = time() - t0

    e_tot = numpy.einsum('ij,ji->', 0.5 * (f1e + h1e), dm0)
    assert e_tot.imag < 1e-10
    e_tot = e_tot.real + cell.energy_nuc()

    chk_path = os.path.join(TMPDIR, f"isdf.chk")
    from pyscf.lib.chkfile import dump
    dump(chk_path, "ke_cutoff", ke_cutoff)
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

    return e_tot
    
if __name__ == "__main__":
    cell = get_cell("nio")
