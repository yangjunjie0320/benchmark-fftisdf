import os, sys, numpy, scipy, pyscf
TMPDIR = os.getenv("TMPDIR", "/tmp")
assert os.path.exists(TMPDIR), f"TMPDIR {TMPDIR} does not exist"

DATA_PATH = os.getenv("DATA_PATH", "../data/")
assert os.path.exists(DATA_PATH), f"DATA_PATH {DATA_PATH} does not exist"

def ase_atoms_to_pyscf(ase_atoms):
    return [[atom.symbol, atom.position] for atom in ase_atoms]
    
def cell_from_poscar(poscar_file: str):
    import ase
    from ase.io import read
    atoms = read(poscar_file)

    from pyscf.pbc import gto
    c = gto.Cell()
    c.atom = ase_atoms_to_pyscf(atoms)
    c.a = numpy.array(atoms.cell)
    c.precision = 1e-8
    c.verbose = 0
    c.unit = 'A'
    c.exp_to_discard = 0.1
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
    "cco": {
        "filename": os.path.join(DATA_PATH, "vasp", "cco-conv.vasp"),
        "basis": os.path.join(DATA_PATH, "basis", "cc-pvdz.dat"),
        "psuedo": 'gth-pbe',
        "ke_cutoff": 180.0,
        "k0": 80.0,
        "c0": 20.0,
    },
    "nio": {
        "filename": os.path.join(DATA_PATH, "vasp", "nio-conv.vasp"),
        "basis": 'gth-dzvp-molopt-sr',
        "psuedo": 'gth-pbe',
        "ke_cutoff": 190.0,
        "k0": 100.0,
        "c0": 20.0,
    },
    "diamond": {
        "filename": os.path.join(DATA_PATH, "vasp", "diamond-conv.vasp"),
        "basis": 'gth-dzvp-molopt-sr',
        "psuedo": 'gth-pbe',
        "ke_cutoff": 70.0,
        "k0": 60.0,
        "c0": 10.0,
    },
}

def get_cell(name: str):
    info = INFO[name]
    from utils import cell_from_poscar
    cell = cell_from_poscar(info["filename"])
    cell.basis = info["basis"]
    cell.pseudo = info["psuedo"]
    cell.ke_cutoff = info["ke_cutoff"]
    return cell
