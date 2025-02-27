import os, sys, numpy, scipy, pyscf
TMPDIR = os.getenv("TMPDIR", "/tmp")
assert os.path.exists(TMPDIR), f"TMPDIR {TMPDIR} does not exist"

DATA_PATH = os.getenv("DATA_PATH", "../data/")
assert os.path.exists(DATA_PATH), f"DATA_PATH {DATA_PATH} does not exist"

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
        "basis": 'gth-dzvp-molopt-sr',
        "psuedo": 'gth-pade',
        "ke_cutoff": 400.0,
        "k0": 40.0,
        "c0": 20.0,
    },
    "cco": {
        "filename": os.path.join(DATA_PATH, "vasp", "cco-conv-2x2x1.vasp"),
        "basis": 'gth-dzvp-molopt-sr',
        "psuedo": 'gth-pade',
        "ke_cutoff": 160.0,
        "k0": 80.0,
        "c0": 20.0,
    },
    "nio": {
        "filename": os.path.join(DATA_PATH, "vasp", "nio-conv.vasp"),
        "basis": 'gth-dzvp-molopt-sr',
        "psuedo": 'gth-pade',
        "ke_cutoff": 190.0,
        "k0": 100.0,
        "c0": 20.0,
    },
    "diamond": {
        "filename": os.path.join(DATA_PATH, "vasp", "diamond-conv.vasp"),
        "basis": 'gth-dzvp-molopt-sr',
        "psuedo": 'gth-pade',
        "ke_cutoff": 70.0,
        "k0": 60.0,
        "c0": 10.0,
    },
}

def get_cell(name: str):
    info = INFO[name]

    c0 = cell_from_poscar(info["filename"])
    basis = info["basis"]
    c0.basis = basis
    c0.pseudo = info["psuedo"]
    c0.ke_cutoff = info["ke_cutoff"]
    c0.verbose = 5
    c0.exp_to_discard = 0.2
    c0.build(dump_input=False)

    basis = c0._basis
    pseudo = c0._pseudo
    ke_cutoff = info["ke_cutoff"]

    cell = cell_from_poscar(info["filename"])
    cell.basis = basis
    cell.pseudo = pseudo
    cell.ke_cutoff = ke_cutoff
    return cell

if __name__ == "__main__":
    cell = get_cell("cco")
