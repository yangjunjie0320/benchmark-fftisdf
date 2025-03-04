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
        "basis": os.path.join(DATA_PATH, "basis", "cc-pvdz.dat"),
        "pseudo": "gth-pbe"
    },
    "cco": {
        "filename": os.path.join(DATA_PATH, "vasp", "cco-conv-2x2x1.vasp"),
        "basis": os.path.join(DATA_PATH, "basis", "cc-pvdz.dat"),
        "pseudo": "gth-pbe"
    },
    "nio": {
        "filename": os.path.join(DATA_PATH, "vasp", "nio-conv.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe"
    },
    "diamond": {
        "filename": os.path.join(DATA_PATH, "vasp", "diamond-conv.vasp"),
        "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe"
    },
}

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
    cell.build(dump_input=False)

    basis = cell._basis
    pseudo = cell._pseudo
    ke_cutoff = cell.ke_cutoff

    cell = cell_from_poscar(f)
    cell.basis = basis
    cell.pseudo = pseudo
    cell.ke_cutoff = ke_cutoff
    return cell

if __name__ == "__main__":
    cell = get_cell("cco")
