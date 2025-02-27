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
        "psuedo": {'Cu': 'GTH-PBE-q19', 'O': 'gth-pbe', 'Hg': 'gth-pbe', 'Ba': 'gth-pbe', 'Ca': 'gth-pbe'},
        "ke_cutoff": 400.0,
        "k0": 40.0,
        "c0": 20.0,
    },
    "cco": {
        "filename": os.path.join(DATA_PATH, "vasp", "cco-conv-2x2x1.vasp"),
        "basis": os.path.join(DATA_PATH, "basis", "cc-pvdz.dat"),
        "psuedo": {'Cu': 'GTH-PBE-q19', 'O': 'gth-pbe', 'Hg': 'gth-pbe', 'Ba': 'gth-pbe', 'Ca': 'gth-pbe'},
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
    # info = INFO[name]

    # cell = cell_from_poscar(info["filename"])
    # basis = info["basis"]
    # cell.basis = basis
    # print(cell.basis)
    # # if os.path.exists(basis):
    # #     print(f"Loading basis from {basis}, natm = {cell.natm}")
    # #     from pyscf.gto.basis.parse_nwchem import load
    # #     basis = {}

    # #     for iatm in range(cell.natm):
    # #         s1 = cell.atom_symbol(iatm)
    # #         s2 = cell.atom_pure_symbol(iatm)
    # #         print(f"iatm = {iatm}, s1 = {s1}, s2 = {s2}")

    # #         if s1 not in basis:
    # #             print(f"Loading basis for {s1}, {s2}")
    # #             basis[s1] = load(basis, s2)

    # #     print(basis)
            
    # cell.pseudo = info["psuedo"]
    # cell.ke_cutoff = info["ke_cutoff"]
    # print(cell)
    # return cell

    chkfile = os.path.join(DATA_PATH, "chk", f"{name}-conv.chk")
    assert os.path.exists(chkfile), f"chkfile {chkfile} does not exist"

    data = {}
    from h5py import File
    f = File(chkfile, "r")
    for k, v in f["cell"].items():
        print(k, v)

    from pyscf.lib.chkfile import load
    atom = load(chkfile, "cell/atom")
    atom = [(s.decode("utf-8"), numpy.array(x)) for s, x in atom]
    lv = load(chkfile, "cell/a")
    basis = load(chkfile, "cell/_basis")
    if isinstance(basis, bytes):
        basis = basis.decode("utf-8")

    pseudo = None
    _pseudo = load(chkfile, "cell/pseudo")
    if isinstance(_pseudo, bytes):
        pseudo = _pseudo.decode("utf-8")
    else:
        assert isinstance(_pseudo, dict)
        pseudo = {}
        for k, v in _pseudo.items():
            # if k[-1].isdigit():
                # pseudo[k[:-1]] = v.decode("utf-8")
            # else:
            pseudo[k] = v.decode("utf-8")
    assert pseudo is not None

    print(basis)
    print(pseudo)
    ke_cutoff = load(chkfile, "cell/ke_cutoff")
    
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = atom
    cell.a = lv
    cell.basis = basis
    cell.pseudo = pseudo
    cell.ke_cutoff = ke_cutoff
    cell.chkfile = chkfile
    return cell