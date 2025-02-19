import os, sys, numpy, scipy, pyscf
TMPDIR = os.getenv("TMPDIR", "/tmp")
assert os.path.exists(TMPDIR), f"TMPDIR {TMPDIR} does not exist"

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
    c.precision = 1e-10
    c.verbose = 0
    c.unit = 'A'
    c.exp_to_discard = 0.1
    return c

def get_jk_time(cell, kmesh=None, df_obj=None, tmp=None, chkfile=None, stdout=None):
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.lib.logger import perf_counter
    from pyscf.lib.logger import process_clock

    # name = df_obj.__class__.__name__
    if stdout is None:
        stdout = open("out.log", "w")
    log = logger.Logger(stdout, 5)

    if kmesh is not None:
        from pyscf.pbc.scf import KRHF
        kpts = cell.get_kpts(kmesh)
        scf_obj = KRHF(cell, kpts=kpts)
        scf_obj.exxdiv = None
        scf_obj.verbose = 4
        dm0 = scf_obj.get_init_guess(key="minao")
    else:
        from pyscf.pbc.scf import RHF
        scf_obj = RHF(cell)
        scf_obj.exxdiv = None
        dm0 = scf_obj.get_init_guess(key="minao")

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = df_obj.build()
    t1 = log.timer("build", *t0)

    t0 = (process_clock(), perf_counter())
    vj1 = scf_obj.get_jk(cell, dm0, with_j=True, with_k=False)[0]
    t1 = log.timer("get_j", *t0)

    t0 = (process_clock(), perf_counter())
    vk1 = scf_obj.get_jk(cell, dm0, with_j=False, with_k=True)[1]
    t2 = log.timer("get_k", *t0)

    log.info("chk file size: %6.2e GB", os.path.getsize(tmp) / 1e9)

    if chkfile is None:
        chkfile = "vjk.chk"
    print(f"Dumping vj and vk to {chkfile}")
    
    from pyscf.lib.chkfile import dump
    dump(chkfile, "vj", vj1)
    dump(chkfile, "vk", vk1)

def scf(cell, kmesh=None, df_obj=None, tmp=None, chkfile=None, stdout=None):
    from pyscf import lib
    from pyscf.lib import logger
    from pyscf.lib.logger import perf_counter
    from pyscf.lib.logger import process_clock

    if stdout is None:
        stdout = open("out.log", "w")
    log = logger.Logger(stdout, 5)

    if kmesh is not None:
        from pyscf.pbc.scf import KRHF
        kpts = cell.get_kpts(kmesh)
        scf_obj = KRHF(cell, kpts=kpts)
        scf_obj.exxdiv = None
        dm0 = scf_obj.get_init_guess(key="minao")

        from pyscf.pbc.scf.addons import smearing_
        scf_obj = smearing_(scf_obj, sigma=0.1, method="fermi")

    else:
        raise NotImplementedError

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = df_obj.build()
    t1 = log.timer("build", *t0)

    t0 = (process_clock(), perf_counter())
    e_tot = scf_obj.kernel(dm0)
    t1 = log.timer("scf", *t0)

    dm = scf_obj.make_rdm1()
    ncycle = scf_obj.cycles

    log.info("ncycle = %2d, e_tot = %16.8f", ncycle, e_tot)
    log.info("chk file size: %6.2e GB", os.path.getsize(tmp) / 1e9)

    if chkfile is None:
        chkfile = "vjk.chk"
    print(f"Dumping vj and vk to {chkfile}")
    
    from pyscf.lib.chkfile import dump
    dump(chkfile, "dm", dm)
