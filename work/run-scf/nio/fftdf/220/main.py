from time import time
from argparse import ArgumentParser

import sys, os
import pyscf, numpy
from pyscf.lib import logger
from pyscf.lib.logger import perf_counter
from pyscf.lib.logger import process_clock

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))
TMPDIR = os.getenv("TMPDIR", "/tmp")
assert os.path.exists(TMPDIR), f"TMPDIR {TMPDIR} does not exist"

def DF(cell, kwargs):
    cell = cell.copy(deep=True)
    df = kwargs.pop("df")

    mesh = kwargs.pop("mesh", "1,1,1")
    mesh = [int(x) for x in mesh.split(",")]
    kpts = cell.make_kpts(mesh)
    assert mesh == [1, 1, 1]

    if df == "gdf":
        assert len(kwargs) == 0, f"kwargs = {kwargs}"

        from pyscf.pbc.df import GDF
        df_obj = GDF(cell, kpts=kpts)
        df_obj.verbose = 5
        df_obj.build()

    elif df == "fftdf":
        assert len(kwargs) == 0, f"kwargs = {kwargs}"

        from pyscf.pbc.df import FFTDF
        df_obj = FFTDF(cell, kpts=kpts)
        df_obj.verbose = 5
        df_obj.build()

    elif df == "fftisdf-jy":
        k0 = kwargs.pop("k0")
        c0 = kwargs.pop("c0")
        assert len(kwargs) == 0, f"kwargs = {kwargs}"

        from fft_isdf import FFTISDF
        df_obj = FFTISDF(cell, kpts=kpts)
        df_obj.verbose = 5

        inpx = None
        if k0 is None:
            inpx = df_obj.get_inpx(c0=c0, g0=None, tol=1e-12)
        else:
            from pyscf.pbc.tools.pbc import cutoff_to_mesh
            lv = cell.lattice_vectors()
            m0 = cutoff_to_mesh(lv, k0)
            g0 = cell.gen_uniform_grids(m0)
            inpx = df_obj.get_inpx(g0=g0, c0=c0, tol=1e-12)

        assert inpx is not None
        df_obj.build(inpx=inpx)

    elif df == "fftisdf-nz":
        aoR_cutoff = kwargs.pop("aoR_cutoff")
        rela_qr = kwargs.pop("rela_qr")
        c0 = kwargs.pop("c0")
        assert len(kwargs) == 0, f"kwargs = {kwargs}"

        kwargs = {
            "aoR_cutoff" : aoR_cutoff,
            "direct" : False,
            "limited_memory" : False,
            "with_robust_fitting" : False,
            "build_V_K_bunchsize" : 64,
        }

        from pyscf.isdf import isdf_local
        FFTISDF = isdf_local.ISDF_Local
        df_obj = FFTISDF(cell, **kwargs)
        df_obj.verbose = 5
        df_obj.build(c=c0, rela_cutoff=rela_qr, group=None)
        nip = df_obj.naux

    elif df == "fftisdf-ks":
        rcut_epsilon = kwargs.pop("rcut_epsilon")
        ke_epsilon = kwargs.pop("ke_epsilon")
        isdf_thresh = kwargs.pop("isdf_thresh")
        assert len(kwargs) == 0, f"kwargs = {kwargs}"

        kwargs = {
            "rcut_epsilon"    : rcut_epsilon,
            "ke_epsilon"      : ke_epsilon,
            "isdf_thresh"     : isdf_thresh,
            "multigrid_on"    : True,
            "fit_dense_grid"  : True,
            "fit_sparse_grid" : False,
        }

        sys.path.append("/home/junjiey/packages/PeriodicIntegrals/PeriodicIntegrals-junjie-benchmark/")
        from isdfx.isdfx import ISDFX
        df_obj = ISDFX(cell, **kwargs)
        df_obj.build(with_j=True, with_k=True)

    else:
        raise NotImplementedError

    return df_obj

def main(config):
    name = config.pop("name")
    exxdiv = config.pop("exxdiv")
    chk_path = config.pop("chk_path")
    ke_cutoff = config.pop("ke_cutoff")

    from utils import get_cell, INFO
    cell = get_cell(name)
    cell.ke_cutoff = INFO[name]["ke_cutoff"]
    if ke_cutoff is not None:
        cell.ke_cutoff = ke_cutoff
    print(f"ke_cutoff = {cell.ke_cutoff}")
    cell.build(dump_input=False)

    t0 = (process_clock(), perf_counter())
    df_obj = DF(cell, config)
    log.timer("build", *t0)

    afm_guess = INFO[name].get("afm_guess", None)
    if afm_guess is not None:
        print("Running UKS with PBE0")
        from pyscf.pbc.scf import UKS
        scf_obj = UKS(cell)
        scf_obj.xc = "PBE0"
        scf_obj.exxdiv = exxdiv
        scf_obj.verbose = 5
        scf_obj.with_df = df_obj
        scf_obj._is_mem_enough = lambda : False      
        scf_obj.conv_tol = 1e-6
    else:
        print("Running RKS with PBE0")
        from pyscf.pbc.scf import RKS
        scf_obj = RKS(cell)
        scf_obj.xc = "PBE0"
        scf_obj.exxdiv = exxdiv
        scf_obj.verbose = 5
        scf_obj.with_df = df_obj
        scf_obj._is_mem_enough = lambda : False
        scf_obj.conv_tol = 1e-6

    if chk_path is not None:
        from pyscf.lib.chkfile import load
        dm0 = load(chk_path, "dm0")
        c0_mo = load(chk_path, "c0_mo")
        n0_mo = load(chk_path, "n0_mo")

        from pyscf.lib import tag_array
        dm0 = tag_array(dm0, mo_coeff=c0_mo, mo_occ=n0_mo)
    else:
        dm0 = scf_obj.get_init_guess(key="minao")
        if afm_guess is not None:
            from utils import gen_afm_guess
            dm0 = gen_afm_guess(cell, dm0, afm_guess)

    assert exxdiv == None
    t0 = (process_clock(), perf_counter())
    vj = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv=exxdiv, with_k=False, with_j=True)[0]
    log.timer("vj", *t0)

    t0 = (process_clock(), perf_counter())
    vk = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv=exxdiv, with_k=True, with_j=False)[1]
    log.timer("vk", *t0)

    scf_obj.kernel(dm0)
    e_tot = scf_obj.e_tot
    c0_mo = scf_obj.mo_coeff
    n0_mo = scf_obj.mo_occ
    e0_mo = scf_obj.mo_energy
    dm0 = scf_obj.make_rdm1(c0_mo, n0_mo)

    chk_path = os.path.join(TMPDIR, f"scf.h5")
    from pyscf.lib.chkfile import dump
    dump(chk_path, "natm", cell.natm)
    dump(chk_path, "ke_cutoff", cell.ke_cutoff)
    dump(chk_path, "basis", cell._basis)
    dump(chk_path, "pseudo", cell._pseudo)

    dump(chk_path, "dm0", dm0)
    dump(chk_path, "c0_mo", c0_mo)
    dump(chk_path, "n0_mo", n0_mo)
    dump(chk_path, "e0_mo", e0_mo)
    dump(chk_path, "e_tot", e_tot)
    print("Successfully saved all results to %s" % chk_path)
    return e_tot

if __name__ == "__main__":
    from pyscf.lib import logger
    log = logger.Logger(open("out.log", "w"), logger.DEBUG)

    parser = ArgumentParser()
    # arguments for all methods
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ke_cutoff", type=(lambda x: None if x == "None" else float(x)), default=None)
    parser.add_argument("--exxdiv", type=(lambda x: None if x == "None" else str(x)), default=None)
    parser.add_argument("--df", type=str, default="fftdf", choices=["gdf", "fftdf", "fftisdf-jy", "fftisdf-nz", "fftisdf-ks"])
    parser.add_argument("--mesh", type=str, default="1,1,1", choices=["1,1,1", "2,2,2"])
    parser.add_argument("--chk_path", type=str, default=None)
    args = parser.parse_args()

    # arguments for FFTISDF-JY
    if args.df == "fftisdf-jy":
        parser.add_argument("--c0", type=float, default=5.0)
        parser.add_argument("--k0", type=(lambda x: None if x == "None" else float(x)), default=None)
        args = parser.parse_args()

    # arguments for FFTISDF-NZ
    if args.df == "fftisdf-nz":
        parser.add_argument("--c0", type=float, default=5.0)
        parser.add_argument("--rela_qr", type=float, default=1e-3)
        parser.add_argument("--aoR_cutoff", type=float, default=1e-8)
        args = parser.parse_args()

    # arguments for FFTISDF-KS
    if args.df == "fftisdf-ks":
        parser.add_argument("--rcut_epsilon", type=float,  default=1e-6)
        parser.add_argument("--ke_epsilon",   type=float,  default=1e-6)
        parser.add_argument("--isdf_thresh",  type=float,  default=1e-6)
        args = parser.parse_args()

    config = vars(args)

    kl = []
    vl = []
    for k, v in config.items():
        if isinstance(v, float):
            print("%s = % 6.2e" % (k, v))
            kl.append(k)
            vl.append(v)
        else:
            print("%s = %s" % (k, v))
    print()

    from utils import get_cell
    e_tot = main(config)

    l = max(len(k) for k in kl)
    l = max(l, 10)
    kline = ", ".join(f"%{l}s" % k for k in kl)
    vline = ", ".join(f"%{l}.2e" % v for v in vl)
    kline += ", %12s" % "e_tot"
    vline += ", % 12.6f" % e_tot
    log.info("### " + kline % kl)
    log.info("### " + vline % vl)
