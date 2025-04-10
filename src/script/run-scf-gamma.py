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

def DF(cell, **kwargs):
    cell = cell.copy(deep=True)
    df = kwargs.pop("df")

    if df == "gdf":
        assert len(kwargs) == 0, f"kwargs = {kwargs}"
        from pyscf.pbc.df import GDF
        df_obj = GDF(cell)
        df_obj.verbose = 5
        df_obj.build()

    elif df == "fftdf":
        assert len(kwargs) == 0, f"kwargs = {kwargs}"
        from pyscf.pbc.df import FFTDF
        df_obj = FFTDF(cell)
        df_obj.verbose = 5
        df_obj.build()

    elif df == "fftisdf-jy":
        k0 = kwargs.pop("k0")
        c0 = kwargs.pop("c0")
        assert len(kwargs) == 0, f"kwargs = {kwargs}"

        from fft_isdf import FFTISDF
        df_obj = FFTISDF(cell)
        df_obj.verbose = 5

        g0 = None
        if k0 is not None:
            from pyscf.pbc.tools.pbc import cutoff_to_mesh
            lv = cell.lattice_vectors()
            m0 = cutoff_to_mesh(lv, k0)
            g0 = cell.gen_uniform_grids(m0)
        inpx = df_obj.get_inpx(g0=g0, c0=c0, tol=1e-12)

        assert inpx is not None
        df_obj.build(inpx=inpx)

    elif df == "fftisdf-nz":
        from pyscf.isdf import isdf_local
        FFTISDF = isdf_local.ISDF_Local

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

        df_obj = FFTISDF(cell, **kwargs)
        df_obj.verbose = 5
        df_obj.build(c=c0, rela_cutoff=rela_qr, group=None)

    elif df == "fftisdf-ks":
        path  = [os.path.expanduser("~/packages"), "PeriodicIntegrals"]
        path += ["PeriodicIntegrals-junjie-benchmark", "isdfx"]
        sys.path.append(os.path.join(*path))
        from isdfx.isdfx import ISDFX

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
    cell.build(dump_input=False)
    print(f"ke_cutoff = {cell.ke_cutoff}")

    afm_guess = INFO[name].get("afm_guess", None)
    is_u_scf = (afm_guess is not None)

    t0 = (process_clock(), perf_counter())
    df_obj = DF(cell, **config)
    log.timer("df", *t0)

    from pyscf.pbc.scf import RKS, UKS
    scf_obj = RKS(cell) if not is_u_scf else UKS(cell)
    scf_obj.xc = "PBE0"
    scf_obj.with_df = df_obj
    scf_obj.exxdiv = exxdiv
    scf_obj.verbose = 5
    scf_obj._is_mem_enough = lambda : False
    scf_obj.conv_tol = 1e-8

    dm0 = scf_obj.get_init_guess(key="minao")
    if afm_guess is not None:
        from utils import gen_afm_guess
        dm0 = gen_afm_guess(cell, dm0, afm_guess)

    if chk_path is not None:
        from pyscf.lib import tag_array
        from pyscf.lib.chkfile import load
        dm0 = load(chk_path, "dm0")
        c0_mo = load(chk_path, "c0_mo")
        n0_mo = load(chk_path, "n0_mo")
        dm0 = tag_array(dm0, mo_coeff=c0_mo, mo_occ=n0_mo)
        print("Successfully loaded dm0 from %s" % chk_path)

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
    parser.add_argument("--chk_path", type=(lambda x: None if x == "None" else str(x)), default=None)

    parser.add_argument("--c0", type=float, default=5.0)
    parser.add_argument("--k0", type=(lambda x: None if x == "None" else float(x)), default=None)
    parser.add_argument("--rela_qr", type=float, default=1e-3)
    parser.add_argument("--aoR_cutoff", type=float, default=1e-8)

    parser.add_argument("--rcut_epsilon", type=float,  default=1e-6)
    parser.add_argument("--ke_epsilon",   type=float,  default=1e-6)
    parser.add_argument("--isdf_thresh",  type=float,  default=1e-6)
    args = parser.parse_args()
    config = vars(args)

    if args.df not in ["fftisdf-jy", "fftisdf-nz"]:
        config.pop("c0")

    if not args.df == "fftisdf-nz":
        config.pop("aoR_cutoff")
        config.pop("rela_qr")

    if not args.df == "fftisdf-ks":
        config.pop("rcut_epsilon")
        config.pop("ke_epsilon")
        config.pop("isdf_thresh")

    if not args.df == "fftisdf-jy":
        config.pop("k0")

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

    e_tot = main(config)

    kl.append("e_tot")
    vl.append(e_tot)
    l = max([len(k) for k in kl] + [10])
    # kline = ", ".join(f"%{l}s" % k for k in kl[:-1])
    # vline = ", ".join(f"%{l}.2e" % v for v in vl[:-1])
    # kline += "%12s"
    # vline += "% 12.6f"
    kline = [f"%{l}s"   % k for k in kl[:-1]] + ["%12s" % kl[-1]]
    vline = [f"%{l}.2e" % v for v in vl[:-1]] + ["% 12.6f" % vl[-1]]
    kline = ", ".join(kline)
    vline = ", ".join(vline)
    log.info("### " + kline)
    log.info("### " + vline)
