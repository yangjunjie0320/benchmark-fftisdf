from utils import get_cell

from time import time
from argparse import ArgumentParser
import pyscf, numpy

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

from pyscf.isdf import isdf_local
ISDF_Local = isdf_local.ISDF_Local

def ISDF(cell=None, c0=10.0, rela_qr=1e-3, aoR_cutoff=1e-8):
    cell = cell.copy(deep=True)

    direct = False
    limited_memory = True
    with_robust_fitting = False
    build_V_K_bunchsize = 512

    from pyscf.lib import logger
    from pyscf.lib.logger import perf_counter
    from pyscf.lib.logger import process_clock
    t0 = (process_clock(), perf_counter())
    log = logger.new_logger(cell, 10)
    log.info("ISDF module: %s" % isdf_local.__file__)

    isdf_obj = ISDF_Local(
        cell, direct=direct,
        limited_memory=limited_memory, 
        with_robust_fitting=with_robust_fitting,
        build_V_K_bunchsize=build_V_K_bunchsize,
        aoR_cutoff=aoR_cutoff
    )

    isdf_obj.verbose = 10
    log.info("c0 = %6.2f" % c0)

    isdf_obj.build(c=c0, rela_cutoff=rela_qr, group=None)

    nip = isdf_obj.naux
    log.info(
        "Number of interpolation points = %d, effective CISDF = %6.2f",
        nip, nip / isdf_obj.nao
    )
    log.timer("ISDF build", *t0)
    return isdf_obj, nip / isdf_obj.nao

t = {}

def main(args : ArgumentParser):
    ke_cutoff = args.ke_cutoff
    cell = get_cell(args.name)
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.verbose = 10
    cell.ke_cutoff = ke_cutoff
    cell.build(dump_input=False)

    from pyscf.pbc.scf import RHF
    scf_obj = RHF(cell)
    scf_obj.exxdiv = "ewald"
    scf_obj.verbose = 10

    h1e = scf_obj.get_hcore()
    s1e = scf_obj.get_ovlp()
    e0_mo, c0_mo = scf_obj.eig(h1e, s1e)
    n0_mo = scf_obj.get_occ(e0_mo, c0_mo)
    dm0 = scf_obj.make_rdm1(c0_mo, n0_mo)

    c0 = args.c0
    rela_qr = args.rela_qr
    aoR_cutoff = args.aoR_cutoff

    t0 = time()
    isdf_obj, cisdf = ISDF(
        cell.copy(deep=True),
        c0=c0, rela_qr=rela_qr,
        aoR_cutoff=aoR_cutoff
    )
    scf_obj.with_df = isdf_obj
    t["ISDF build"] = time() - t0

    t0 = time()
    vj_sol, vk_sol = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv="ewald")
    vj_sol = vj_sol.reshape(h1e.shape)
    vk_sol = vk_sol.reshape(h1e.shape)
    vjk_sol = vj_sol - 0.5 * vk_sol
    f1e_sol = h1e + vjk_sol
    f1e_sol = f1e_sol.reshape(h1e.shape)
    t["ISDF JK"] = time() - t0

    e_sol = numpy.einsum('ij,ji->', 0.5 * (f1e_sol + h1e), dm0)
    assert e_sol.imag < 1e-10
    e_sol = e_sol.real + cell.energy_nuc()

    # print(f"### c0 = {args.c0:6.2f}, cisdf = {cisdf:6.2f}")
    # print(f"### rela_qr    = {args.rela_qr:6.2e}")
    # print(f"### aoR_cutoff = {args.aoR_cutoff:6.2e}")
    # print(f"### e_tot      = {e_sol: 6.2e}")

    print(f"### %10s, %10s, %10s, %10s, %16s" % ("c0", "rela_qr", "aoR_cutoff", "ke_cutoff", "e_tot"))
    print(f"### %10.2f, %10.2e, %10.2e, %10.2e, %16.6f" % (c0, rela_qr, aoR_cutoff, ke_cutoff, e_sol))

    l = max(len(k) for k in t.keys())
    info = f"### Time for %{l}s: %6.2f s"
    for k, v in t.items():
        print(info % (k, v))

    from pyscf.lib.chkfile import dump
    dump("isdf.h5", "ke_cutoff", ke_cutoff)
    dump("isdf.h5", "basis", cell._basis)
    dump("isdf.h5", "pseudo", cell._pseudo)
    dump("isdf.h5", "h1e", h1e)
    dump("isdf.h5", "s1e", s1e)
    dump("isdf.h5", "vjk", vjk_sol)
    dump("isdf.h5", "vj", vj_sol)
    dump("isdf.h5", "vk", vk_sol)
    dump("isdf.h5", "f1e", f1e_sol)
    dump("isdf.h5", "e_sol", e_sol)
    dump("isdf.h5", "dm0", dm0)
    dump("isdf.h5", "c0_mo", c0_mo)
    dump("isdf.h5", "n0_mo", n0_mo)
    dump("isdf.h5", "e0_mo", e0_mo)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--c0", type=float, default=5.0)
    parser.add_argument("--rela_qr", type=float, default=1e-3)
    parser.add_argument("--aoR_cutoff", type=float, default=1e-8)
    parser.add_argument("--ke_cutoff", type=float, default=40.0)
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k} = {v}")

    main(args)
