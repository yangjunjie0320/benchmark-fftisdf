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
    cell = get_cell(args.name)
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.verbose = 10
    cell.ke_cutoff = args.ke_cutoff
    cell.build(dump_input=False)

    from pyscf.pbc.scf import RHF
    scf_obj = RHF(cell)
    # scf_obj.exxdiv = None
    scf_obj.verbose = 10
    h1e = scf_obj.get_hcore()
    s1e = scf_obj.get_ovlp()
    e0_mo, c0_mo = scf_obj.eig(h1e, s1e)
    n0_mo = scf_obj.get_occ(e0_mo, c0_mo)
    dm0 = scf_obj.make_rdm1(c0_mo, n0_mo)

    t0 = time()
    scf_obj.with_df.verbose = 10
    vj_ref, vk_ref = scf_obj.with_df.get_jk(dm0, hermi=1, exxdiv="ewald")
    vjk_ref = vj_ref - 0.5 * vk_ref
    f1e_ref = h1e + vjk_ref
    t["FFTDF JK"] = time() - t0

    e_ref = numpy.einsum('ij,ji->', 0.5 * (f1e_ref + h1e), dm0)
    assert e_ref.imag < 1e-10
    e_ref = e_ref.real
    assert abs(e_ref - scf_obj.energy_tot(dm=dm0)) < 1e-10

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

    err_ene = abs(e_ref - e_sol) / cell.natm
    err_vj = abs(vj_ref - vj_sol).max()
    err_vk = abs(vk_ref - vk_sol).max()
    err_vjk = abs(vjk_ref - vjk_sol).max()

    print(f"### c0 = {args.c0:6.2f}, cisdf = {cisdf:6.2f}")
    print(f"### rela_qr = {args.rela_qr:6.2e}")
    print(f"### aoR_cutoff = {args.aoR_cutoff:6.2e}")
    print(f"### err_ene = {err_ene: 6.2e}")
    print(f"### err_vj  = {err_vj: 6.2e}")
    print(f"### err_vk  = {err_vk: 6.2e}")
    print(f"### err_vjk = {err_vjk: 6.2e}")

    l = max(len(k) for k in t.keys())
    info = f"### Time for %{l}s: %6.2f s"
    for k, v in t.items():
        print(info % (k, v))

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
