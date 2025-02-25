from utils import get_cell
from time import time

from argparse import ArgumentParser
import pyscf, numpy

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

from pyscf.isdf import isdf_local
ISDF_Local = isdf_local.ISDF_Local

class ISDF(object):
    cell = None
    group = None

    kpts = None
    kmesh = None
    c0 = None
    verbose = 10

    _isdf = None
    _isdf_to_save = None
    
    def __init__(self, cell=None, kpts=None):
        assert kpts is None
        self.cell = cell
        self.kpts = kpts

        self.c0 = 5.0
        self.rela_qr = 1e-3
        self.aoR_cutoff = 1e-8
        self.direct = True
        self.with_robust_fitting = False
        self.build_V_K_bunchsize = 512

    def build(self):
        cell = self.cell.copy(deep=True)

        # group = self.group
        # assert group is not None

        direct = self.direct
        c0 = self.c0
        rela_qr = self.rela_qr
        aoR_cutoff = self.aoR_cutoff
        build_V_K_bunchsize = self.build_V_K_bunchsize
        with_robust_fitting = self.with_robust_fitting

        from pyscf.lib import logger
        from pyscf.lib.logger import perf_counter
        from pyscf.lib.logger import process_clock
        t0 = (process_clock(), perf_counter())
        log = logger.new_logger(cell, 10)
        log.info("ISDF module: %s" % isdf_local.__file__)

        isdf_obj = ISDF_Local(
            cell, limited_memory=True, direct=direct,
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
        self.cisdf = nip / isdf_obj.nao
        log.timer("ISDF build", *t0)
        return isdf_obj

t = {}

def main(args : ArgumentParser):
    cell = get_cell(args.name)
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.build(dump_input=False)

    scf_obj = pyscf.pbc.scf.RHF(cell)
    scf_obj.exxdiv = None
    scf_obj.verbose = 0
    h1e = scf_obj.get_hcore()
    s1e = scf_obj.get_ovlp()
    e0_mo, c0_mo = scf_obj.eig(h1e, s1e)
    n0_mo = scf_obj.get_occ(e0_mo, c0_mo)
    dm0 = scf_obj.make_rdm1(c0_mo, n0_mo)

    gmesh = scf_obj.with_df.mesh
    gmesh = "-".join([str(x) for x in gmesh])

    t0 = time()
    vj_ref, vk_ref = scf_obj.with_df.get_jk(dm0, hermi=1)
    vjk_ref = vj_ref - 0.5 * vk_ref
    f1e_ref = h1e + vjk_ref
    t["FFTDF JK"] = time() - t0

    e_ref = numpy.einsum('ij,ji->', 0.5 * (f1e_ref + h1e), dm0)
    assert e_ref.imag < 1e-10
    e_ref = e_ref.real
    assert abs(e_ref - scf_obj.energy_elec(dm0)[0]) < 1e-10

    isdf_obj = ISDF(cell)
    isdf_obj.rela_qr = args.rela_qr
    isdf_obj.aoR_cutoff = args.aoR_cutoff
    isdf_obj.direct = True
    isdf_obj.with_robust_fitting = True
    isdf_obj.c0 = args.c0

    t0 = time()
    scf_obj.with_df = isdf_obj.build()
    t["ISDF build"] = time() - t0

    t0 = time()
    vj_sol, vk_sol = scf_obj.with_df.get_jk(dm0, hermi=1)
    vjk_sol = vj_sol - 0.5 * vk_sol
    f1e_sol = h1e + vjk_sol
    f1e_sol = f1e_sol.reshape(h1e.shape)
    t["ISDF JK"] = time() - t0

    e_sol = numpy.einsum('ij,ji->', 0.5 * (f1e_sol + h1e), dm0)
    assert e_sol.imag < 1e-10
    e_sol = e_sol.real

    err_ene = abs(e_ref - e_sol) / cell.natm
    err_vj = abs(vj_ref - vj_sol).max()
    err_vk = abs(vk_ref - vk_sol).max()
    err_vjk = abs(vjk_ref - vjk_sol).max()

    c0 = isdf_obj.cisdf

    print(f"### c0 = {c0:6.2f}, rela_qr = {isdf_obj.rela_qr:6.2e}")
    print(f"### aoR_cutoff = {isdf_obj.aoR_cutoff:6.2e}")
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
    parser.add_argument("--kmesh", type=str, default="1-1-1")
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k} = {v}")

    main(args)
