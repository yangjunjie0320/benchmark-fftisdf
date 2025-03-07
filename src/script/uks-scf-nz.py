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

    from pyscf.pbc.scf import UKS
    scf_obj = UKS(cell)
    scf_obj.exxdiv = None
    if args.df == "gdf":
        from pyscf.pbc.df import GDF
        scf_obj.with_df = GDF(cell)
        scf_obj.with_df.build()
    else:
        raise NotImplementedError

    scf_obj.xc = "pbe0"
    scf_obj.exxdiv = "ewald"
    scf_obj.verbose = 10

    from utils import gen_afm_guess, INFO
    info = INFO[args.name]
    afm_guess = info.get("afm_guess", None)

    assert afm_guess is not None
    if afm_guess is not None:
    nao = cell.nao_nr()
    dm0 = scf_obj.get_init_guess(key="minao")
    dm0 = gen_afm_guess(cell, dm0, info["afm_guess"])
    dm0 = numpy.asarray(dm0).reshape(2, nao, nao)

    ovlp = scf_obj.get_ovlp()

    from pyscf.scf.uhf import mulliken_spin_pop
    mulliken_spin_pop(cell, dm0, ovlp)
    scf_obj.kernel(dm0=dm0)

    dm = scf_obj.make_rdm1()
    mulliken_spin_pop(cell, dm, ovlp)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--c0", type=float, default=5.0)
    parser.add_argument("--rela_qr", type=float, default=1e-3)
    parser.add_argument("--aoR_cutoff", type=float, default=1e-8)
    parser.add_argument("--ke_cutoff", type=float, default=40.0)
    parser.add_argument("--df", type=str, default="df", choices=["fftdf", "gdf", "fftisdf"])
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k} = {v}")

    main(args)
