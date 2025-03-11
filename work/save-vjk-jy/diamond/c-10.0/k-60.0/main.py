from time import time
from argparse import ArgumentParser

import pyscf
from pyscf.lib import logger
from pyscf.lib.logger import perf_counter
from pyscf.lib.logger import process_clock

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

def ISDF(cell=None, **kwargs):
    cell = cell.copy(deep=True)
    kpts = cell.make_kpts([1, 1, 1])

    from fft_isdf import FFTISDF
    isdf_obj = FFTISDF(cell, kpts=kpts)
    isdf_obj.verbose = 10
    return isdf_obj

if __name__ == "__main__":
    from pyscf.lib import logger
    log = logger.Logger(open("out.log", "w"), logger.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ke_cutoff", type=float, default=70.0)
    parser.add_argument("--c0", type=float, default=5.0)
    parser.add_argument("--k0",     type=(lambda x: None if x == "None" else float(x)), default=None)
    parser.add_argument("--exxdiv", type=(lambda x: None if x == "None" else str(x)),   default=None)
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
    from utils import save_vjk_from_1e_dm0 as main
    name = config.pop("name")
    exxdiv = config.pop("exxdiv")

    cell = get_cell(name)
    cell.ke_cutoff = config.pop("ke_cutoff")
    cell.build(dump_input=False)

    from pyscf.lib.logger import process_clock, perf_counter
    t0 = (process_clock(), perf_counter())
    isdf_obj  = ISDF(cell)
    c0 = config.pop("c0")
    k0 = config.pop("k0")
    if k0 is None:
        k0 = cell.ke_cutoff

    from pyscf.pbc.tools.pbc import cutoff_to_mesh
    lv = cell.lattice_vectors()
    m0 = cutoff_to_mesh(lv, k0)
    g0 = cell.gen_uniform_grids(m0)
    print("c0 = ", c0, "k0 = ", k0)
    print("Parent grid mesh = %s, parent grid points = %s" % (m0, g0.shape))

    t0 = (process_clock(), perf_counter())
    inpx = isdf_obj.get_inpx(c0=c0, g0=g0, tol=1e-12)
    nip = inpx.shape[0]
    nao = cell.nao_nr()
    cisdf = nip / nao
    print(inpx.shape, nao, nip, cisdf)
    log.timer("Cholesky", *t0)

    t0 = (process_clock(), perf_counter())
    isdf_obj.build(inpx=inpx)
    log.timer("ISDF build", *t0)

    e_tot, chk_path = main(isdf_obj, exxdiv, log)
    print("Successfully saved all results to %s" % chk_path)

    kl.append("m0")
    vl.append(inpx.shape[0])

    kl.append("cisdf")
    vl.append(cisdf)

    l = max(len(k) for k in kl)
    l = max(l, 10)
    kline = ", ".join(f"%{l}s" % k for k in kl)
    vline = ", ".join(f"%{l}.2e" % v for v in vl)
    kline += ", %12s" % "e_tot"
    vline += ", % 12.6f" % e_tot
    log.info("### " + kline % kl)
    log.info("### " + vline % vl)
