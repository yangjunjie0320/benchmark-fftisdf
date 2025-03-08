from time import time
from argparse import ArgumentParser
import pyscf, numpy

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

def ISDF(cell, rcut_epsilon=1e-6, ke_epsilon=1e-6, isdf_thresh=1e-6):
    kwargs = {
        "rcut_epsilon"    : rcut_epsilon, 
        "ke_epsilon"      : ke_epsilon,
        "isdf_thresh"     : isdf_thresh,
        "multigrid_on"    : True,
        "fit_dense_grid"  : True,
        "fit_sparse_grid" : False,
    }

    import isdfx
    isdf_obj = isdfx.ISDFX(cell, **kwargs)

    t0 = time()
    isdf_obj.build(with_j=True, with_k=False)
    isdf_obj.build(with_j=False, with_k=True)
    time_table["ISDF build"] = time() - t0
    return isdf_obj

if __name__ == "__main__":
    from pyscf.lib.logger import logger
    log = logger.new_logger(logger, logger.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ke_cutoff", type=float, default=40.0)
    parser.add_argument("--rcut_epsilon", type=float, default=1e-6)
    parser.add_argument("--ke_epsilon", type=float, default=1e-6)
    parser.add_argument("--isdf_thresh", type=float, default=1e-6)
    parser.add_argument("--exxdiv", type=str, default="ewald")
    args = parser.parse_args()
    config = vars(args)

    kl = []
    vl = []
    for k, v in config.items():
        if isinstance(v, float):
            log.info("%s = % 6.2e", k, v)
            kl.append(k)
            vl.append(v)
        else:
            log.info("%s = %s", k, v)

    time_table = {}
    from utils import get_cell
    from utils import save_vjk_from_1e_dm0 as main
    name = config.pop("name")
    exxdiv = config.pop("exxdiv")

    kline = ", ".join(f"%{l}s" % k for k in kl)
    vline = ", ".join(f"%{l-2}.2e" % v for v in vl)
    log.info("### " + kline % kl)
    log.info("### " + vline % vl)
    ke_cutoff = config.pop("ke_cutoff")

    cell = get_cell(name)
    cell.ke_cutoff = ke_cutoff
    cell.build(dump_input=False)

    from pyscf.lib.logger import process_clock, perf_counter
    t0 = (process_clock(), perf_counter())
    isdf_obj = ISDF(cell, **config)
    log.timer("ISDF build", t0)

    e_tot, chk_path = main(isdf_obj, exxdiv, log)
    k.append("e_tot")
    v.append(e_tot)

    l = max(len(k) for k in kl)
    l = max(l, 10)
    kline = ", ".join(f"%{l}s" % k for k in kl)
    vline = ", ".join(f"%{l-2}.2e" % v for v in vl)
    log.info("### " + kline % kl)
    log.info("### " + vline % vl)
