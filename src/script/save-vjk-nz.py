from time import time
from argparse import ArgumentParser

import pyscf
from pyscf.lib import logger
from pyscf.lib.logger import perf_counter
from pyscf.lib.logger import process_clock

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

from pyscf.isdf import isdf_local
ISDF_Local = isdf_local.ISDF_Local

def ISDF(cell=None, c0=10.0, rela_qr=1e-3, aoR_cutoff=1e-8):
    kwargs = {
        "aoR_cutoff" : aoR_cutoff,
        "direct" : False,
        "limited_memory" : False,
        "with_robust_fitting" : False,
        "build_V_K_bunchsize" : 512,
    }

    cell = cell.copy(deep=True)
    isdf_obj = ISDF_Local(cell, **kwargs)

    isdf_obj.verbose = 10
    isdf_obj.build(c=c0, rela_cutoff=rela_qr, group=None)

    nip = isdf_obj.naux
    return isdf_obj, nip / isdf_obj.nao

if __name__ == "__main__":
    from pyscf.lib.logger import logger
    log = logger.new_logger(logger, logger.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ke_cutoff", type=float, default=40.0)
    parser.add_argument("--c0", type=float, default=5.0)
    parser.add_argument("--rela_qr", type=float, default=1e-3)
    parser.add_argument("--aoR_cutoff", type=float, default=1e-8)
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
    isdf_obj, cisdf = ISDF(cell, **config)
    log.timer("ISDF build", t0)

    e_tot, chk_path = main(isdf_obj, exxdiv, log)
    k.append("e_tot")
    k.append("cisdf")
    v.append(e_tot)
    v.append(cisdf)

    l = max(len(k) for k in kl)
    l = max(l, 10)
    kline = ", ".join(f"%{l}s" % k for k in kl)
    vline = ", ".join(f"%{l-2}.2e" % v for v in vl)
    log.info("### " + kline % kl)
    log.info("### " + vline % vl)
