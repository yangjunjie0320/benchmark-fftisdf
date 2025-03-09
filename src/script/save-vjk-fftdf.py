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

if __name__ == "__main__":
    from pyscf.lib import logger
    log = logger.Logger(open("out.log", "w"), logger.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ke_cutoff", type=float, default=40.0)
    parser.add_argument("--exxdiv", type=str, default="ewald")
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

    time_table = {}
    from utils import get_cell
    from utils import save_vjk_from_1e_dm0 as main
    name = config.pop("name")
    exxdiv = config.pop("exxdiv")

    cell = get_cell(name)
    cell.ke_cutoff = config.pop("ke_cutoff")
    cell.build(dump_input=False)

    from pyscf.lib.logger import process_clock, perf_counter
    t0 = (process_clock(), perf_counter())
    from pyscf.pbc.df import FFTDF
    df = FFTDF(cell)
    log.timer("FFTDF build", *t0)

    e_tot, chk_path = main(df, exxdiv, log)
    print("Successfully saved all results to %s" % chk_path)

    l = max(len(k) for k in kl)
    l = max(l, 10)
    kline = ", ".join(f"%{l}s" % k for k in kl)
    vline = ", ".join(f"%{l}.2e" % v for v in vl)
    kline += ", %12s" % "e_tot"
    vline += ", % 12.6f" % e_tot
    log.info("### " + kline % kl)
    log.info("### " + vline % vl)
