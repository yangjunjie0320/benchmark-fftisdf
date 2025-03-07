from time import time
from argparse import ArgumentParser
import pyscf, numpy

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

def ISDF(cell, rcut_epsilon=1e-6, ke_epsilon=1e-6, isdf_thresh=1e-6):
    import isdfx
    isdf_obj = isdfx.ISDFX(
        cell,
        rcut_epsilon    = rcut_epsilon, 
        ke_epsilon      = ke_epsilon,
        isdf_thresh     = isdf_thresh,
        multigrid_on    = True,
        fit_dense_grid  = True,
        fit_sparse_grid = False,
        )   
    return isdf_obj

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ke_cutoff", type=float, default=40.0)
    parser.add_argument("--rcut_epsilon", type=float, default=1e-6)
    parser.add_argument("--ke_epsilon", type=float, default=1e-6)
    parser.add_argument("--isdf_thresh", type=float, default=1e-6)
    args = parser.parse_args()
    config = vars(args)

    for k, v in config.items():
        print(f"{k} = {v}")

    time_table = {}
    from utils import save_vjk_from_1e_dm0 as main
    e_tot = main(config, time_table)

    l = max(len(k) for k in time_table.keys())
    info = f"### Time for %{l}s: %6.2f s"
    for k, v in time_table.items():
        print(info % (k, v))

    print(f"### e_tot = {e_tot}")
