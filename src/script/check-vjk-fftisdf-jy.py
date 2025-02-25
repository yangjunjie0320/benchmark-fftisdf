from utils import get_cell
from time import time

from argparse import ArgumentParser
import pyscf, numpy

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

t = {}

def main(args : ArgumentParser):
    cell = get_cell(args.name)
    cell.max_memory = PYSCF_MAX_MEMORY
    # cell.ke_cutoff = args.ke_cutoff
    cell.build(dump_input=False)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    scf_obj.verbose = 0
    dm0 = scf_obj.get_init_guess()

    scf_obj.with_df.dump_flags()
    scf_obj.with_df.check_sanity()

    t0 = time()
    vj_ref, vk_ref = scf_obj.with_df.get_jk(dm0, hermi=1)
    vjk_ref = vj_ref - 0.5 * vk_ref
    t["vjk-ref"] = time() - t0

    from fft_isdf import FFTISDF
    scf_obj.with_df = FFTISDF(cell, kpts)
    
    from utils import INFO
    c0 = INFO[args.name]["c0"]
    k0 = INFO[args.name]["k0"]
    from pyscf.pbc.tools.pbc import cutoff_to_mesh
    lv = cell.lattice_vectors()
    g0 = cell.gen_uniform_grids(cutoff_to_mesh(lv, k0))
    scf_obj.with_df.verbose = 10
    inpx = scf_obj.with_df.get_inpx(g0=g0, c0=c0, tol=1e-12)
    
    t0 = time()
    scf_obj.with_df.tol = 1e-10
    scf_obj.with_df.max_memory = PYSCF_MAX_MEMORY
    scf_obj.with_df.build(inpx)
    t["FFTISDF build"] = time() - t0

    t0 = time()
    vj_sol, vk_sol = scf_obj.with_df.get_jk(dm0, hermi=1)
    vjk_sol = vj_sol - 0.5 * vk_sol
    t["FFTISDF JK"] = time() - t0

    gmesh = scf_obj.with_df.mesh
    gmesh = "-".join([str(x) for x in gmesh])

    err_vj = abs(vj_ref - vj_sol).max()
    err_vk = abs(vk_ref - vk_sol).max()
    err_vjk = abs(vjk_ref - vjk_sol).max()

    print(f"### c0 = {c0:6.2f}, kmesh = {args.kmesh}, gmesh = {gmesh}")
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
    parser.add_argument("--kmesh", type=str, default="1-1-1")
    args = parser.parse_args()

    main(args)
