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
    cell.ke_cutoff = args.ke_cutoff
    cell.build(dump_input=False)

    kmesh = [int(x) for x in args.kmesh.split("-")]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
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

    e_ref = numpy.einsum('kij,kji->', 0.5 * (f1e_ref + h1e), dm0)
    assert e_ref.imag < 1e-10
    e_ref = e_ref.real / nkpts
    assert abs(e_ref - scf_obj.energy_elec(dm0)[0]) < 1e-10

    from fft_isdf import FFTISDF
    scf_obj.with_df = FFTISDF(cell, kpts)
    
    from utils import INFO
    # c0 = INFO[args.name]["c0"]
    # k0 = INFO[args.name]["k0"]
    c0 = args.c0
    k0 = args.k0
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
    f1e_sol = h1e + vjk_sol
    f1e_sol = f1e_sol.reshape(h1e.shape)
    t["FFTISDF JK"] = time() - t0

    e_sol = numpy.einsum('kij,kji->', 0.5 * (f1e_sol + h1e), dm0)
    assert e_sol.imag < 1e-10
    e_sol = e_sol.real / nkpts

    err_ene = abs(e_ref - e_sol) / cell.natm
    err_vj = abs(vj_ref - vj_sol).max()
    err_vk = abs(vk_ref - vk_sol).max()
    err_vjk = abs(vjk_ref - vjk_sol).max()

    c0 = args.c0
    k0 = args.k0
    g0 = cutoff_to_mesh(lv, k0)
    g0 = "-".join([str(x) for x in g0])
    print(f"### c0 = {c0:6.2f}, g0 = {g0}")
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
    parser.add_argument("--k0", type=float, default=20.0)
    parser.add_argument("--ke_cutoff", type=float, default=200.0)
    parser.add_argument("--kmesh", type=str, default="1-1-1")
    args = parser.parse_args()

    for k, v in args.__dict__.items():
        print(f"{k} = {v}")

    main(args)
