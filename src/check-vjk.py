from utils import get_cell
from time import time

from argparse import ArgumentParser
import pyscf, numpy

from os import environ
PYSCF_MAX_MEMORY = int(environ.get("PYSCF_MAX_MEMORY", 1000))

def main(args : ArgumentParser):
    cell = get_cell(args.name)
    cell.max_memory = PYSCF_MAX_MEMORY
    cell.build(dump_input=False)

    kmesh = args.kmesh
    kpts = cell.make_kpts(kmesh)
    print(f"cell.max_memory = {cell.max_memory} MB")

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    scf_obj.verbose = 0
    h1e = scf_obj.get_hcore()
    s1e = scf_obj.get_ovlp()
    e0_mo, c0_mo = scf_obj.eig(h1e, s1e)
    n0_mo = scf_obj.get_occ(e0_mo, c0_mo)
    dm0 = scf_obj.make_rdm1(c0_mo, n0_mo)

    t0 = time()
    vj_ref, vk_ref = scf_obj.with_df.get_jk(dm0, hermi=1)
    v_ref = vj_ref - 0.5 * vk_ref
    f1e = h1e + v_ref
    t1 = time() - t0

    e_ref = numpy.einsum('kij,kji->', 0.5 * (f1e + h1e), dm0)
    assert e_ref.imag < 1e-10
    e_ref = e_ref.real

    assert abs(e_ref - scf_obj.energy_elec(dm_kpts=dm0)[0]) < 1e-10

    from fft_isdf import FFTISDF
    scf_obj.with_df = FFTISDF(cell, kpts)
    
    from pyscf.pbc.tools.pbc import cutoff_to_mesh
    lv = cell.lattice_vectors()
    k0 = args.k0
    g0 = cell.gen_uniform_grids(cutoff_to_mesh(lv, k0))
    
    c0 = args.c0
    scf_obj.with_df.verbose = 10
    inpx = scf_obj.with_df.get_inpx(g0=g0, c0=c0, tol=1e-12)
    
    t0 = time()
    scf_obj.with_df.tol = 1e-10
    scf_obj.with_df.max_memory = PYSCF_MAX_MEMORY
    scf_obj.with_df.build(inpx)
    t2 = time() - t0

    t0 = time()
    vj_sol, vk_sol = scf_obj.with_df.get_jk(dm0, hermi=1)
    v_sol = vj_sol - 0.5 * vk_sol
    f1e_sol = h1e + v_sol
    t3 = time() - t0

    e_sol = numpy.einsum('kij,kji->', 0.5 * (f1e_sol + h1e), dm0)
    assert e_sol.imag < 1e-10
    e_sol = e_sol.real

    err_e_per_atom = abs(e_ref - e_sol) / cell.natm
    err_v = abs(v_ref - v_sol).max() # / cell.natm

    print(f"### c0 = {c0:6.2f}, ng = {len(g0):6d}, err_e_per_atom = {err_e_per_atom: 6.2e}, err_v = {err_v: 6.2e}")
    print(f"### Time for FFTDF      JK: {t1:6.2f} s")
    print(f"### Time for FFTISDF build: {t2:6.2f} s")
    print(f"### Time for FFTISDF    JK: {t3:6.2f} s")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ke_cutoff", type=float, default=10.0)
    parser.add_argument("--k0", type=float, default=10.0)
    parser.add_argument("--c0", type=float, default=5.0)
    parser.add_argument("--kmesh", type=list, default=[1, 1, 1])
    args = parser.parse_args()

    main(args)
