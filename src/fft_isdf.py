import os, sys, h5py
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.pbc import fft, ifft

from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks

import line_profiler
# from utils import print_current_memory

PYSCF_MAX_MEMORY = int(os.environ.get("PYSCF_MAX_MEMORY", 2000))

# Naming convention:
# *_kpt: k-space array, which shapes as (nkpt, x, x)
# *_spc: super-cell stripe array, which shapes as (nspc, x, x)
# *_full: full array, shapes as (nspc * x, nspc * x)
# *_k1, *_k2: the k-space array at specified k-point

def kpts_to_kmesh(df_obj, kpts):
    from pyscf.pbc.tools import k2gamma
    kmesh = k2gamma.kpts_to_kmesh(df_obj.cell, kpts)
    assert numpy.allclose(kpts, df_obj.cell.get_kpts(kmesh))
    return kpts, kmesh

def spc_to_kpt(m_spc, phase):
    """Convert a matrix from the stripe form (in super-cell)
    to the k-space form.
    """
    nspc, nkpt = phase.shape
    m_kpt = numpy.dot(phase.conj().T, m_spc.reshape(nspc, -1))
    return m_kpt.reshape(m_spc.shape)

def kpt_to_spc(m_kpt, phase):
    """Convert a matrix from the k-space form to
    stripe form (in super-cell).
    """
    nspc, nkpt = phase.shape
    m_spc = numpy.dot(phase, m_kpt.reshape(nkpt, -1))
    m_spc = m_spc.reshape(m_kpt.shape)
    return m_spc.real

# should I put this in the build method? Instead of a separate function?
@line_profiler.profile
def build(df_obj, inpx=None, verbose=0):
    """
    Build the FFT-ISDF object.
    
    Args:
        df_obj: The FFT-ISDF object to build.
    """
    log = logger.new_logger(df_obj, verbose)
    t0 = (process_clock(), perf_counter())
    max_memory = max(2000, df_obj.max_memory - current_memory()[0])

    if df_obj._isdf is not None:
        log.info("Loading ISDF results from %s, skipping build", df_obj._isdf)
        from pyscf.lib.chkfile import load
        inpv_kpt = load(df_obj._isdf, "inpv_kpt")
        coul_kpt = load(df_obj._isdf, "coul_kpt")
        df_obj._inpv_kpt = inpv_kpt
        df_obj._coul_kpt = coul_kpt
        return inpv_kpt, coul_kpt

    df_obj.dump_flags()
    df_obj.check_sanity()

    cell = df_obj.cell
    kpts, kmesh = kpts_to_kmesh(df_obj, df_obj.kpts)
    nkpt = len(kpts)
    log.info("kmesh = %s", kmesh)
    log.info("kpts = \n%s", kpts)

    if inpx is None:
        inpx = df_obj.get_inpx(g0=None, c0=df_obj.c0)

    nip = inpx.shape[0]
    assert inpx.shape == (nip, 3)
    ngrid = df_obj.grids.coords.shape[0]
    nao = cell.nao_nr()

    if df_obj.blksize is None:
        blksize = max_memory * 1e6 * 0.2 / (nkpt * nip * 16)
        df_obj.blksize = max(1, int(blksize))
    df_obj.blksize = min(df_obj.blksize, ngrid)

    if df_obj.blksize >= ngrid:
        df_obj._fswap = None

    inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
    inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
    assert inpv_kpt.shape == (nkpt, nip, nao)
    log.debug("nip = %d, nao = %d, cisdf = %6.2f", nip, nao, nip / nao)
    t1 = log.timer("get interpolating vectors")
    
    fswap = None if df_obj._fswap is None else h5py.File(df_obj._fswap, "w")
    if fswap is None:
        log.debug("In-core version is used for eta_kpt, memory required = %6.2e GB, max_memory = %6.2e GB", nkpt * nip * 16 * ngrid / 1e9, max_memory / 1e3)
    else:
        log.debug("Out-core version is used for eta_kpt, disk space required = %6.2e GB.", nkpt * nip * 16 * ngrid / 1e9)
        log.debug("memory used for each block = %6.2e GB, each k-point = %6.2e GB", nkpt * nip * 16 * df_obj.blksize / 1e9, nip * ngrid * 16 / 1e9)
        log.debug("max_memory = %6.2e GB", max_memory / 1e3)

    # metx_kpt: (nkpt, nip, nip), eta_kpt: (nkpt, ngrid, nip)
    # assume metx_kpt is a numpy.array, while eta_kpt is a hdf5 dataset
    metx_kpt, eta_kpt = get_lhs_and_rhs(df_obj, inpv_kpt, fswap=fswap)

    coul_kpt = []
    for q in range(nkpt):
        t0 = (process_clock(), perf_counter())

        metx_q = metx_kpt[q]
        assert metx_q.shape == (nip, nip)

        kern_q = get_kern(df_obj, eta_q=eta_kpt[q], kpt_q=kpts[q])
        coul_q = df_obj.lstsq(metx_q, kern_q, tol=df_obj.tol, verbose=verbose)
        assert coul_q.shape == (nip, nip)

        log.info("Finished solving Coulomb kernel for q = %3d / %3d", q + 1, nkpt)
        log.timer("solving Coulomb kernel", *t0)
        coul_kpt.append(coul_q)

    coul_kpt = numpy.asarray(coul_kpt)
    df_obj._coul_kpt = coul_kpt
    df_obj._inpv_kpt = inpv_kpt

    if df_obj._isdf_to_save is not None:
        df_obj._isdf = df_obj._isdf_to_save

    if df_obj._isdf is not None:
        from pyscf.lib.chkfile import dump
        dump(df_obj._isdf, "coul_kpt", coul_kpt)
        dump(df_obj._isdf, "inpv_kpt", inpv_kpt)

    t1 = log.timer("building ISDF", *t0)
    if fswap is not None:
        fswap.close()
    return inpv_kpt, coul_kpt

@line_profiler.profile
def get_lhs_and_rhs(df_obj, inpv_kpt, fswap=None):
    log = logger.new_logger(df_obj, df_obj.verbose)

    grids = df_obj.grids
    assert grids is not None

    coord = grids.coords
    ngrid = coord.shape[0]

    kpts, kmesh = kpts_to_kmesh(df_obj, df_obj.kpts)
    nkpt = nspc = len(kpts)
    assert numpy.prod(kmesh) == nkpt

    pcell = df_obj.cell
    nao = pcell.nao_nr()
    nip = inpv_kpt.shape[1]
    assert inpv_kpt.shape == (nkpt, nip, nao)

    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )
    assert phase.shape == (nspc, nkpt)

    t_kpt = numpy.asarray([xk.conj() @ xk.T for xk in inpv_kpt])
    assert t_kpt.shape == (nkpt, nip, nip)

    t_spc = kpt_to_spc(t_kpt, phase)
    assert t_spc.shape == (nspc, nip, nip)

    metx_kpt = spc_to_kpt(t_spc * t_spc, phase)

    eta_kpt = fswap.create_dataset("eta_kpt", shape=(nkpt, ngrid, nip), dtype=numpy.complex128) \
        if fswap is not None else numpy.zeros((nkpt, ngrid, nip), dtype=numpy.complex128)

    l = len("%s" % ngrid)
    info = f"aoR_loop: [% {l+2}d, % {l+2}d]"

    aoR_loop = df_obj.aoR_loop(grids, kpts, 0, blksize=df_obj.blksize)
    for ig, (ao_kpt, g0, g1) in enumerate(aoR_loop):
        t0 = (process_clock(), perf_counter())

        # input: ao_kpt: (nkpt, nao, ngrid), inpv_kpt: (nkpt, nao, nip)
        # output: eta_kpt: (nkpt, ngrid, nip)
        t_kpt = numpy.asarray([fk.conj() @ xk.T for fk, xk in zip(ao_kpt[0], inpv_kpt)])
        assert t_kpt.shape == (nkpt, g1 - g0, nip)
        t_spc = kpt_to_spc(t_kpt, phase)
        
        eta_spc_g0g1 = t_spc ** 2
        eta_kpt_g0g1 = spc_to_kpt(eta_spc_g0g1, phase).conj()
        t_kpt = None
        t_spc = None

        eta_kpt[:, g0:g1, :] += eta_kpt_g0g1
        eta_spc_g0g1 = None
        eta_kpt_g0g1 = None

        t1 = log.timer(info % (g0, g1), *t0)

    return metx_kpt, eta_kpt

def get_kern(df_obj, eta_q=None, kpt_q=None):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())
    
    kpts, kmesh = kpts_to_kmesh(df_obj, df_obj.kpts)
    nkpt = len(kpts)
    pcell = df_obj.cell
    nao = pcell.nao_nr()

    wrap_around = df_obj.wrap_around
    scell, phase = get_phase(
        pcell, kpts, kmesh=kmesh,
        wrap_around=wrap_around
    )
    nspc = phase.shape[0]
    assert phase.shape == (nspc, nkpt)

    grids = df_obj.grids
    assert grids is not None
    mesh = grids.mesh
    coord = grids.coords
    ngrid = coord.shape[0]

    nip = eta_q.shape[1]
    assert eta_q.shape == (ngrid, nip)

    kern_q = numpy.zeros((nip, nip), dtype=numpy.complex128)
    vg = pbctools.get_coulG(pcell, k=kpt_q, mesh=mesh)

    t = numpy.dot(coord, kpt_q)
    f = numpy.exp(-1j * t)
    assert f.shape == (ngrid, )

    v_q = fft(eta_q.T * f, mesh) * vg
    v_q *= pcell.vol / ngrid

    w_q = ifft(v_q, mesh) * f.conj()
    assert w_q.shape == (nip, ngrid)

    kern_q = numpy.dot(w_q, eta_q.conj())
    assert kern_q.shape == (nip, nip)
    return kern_q

class InterpolativeSeparableDensityFitting(FFTDF):
    wrap_around = False

    _fswap = None
    _isdf = None
    _isdf_to_save = None

    _coul_kpt = None
    _inpv_kpt = None

    blksize = None
    tol = 1e-10
    c0 = 10.0

    _keys = {"tol", "c0"}

    def __init__(self, cell, kpts=numpy.zeros((1, 3))):
        FFTDF.__init__(self, cell, kpts)
        from tempfile import NamedTemporaryFile
        fswap = NamedTemporaryFile(dir=lib.param.TMPDIR)
        self._fswap = fswap.name

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("mesh = %s (%d PWs)", self.mesh, numpy.prod(self.mesh))
        log.info("len(kpts) = %d", len(self.kpts))
        return self
    
    build = build
    
    def aoR_loop(self, grids=None, kpts=None, deriv=0, blksize=None):
        if grids is None:
            grids = self.grids
            cell = self.cell
        else:
            cell = grids.cell

        if grids.non0tab is None:
            grids.build(with_non0tab=True)

        if kpts is None:
            kpts = self.kpts
        kpts = numpy.asarray(kpts)

        assert cell.dimension == 3

        max_memory = max(2000, self.max_memory - current_memory()[0])

        ni = self._numint
        nao = cell.nao_nr()
        p1 = 0

        block_loop = ni.block_loop(
            cell, grids, nao, deriv, kpts,
            max_memory=max_memory,
            blksize=blksize
            )
        
        for ao_etc_kpt in block_loop:
            coords = ao_etc_kpt[4]
            p0, p1 = p1, p1 + coords.shape[0]
            yield ao_etc_kpt, p0, p1
    
    @line_profiler.profile
    def get_inpx(self, g0=None, c0=None, tol=None):
        log = logger.new_logger(self, self.verbose)
        t0 = (process_clock(), perf_counter())

        if g0 is None:
            assert c0 is not None
            nip = self.cell.nao_nr() * c0

            from pyscf.pbc.tools.pbc import mesh_to_cutoff
            lv = self.cell.lattice_vectors()
            k0 = mesh_to_cutoff(lv, [int(numpy.power(nip, 1/3) + 1)] * 3)
            k0 = max(k0)

            from pyscf.pbc.tools.pbc import cutoff_to_mesh
            g0 = self.cell.gen_uniform_grids(cutoff_to_mesh(lv, k0))

        if tol is None:
            tol = self.tol
        
        pcell = self.cell
        ng = len(g0)

        x0 = pcell.pbc_eval_gto("GTOval", g0)
        m0 = numpy.dot(x0.conj(), x0.T) ** 2

        from pyscf.lib.scipy_helper import pivoted_cholesky
        tol2 = tol ** 2
        chol, perm, rank = pivoted_cholesky(m0, tol=tol2)

        nip = pcell.nao_nr() * c0 if c0 is not None else rank
        nip = int(nip)
        mask = perm[:nip]

        nip = mask.shape[0]
        log.info("Pivoted Cholesky rank = %d, estimated error = %6.2e", rank, chol[nip-1, nip-1])
        log.info("Parent grid size = %d, selected grid size = %d", ng, nip)

        inpx = g0[mask]
        t1 = log.timer("interpolating functions", *t0)
        return inpx
    
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        assert omega is None and exxdiv is None

        from pyscf.pbc.df.aft import _check_kpts
        kpts, is_single_kpt = _check_kpts(self, kpts)
        # if is_single_kpt:
        #     raise NotImplementedError
        
        vj = vk = None
        if with_k:
            from fft_isdf_jk import get_k_kpts
            vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            from pyscf.pbc.df.fft_jk import get_j_kpts
            # from fft_isdf_jk import get_j_kpts
            vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)

        return vj, vk
    
    def lstsq(self, a, b, tol=1e-10, verbose=0):
        # make sure a is Hermitian
        log = logger.new_logger(self, verbose)
        assert numpy.allclose(a, a.conj().T)

        u, s, vh = scipy.linalg.svd(a, full_matrices=False)
        uh = u.conj().T
        v = vh.conj().T

        r = s[None, :] * s[:, None]
        m = abs(r) > tol
        t = (uh @ b @ u) * m / r
        x = v @ t @ vh

        if self.verbose > logger.DEBUG1:
            err = abs(a @ x @ a.conj().T - b).max()
            log.debug1(
                "Solving least square problem: rank = %3d / %3d, error = %6.2e", 
                int(m.sum() / m.shape[0]), a.shape[0], err
                )
        return x

ISDF = FFTISDF = InterpolativeSeparableDensityFitting

if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "../data/vasp")
    assert os.path.exists(DATA_PATH), f"DATA_PATH {DATA_PATH} does not exist"

    from utils import cell_from_poscar
    cell = cell_from_poscar(os.path.join(DATA_PATH, "diamond-prim.vasp"))
    cell.basis = 'gth-dzvp-molopt-sr'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.unit = 'aa'
    cell.exp_to_discard = 0.1
    cell.max_memory = 2000
    cell.ke_cutoff = 20.0
    cell.build(dump_input=False)
    nao = cell.nao_nr()

    kmesh = [4, 4, 4]
    # kmesh = [2, 2, 2]
    nkpt = nspc = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    scf_obj.conv_tol = 1e-8
    dm_kpts = scf_obj.get_init_guess(key="minao")

    log = logger.new_logger(None, 5)

    vv = []
    ee = []
    cc = [5.0, 10.0, 15.0, 20.0]
    for c0 in cc:
        from pyscf.pbc.tools.pbc import cutoff_to_mesh
        lv = cell.lattice_vectors()
        g0 = cell.gen_uniform_grids(cutoff_to_mesh(lv, cell.ke_cutoff))

        scf_obj.with_df = ISDF(cell, kpts=kpts)
        scf_obj.with_df.verbose = 10
        scf_obj.with_df.tol = 1e-10
        scf_obj.with_df.max_memory = 2000

        df_obj = scf_obj.with_df
        inpx = df_obj.get_inpx(g0=g0, c0=c0, tol=1e-10)
        df_obj.build(inpx=inpx, verbose=10)

        vj, vk = df_obj.get_jk(dm_kpts)
        vv.append((vj, vk))
        ee.append(scf_obj.energy_tot(dm_kpts))

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 0
    scf_obj.with_df.dump_flags()
    scf_obj.with_df.check_sanity()

    vj_ref, vk_ref = scf_obj.with_df.get_jk(dm_kpts)
    e_ref = scf_obj.energy_tot(dm_kpts)

    print("-> FFTDF e_tot = %12.8f" % e_ref)
    for ic, c0 in enumerate(cc):
        print("-> FFTISDF c0 = %6s, ene_err = % 6.2e, vj_err = % 6.2e, vk_err = % 6.2e" % (c0, abs(ee[ic] - e_ref), abs(vv[ic][0] - vj_ref).max(), abs(vv[ic][1] - vk_ref).max()))

