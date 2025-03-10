import os, sys
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.lib import logger, current_memory
from pyscf.lib.logger import process_clock, perf_counter

from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero

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
    cell = df_obj.cell
    from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
    kmesh = kpts_to_kmesh(cell, kpts)

    # [1] check if kpts is identical to df_obj.kpts
    assert numpy.allclose(kpts, df_obj.kpts)

    # [2] check if kmesh is identical to df_obj.kmesh
    assert numpy.allclose(kmesh, df_obj.kmesh)

    # [3] check if kpts is uniform
    assert numpy.allclose(kpts, cell.get_kpts(kmesh))

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

def lstsq(a, b, tol=1e-10):
    """
    Solve the least squares problem of the form:
        x = ainv @ b @ ainv.conj().T
    using SVD. In which a is not full rank, and
    ainv is the pseudo-inverse of a.

    Args:
        a: The matrix A.
        b: The matrix B.
        tol: The tolerance for the singular values.

    Returns:
        x: The solution to the least squares problem.
        rank: The rank of the matrix a.
    """

    # make sure a is Hermitian
    assert numpy.allclose(a, a.conj().T)

    u, s, vh = scipy.linalg.svd(a, full_matrices=False)
    uh = u.conj().T
    v = vh.conj().T

    r = s[None, :] * s[:, None]
    m = abs(r) > tol
    rank = m.sum() / m.shape[0]
    t = (uh @ b @ u) * m / r
    return v @ t @ vh, int(rank)

@line_profiler.profile
def build(df_obj, inpx=None, kpts=None, kmesh=None):
    """
    Build the FFT-ISDF object.
    
    Args:
        df_obj: The FFT-ISDF object to build.
    """
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())

    cell = df_obj.cell
    assert numpy.allclose(cell.get_kpts(kmesh), kpts)
    nkpt = len(kpts)

    nip = inpx.shape[0]
    assert inpx.shape == (nip, 3)
    nao = cell.nao_nr()

    inpv_kpt = cell.pbc_eval_gto("GTOval", inpx, kpts=kpts)
    inpv_kpt = numpy.asarray(inpv_kpt, dtype=numpy.complex128)
    assert inpv_kpt.shape == (nkpt, nip, nao)
    log.debug("nip = %d, nao = %d, cisdf = %6.2f", nip, nao, nip / nao)
    t1 = log.timer("get interpolating vectors")
    
    max_memory = max(2000, df_obj.max_memory - current_memory()[0])

    # metx_kpt: (nkpt, nip, nip), eta_kpt: (nkpt, ngrid, nip)
    # assume metx_kpt is a numpy.array, while eta_kpt is a hdf5 dataset
    metx_kpt, eta_kpt = get_lhs_and_rhs(
        df_obj, inpv_kpt,
        max_memory=max_memory,
        fswp=df_obj._fswap
    )
    ngrid = eta_kpt.shape[1]

    eta_kpt = numpy.asarray(eta_kpt)
    assert eta_kpt.shape == (nkpt, ngrid, nip)
    log.debug("eta_kpt.shape = %s", eta_kpt.shape)
    log.debug("Memory used for eta_kpt = %6.2e GB", eta_kpt.nbytes / 1e9)

    coul_kpt = []
    for q in range(nkpt):
        t0 = (process_clock(), perf_counter())

        metx_q = metx_kpt[q]
        assert metx_q.shape == (nip, nip)

        kern_q = get_kern(
            df_obj, eta_kpt=eta_kpt, q=q,
            fswp=df_obj._fswap,
            max_memory=max_memory
        )

        coul_q, rank = lstsq(metx_q, kern_q, tol=df_obj.tol)
        assert coul_q.shape == (nip, nip)
        
        coul_kpt.append(coul_q)
        log.timer("solving Coulomb kernel", *t0)
        log.info("Finished solving Coulomb kernel for q = %3d / %3d, rank = %d / %d", q + 1, nkpt, rank, nip)

    coul_kpt = numpy.asarray(coul_kpt)
    return inpv_kpt, coul_kpt

@line_profiler.profile
def get_lhs_and_rhs(df_obj, inpv_kpt, max_memory=2000, fswp=None):
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

    # [1] compute the metric tensor
    t_kpt = numpy.asarray([xk.conj() @ xk.T for xk in inpv_kpt])
    assert t_kpt.shape == (nkpt, nip, nip)

    t_spc = kpt_to_spc(t_kpt, phase)
    assert t_spc.shape == (nspc, nip, nip)

    metx_kpt = spc_to_kpt(t_spc * t_spc, phase)

    blksize = max(max_memory * 1e6 * 0.2 / (nkpt * nip * 16), 1)
    blksize = min(int(blksize), ngrid)

    log.debug("blksize = %d, ngrid = %d", blksize, ngrid)
    eta_kpt = None

    if blksize == ngrid:    
        eta_kpt = numpy.zeros((nkpt, ngrid, nip), dtype=numpy.complex128)
        log.debug("Use in-core for eta_kpt, memory used for eta_kpt = %6.2e GB", eta_kpt.nbytes / 1e9)
    else:
        eta_kpt = fswp.create_dataset("eta_kpt", shape=(nkpt, ngrid, nip), dtype=numpy.complex128)
        log.debug("Use out-core for eta_kpt, disk space used for eta_kpt = %6.2e GB", eta_kpt.nbytes / 1e9)
        log.debug("memory used for each block = %6.2e GB, max_memory = %6.2e GB", nkpt * nip * 16 * blksize / 1e9, max_memory / 1e3)

    assert eta_kpt is not None

    l = len("%s" % ngrid)
    info = f"aoR_loop: [% {l+2}d, % {l+2}d]"

    aoR_loop = df_obj.aoR_loop(grids, kpts, 0, blksize=blksize)
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

def get_kern(df_obj, eta_kpt=None, q=0, fswp=None, max_memory=2000):
    log = logger.new_logger(df_obj, df_obj.verbose)
    t0 = (process_clock(), perf_counter())
    
    kpts = df_obj.kpts
    kmesh = df_obj.kmesh
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

    nip = eta_kpt.shape[2]
    assert eta_kpt.shape == (nkpt, ngrid, nip)

    kern_q = numpy.zeros((nip, nip), dtype=numpy.complex128)
    vg = pbctools.get_coulG(pcell, k=kpts[q], mesh=mesh)

    t = numpy.dot(coord, kpts[q])
    f = numpy.exp(-1j * t)
    assert f.shape == (ngrid, )

    blksize = max(max_memory * 1e6 * 0.2 / (ngrid * 16), 1)
    blksize = min(int(blksize), nip)

    log.debug("\nCalculating Coulomb kernel with outcore method: q = %d / %d", q + 1, nkpt)
    log.debug("blksize = %d, nip = %d, max_memory = %6.2e GB", blksize, nip, max_memory / 1e3)
    log.debug("memory used for each block = %6.2e GB", ngrid * 16 * blksize / 1e9)

    i0, i1 = 0, 0
    j0, j1 = 0, 0

    for i0 in range(0, nip, blksize):
        eta_qi = eta_kpt[q, :, i0:(i0+blksize)]
        i1 = i0 + eta_qi.shape[1]
        assert eta_qi.shape == (ngrid, i1 - i0)

        v_qi = pbctools.fft(eta_qi.T * f, mesh) * vg
        v_qi *= pcell.vol / ngrid

        from pyscf.pbc.tools.pbc import ifft
        w_qi = ifft(v_qi, mesh) * f.conj()
        w_qi = w_qi.T
        assert w_qi.shape == (ngrid, i1 - i0)

        for j0 in range(0, nip, blksize):
            eta_qj = eta_kpt[q, :, j0:(j0+blksize)]
            j1 = j0 + eta_qj.shape[1]
            assert eta_qj.shape == (ngrid, j1 - j0)

            kern_q_ij = numpy.dot(w_qi.T, eta_qj.conj())
            assert kern_q_ij.shape == (i1 - i0, j1 - j0)

            kern_q[i0:i1, j0:j1] = kern_q_ij

    return kern_q

class InterpolativeSeparableDensityFitting(FFTDF):
    wrap_around = False

    _isdf = None
    _isdf_to_save = None

    _keys = ['_isdf', '_coul_kpt', '_inpv_kpt']

    def __init__(self, cell, kpts=numpy.zeros((1, 3)), kmesh=None):
        FFTDF.__init__(self, cell, kpts)

        # from pyscf.pbc.lib.kpts import KPoints
        # self.kpts = KPoints(cell, kpts)
        # self.kpts.build()

        self.kmesh = kmesh
        self.c0 = 10.0

        self.tol = 1e-10
        from pyscf.lib import H5TmpFile
        self._fswap = H5TmpFile()
        self._keys = ['_isdf', '_coul_kpt', '_inpv_kpt', '_fswap']

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info("mesh = %s (%d PWs)", self.mesh, numpy.prod(self.mesh))
        log.info("len(kpts) = %d", len(self.kpts))
        log.debug1("    kpts = %s", self.kpts)
        return self
    
    @line_profiler.profile
    def build(self, inpx=None):
        self.dump_flags()
        self.check_sanity()

        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        kmesh = kpts_to_kmesh(self.cell, self.kpts)
        self.kmesh = kmesh

        log = logger.new_logger(self, self.verbose)
        log.info("kmesh = %s", kmesh)
        t0 = (process_clock(), perf_counter())

        kpts = self.cell.get_kpts(kmesh)
        assert numpy.allclose(self.kpts, kpts), \
            "kpts mismatch, only uniform kpts is supported"

        if self._isdf is not None:
            pass

        if inpx is None:
            inpx = self.get_inpx(g0=None, c0=self.c0)

        inpv_kpt, coul_kpt = build(
            df_obj=self,
            inpx=inpx,
            kpts=kpts,
            kmesh=kmesh
        )

        self._inpv_kpt = inpv_kpt
        self._coul_kpt = coul_kpt

        if self._isdf_to_save is not None:
            self._isdf = self._isdf_to_save

        if self._isdf is None:
            import tempfile
            isdf = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self._isdf = isdf.name
        
        log.info("Saving FFTISDF results to %s", self._isdf)
        from pyscf.lib.chkfile import dump
        dump(self._isdf, "coul_kpt", coul_kpt)
        dump(self._isdf, "inpv_kpt", inpv_kpt)

        t1 = log.timer("building ISDF", *t0)
        return self
    
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
    
    def get_inpx(self, g0=None, c0=None, tol=None):
        log = logger.new_logger(self, self.verbose)

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
        t0 = numpy.dot(x0.conj(), x0.T)
        m0 = t0 * t0

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

    # kmesh = [4, 4, 4]
    kmesh = [2, 2, 2]
    nkpt = nspc = numpy.prod(kmesh)
    kpts = cell.get_kpts(kmesh)

    scf_obj = pyscf.pbc.scf.KRHF(cell, kpts=kpts)
    scf_obj.exxdiv = None
    scf_obj.conv_tol = 1e-8
    dm_kpts = scf_obj.get_init_guess(key="minao")

    log = logger.new_logger(None, 5)

    ee = []
    kk = [10.0, 20.0, 30.0, 40.0]
    for k0 in kk:
        from pyscf.pbc.tools.pbc import cutoff_to_mesh
        lv = cell.lattice_vectors()
        g0 = cell.gen_uniform_grids(cutoff_to_mesh(lv, k0))

        t0 = (process_clock(), perf_counter())
        c0 = 10.0
        scf_obj.with_df = ISDF(cell, kpts=kpts)
        scf_obj.with_df.verbose = 5
        scf_obj.with_df.tol = 1e-10
        scf_obj.with_df.max_memory = 2000
        df_obj = scf_obj.with_df
        inpx = df_obj.get_inpx(g0=g0, c0=c0)
        df_obj.build(inpx)
        t1 = log.timer("-> ISDF build", *t0)

        e_tot = scf_obj.kernel(dm_kpts)
        ee.append(e_tot)
        print("-> ISDF c0 = %6s, k0 = %6.2f, e_tot = %12.8f" % (c0, k0, e_tot))

    t0 = (process_clock(), perf_counter())
    scf_obj.with_df = FFTDF(cell, kpts)
    scf_obj.with_df.verbose = 0
    scf_obj.with_df.dump_flags()
    scf_obj.with_df.check_sanity()

    e_tot = scf_obj.kernel(dm_kpts)

    print("-> FFTDF e_tot = %12.8f" % e_tot)
    for ik, k0 in enumerate(kk):
        print("-> FFTISDF c0 = %6s, k0 = %6.2f, e_tot = %12.8f, err = % 6.2e" % (c0, k0, ee[ik], ee[ik] - e_tot))

