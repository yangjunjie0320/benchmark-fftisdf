"""
[Step 1]: download Ning's ISDF code: https://github.com/NingZhang1/pyscf-forge/tree/ning_isdf4 and change to the branch
```bash
    git clone https://github.com/NingZhang1/pyscf-forge.git pyscf-forge-ning-isdf4    
    cd pyscf-forge-ning-isdf4
    git fetch
    git checkout ning_isdf4
```

[Step 2]: add the path to PYSCF_EXT
```bash
    export PYSCF_EXT_PATH=$(readlink -f pyscf-forge-ning-isdf4)
```

[Step 3]: run the script
```bash
    python fftisdf-nz.py
```
"""

import os, sys, numpy, scipy
import pyscf
from pyscf.pbc import gto
cell = gto.Cell()
cell.atom = [
    ["C", (0.0, 0.0, 0.0)],
    ["C", (0.8917, 0.8917, 0.8917)],
    ["C", (1.7834, 1.7834, 0.0)],
    ["C", (2.6751, 2.6751, 0.8917)],
    ["C", (1.7834, 0.0, 1.7834)],
    ["C", (2.6751, 0.8917, 2.6751)],
    ["C", (0.0, 1.7834, 1.7834)],
    ["C", (0.8917, 2.6751, 2.6751)],
]
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = numpy.eye(3) * 3.5668
cell.ke_cutoff = 70.0
cell.build()

from pyscf.pbc import scf
mf = scf.RHF(cell)
mf.exxdiv = None
dm0 = mf.get_init_guess(key='minao')
vj_ref, vk_ref = mf.with_df.get_jk(dm0, with_j=True, with_k=True, exxdiv=None)

from pyscf.isdf import isdf_local
mf.with_df = isdf_local.ISDF_Local(cell, aoR_cutoff=1e-8, direct=False, limited_memory=False, with_robust_fitting=False)
mf.with_df.build(c=20.0, rela_cutoff=1e-4, group=None)
vj_sol, vk_sol = mf.with_df.get_jk(dm0, with_j=True, with_k=True, exxdiv=None)
vj_sol = vj_sol.reshape(vj_ref.shape).real
vk_sol = vk_sol.reshape(vk_ref.shape).real

err_vj = abs(vj_sol - vj_ref).max()
err_vk = abs(vk_sol - vk_ref).max()
print(f"err_vj = {err_vj:6.2e}, err_vk = {err_vk:6.2e}")

print(f"vk_ref = ")
numpy.savetxt(cell.stdout, vk_ref[:10, :10], fmt='% 6.4f', delimiter=', ')
print(f"vk_sol = ")
numpy.savetxt(cell.stdout, vk_sol[:10, :10], fmt='% 6.4f', delimiter=', ')
