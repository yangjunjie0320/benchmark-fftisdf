import numpy
from pyscf.pbc import gto
import pyscf.pbc.scf
# Define the periodic NiO system
cell = gto.Cell()
cell.atom = '''
    Ni  0.000000  0.000000  0.000000
    O   0.000000  2.093169  2.093169
    Ni  2.093169  0.000000  2.093169
    O   2.093169  2.093169  0.000000
    Ni  2.093169  2.093169  2.093169
    O   0.000000  0.000000  2.093169
    Ni  0.000000  2.093169  0.000000
    O   2.093169  0.000000  0.000000
'''

cell.a = numpy.array([
    [4.186338, 0.000000, 0.000000],
    [0.000000, 4.186338, 0.000000],
    [0.000000, 0.000000, 4.186338]
])  # Lattice vectors
cell.basis = 'gth-dzvp-molopt-sr'
cell.pseudo = 'gth-pbe'
cell.spin = 0  # AFM system
cell.verbose = 10
cell.ke_cutoff = 200.0
cell.exp_to_discard = 0.2
cell.build()

# Use UHF to allow AFM ordering
mf = pyscf.pbc.scf.UHF(cell).density_fit()
mf.conv_tol = 1e-6
mf.max_cycle = 50
dm0 = mf.get_init_guess(key="minao")

# Define the AFM pattern manually
afm_guess = {
    "alpha": ["0 Ni 3dx2-y2", "2 Ni 3dx2-y2"],
    "beta":  ["4 Ni 3dx2-y2", "6 Ni 3dx2-y2"]
}

# Find AO indices for Ni 3dx2−y2 orbitals
alpha_indices = []
beta_indices = []

for key, ao_labels in afm_guess.items():
    for label in ao_labels:
        ao_idx = cell.search_ao_label(label)  # Find indices of specific AOs
        if key == "alpha":
            alpha_indices.extend(ao_idx)
        else:
            beta_indices.extend(ao_idx)

# Apply AFM ordering by modifying the density matrix
for ao_idx in alpha_indices:
    dm0[0][ao_idx, ao_idx] *= 1.0  # α (spin-up)
    dm0[1][ao_idx, ao_idx] *= 0.0  # Remove β (spin-down)

for ao_idx in beta_indices:
    dm0[0][ao_idx, ao_idx] *= 0.0  # Remove α (spin-up)
    dm0[1][ao_idx, ao_idx] *= 1.0  # β (spin-down)

ovlp = mf.get_ovlp()

from pyscf.scf.uhf import mulliken_spin_pop
mulliken_spin_pop(cell, dm0, ovlp)
mf.kernel(dm0=dm0)

dm = mf.make_rdm1()
mulliken_spin_pop(cell, dm, ovlp)
