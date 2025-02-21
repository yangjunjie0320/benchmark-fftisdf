import os, sys, numpy, scipy
API_KEY = os.getenv("MP_API_KEY", None)
TMPDIR = os.getenv("TMPDIR", "/tmp")
assert os.path.exists(TMPDIR), f"TMPDIR {TMPDIR} does not exist"

def download_poscar(mid: str, name=None, path=None, is_conventional=False,
                    supercell_factor=None):
    from mp_api.client import MPRester as M
    if path is None:
        path = TMPDIR

    if name is None:
        name = f"{mid}"

    name = name + "-conv" if is_conventional else name + "-prim"
    # Initialize the Materials Project REST API client
    with M(API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(mid, conventional_unit_cell=is_conventional)
        tmp = os.path.join(TMPDIR, f"{name}.vasp")
        structure.to(filename=str(tmp), fmt="poscar")
        print(f"\nSuccessfully downloaded POSCAR for {mid} to {tmp}")
        
        import ase
        atoms = ase.io.read(tmp)
        poscar_path = os.path.join(path, f"{name}.vasp")

        if supercell_factor:
            atoms = atoms * supercell_factor
            poscar_path = os.path.join(path, f"{name}-{'x'.join(map(str, supercell_factor))}.vasp")
        
        atoms.write(poscar_path, format="vasp")
        return str(poscar_path)
    
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "vasp")
    assert os.path.exists(path), f"Path {path} does not exist"

    data  = [("mp-66", "diamond")]
    data += [("mp-19009", "nio")]
    data += [("mp-390", "tio2")]
    data += [("mp-4826", "cco")]
    data += [("mp-6879", "hg1212")]

    for m, n in data:
        is_conventional = False
        poscar_file = download_poscar(
            m, name=n, path=path, 
            is_conventional=is_conventional,
            supercell_factor=None
        )