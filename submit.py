import os, sys, shutil

def main(time="01:00:00", mem=200, ncpu=1, workdir=None, cmd=None, scr=None, 
         import_pyscf_forge=False, import_periodic_integrals=False):
    pwd = os.path.abspath(os.path.dirname(__file__))

    assert cmd is not None
    assert os.path.exists(scr), f"Script {scr} does not exist"

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # os.system("cp %s %s/main.py" % (scr, workdir))

    # add few lines to the beginning of the run.sh file
    lines = []
    with open(os.path.join(pwd, "src", "script", "run.sh"), "r") as f:
        lines = f.readlines()
        lines.insert(1, "#SBATCH --time=%s\n" % time)
        lines.insert(1, "#SBATCH --mem=%dGB\n" % mem)
        lines.insert(1, "#SBATCH --cpus-per-task=%d\n" % ncpu)
        lines.insert(1, "#SBATCH --job-name=%s\n" % workdir.split("work")[-1][1:])
        lines.insert(1, "#SBATCH --exclude=hpc-21-34\n")

    with open(os.path.join(workdir, "run.sh"), "w") as f:
        f.writelines(lines)
        f.write("\n\n")
        f.write("export PREFIX=%s\n" % pwd)
        f.write("export DATA_PATH=$PREFIX/data/\n")
        f.write("export PYTHONPATH=$PREFIX/src/:$PYTHONPATH\n")

        if import_pyscf_forge:
            f.write("export PYSCF_EXT_PATH=$HOME/packages/pyscf-forge/pyscf-forge-ning-isdf4/\n")

        if import_periodic_integrals:
            f.write("export PYTHONPATH=/home/junjiey/packages/libdmet/libdmet2-main/:$PYTHONPATH\n")
            # f.write("export PYTHONPATH=/home/junjiey/packages/PeriodicIntegrals/PeriodicIntegrals-junjie-benchmark/src/:$PYTHONPATH\n")
            # f.write("export PYTHONPATH=/home/junjiey/packages/PeriodicIntegrals/PeriodicIntegrals-junjie-benchmark/:$PYTHONPATH\n")

        f.write("cp %s %s/main.py\n" % (scr, workdir))
        f.write(" ".join(cmd) + "\n")

    os.chdir(workdir)
    os.system("sbatch run.sh")
    os.chdir(pwd)

if __name__ == "__main__":
    mem = 80
    def run(name, df, ncpu=1, ke_cutoff=100.0, chk_path=None):
        cmd = [f"python main.py --name={name}"]
        cmd += [f"--ke_cutoff={ke_cutoff}"]
        cmd += [f"--exxdiv=None"]
        cmd += [f"--df={df}"]
        cmd += [f"--chk_path={chk_path}"]

        script = f"run-scf-gamma"
        base_dir = os.path.abspath(os.path.dirname(__file__))
        script_path = f"{base_dir}/src/script/{script}.py"

        if ke_cutoff is not None:
            work_subdir = f"work/{script}/{name}/{df}/{ke_cutoff:.0f}"
        else:
            assert df == "gdf"
            work_subdir = f"work/{script}/{name}/{df}"

        if os.path.exists(os.path.join(base_dir, work_subdir)):
            print(f"Work directory {work_subdir} already exists, deleting it")
            shutil.rmtree(os.path.join(base_dir, work_subdir))

        main(
            time="4:00:00", mem=mem, ncpu=ncpu,
            workdir=os.path.join(base_dir, work_subdir),
            cmd=cmd, scr=script_path, import_pyscf_forge=True
        )

    for ke_cutoff in [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]:
        path = "/central/scratch/yangjunjie//run-scf-gamma/diamond-conv/gdf/47958841/scf.h5"
        run("diamond-conv", "fftdf", ncpu=20, ke_cutoff=ke_cutoff, chk_path=path)

        path = "/central/scratch/yangjunjie//run-scf-gamma/diamond-prim/gdf/47958842/scf.h5"
        run("diamond-prim", "fftdf", ncpu=20, ke_cutoff=ke_cutoff, chk_path=path)

    for ke_cutoff in [160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0]:
        path = "/central/scratch/yangjunjie//run-scf-gamma/nio-conv/gdf/47958839/scf.h5"
        run("nio-conv", "fftdf", ncpu=20, ke_cutoff=ke_cutoff, chk_path=path)

        path = "/central/scratch/yangjunjie//run-scf-gamma/nio-prim/gdf/47958840/scf.h5"
        run("nio-prim", "fftdf", ncpu=20, ke_cutoff=ke_cutoff, chk_path=path)
