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
    mem = 200
    def run(name, df, ncpu=1, ke_cutoff=100.0, chk_path=None):
        cmd = [f"python main.py --name={name}"]
        cmd += [f"--ke_cutoff={ke_cutoff}"]
        cmd += [f"--exxdiv=None"]
        cmd += [f"--mesh=1,1,1 --df={df}"]
        cmd += [f"--chk_path={chk_path}"]
        cmd += [f"--rcut_epsilon={1e-5}"]
        cmd += [f"--ke_epsilon={1e-2}"]
        cmd += [f"--isdf_thresh={5e-4}"]

        script = f"run-scf"
        base_dir = os.path.abspath(os.path.dirname(__file__))
        script_path = f"{base_dir}/src/script/{script}.py"

        work_subdir = f"work/{script}/{name}/{df}/{ke_cutoff:.0f}"
        if os.path.exists(os.path.join(base_dir, work_subdir)):
            print(f"Work directory {work_subdir} already exists, deleting it")
            shutil.rmtree(os.path.join(base_dir, work_subdir))

        main(
            time="01:00:00", mem=mem, ncpu=ncpu,
            workdir=os.path.join(base_dir, work_subdir),
            cmd=cmd, scr=script_path, import_pyscf_forge=True
        )

    chk_path = "/central/scratch/yangjunjie/run-scf/diamond/gdf/47941657/scf.h5"
    run("diamond", "fftisdf-ks", ncpu=1, ke_cutoff=100.0, chk_path=chk_path)
