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
        f.write(cmd + "\n")

    os.chdir(workdir)
    os.system("sbatch run.sh")
    os.chdir(pwd)

if __name__ == "__main__":
    ncpu = 1
    mem = ncpu * 20

    info = {
        "diamond": {
            "ke_cutoff": 70.0,
            "rcut_epsilon": 5e-4,
            "ke_epsilon": 1e-1,
            "isdf_thresh": 1e-3,
            "cnz": 25.0,
            "rela_qr": 1e-5,
            "aoR_cutoff": 1e-7,
            "cjy": 10.0,
            "exxdiv": None,
        },
    }

    base_dir = os.path.abspath(os.path.dirname(__file__))
    df = "fftdf"
    def run(system):
        cmd = (
            f"python main.py --name {system} "
            f"--ke_cutoff={info[system]['ke_cutoff']} "
            f"--exxdiv={info[system]['exxdiv']} "
        )

        script = f"save-vjk-{df}"
        script_path = f"{base_dir}/src/script/{script}.py"

        work_subdir = f"work/{script}/{system}/"
        if os.path.exists(os.path.join(base_dir, work_subdir)):
            print(f"Work directory {work_subdir} already exists, deleting it")
            shutil.rmtree(os.path.join(base_dir, work_subdir))

        main(
            time="01:00:00", mem=mem, ncpu=ncpu,
            workdir=os.path.join(base_dir, work_subdir),
            cmd=cmd, scr=script_path, import_pyscf_forge=True
        )
    run("diamond")

    # df = "nz"
    # def run(system):
    #     cmd = (
    #         f"python main.py --name {system} "
    #         f"--c0={info[system]['cnz']} --rela_qr={info[system]['rela_qr']} "
    #         f"--aoR_cutoff={info[system]['aoR_cutoff']} "
    #         f"--ke_cutoff={info[system]['ke_cutoff']} "
    #         f"--exxdiv={info[system]['exxdiv']} "
    #     )

    #     script = f"save-vjk-{df}"
    #     script_path = f"{base_dir}/src/script/{script}.py"

    #     work_subdir = f"work/{script}/{system}/"
    #     if os.path.exists(os.path.join(base_dir, work_subdir)):
    #         print(f"Work directory {work_subdir} already exists, deleting it")
    #         shutil.rmtree(os.path.join(base_dir, work_subdir))

    #     main(
    #         time="01:00:00", mem=mem, ncpu=ncpu,
    #         workdir=os.path.join(base_dir, work_subdir),
    #         cmd=cmd, scr=script_path, import_pyscf_forge=True
    #     )
    # run("diamond")

    # df = "kori"
    # def run(system):
    #     cmd = (
    #         f"python main.py --name {system} "
    #         f"--ke_cutoff={info[system]['ke_cutoff']} "
    #         f"--rcut_epsilon={info[system]['rcut_epsilon']} "
    #         f"--ke_epsilon={info[system]['ke_epsilon']} "
    #         f"--isdf_thresh={info[system]['isdf_thresh']} "
    #         f"--exxdiv={info[system]['exxdiv']} "
    #     )

    #     script = f"save-vjk-{df}"
    #     script_path = f"{base_dir}/src/script/{script}.py"

    #     work_subdir = f"work/{script}/{system}/"
    #     if os.path.exists(os.path.join(base_dir, work_subdir)):
    #         print(f"Work directory {work_subdir} already exists, deleting it")
    #         shutil.rmtree(os.path.join(base_dir, work_subdir))

    #     main(
    #         time="01:00:00", mem=mem, ncpu=ncpu,
    #         workdir=os.path.join(base_dir, work_subdir),
    #         cmd=cmd, scr=script_path, import_pyscf_forge=False,
    #         import_periodic_integrals=True
    #     )
    # run("diamond")

    df = "jy"
    for c in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
        for k0 in [20.0, 40.0, 60.0, 80.0, 100.0]:
            def run(system):
                cmd = (
                    f"python main.py --name {system} "
                    f"--c0={c} --k0={k0} "
                    f"--ke_cutoff={info[system]['ke_cutoff']} "
                    f"--exxdiv={info[system]['exxdiv']} "
                )

                script = f"save-vjk-{df}"
                script_path = f"{base_dir}/src/script/{script}.py"

                work_subdir = f"work/{script}/{system}/c-{c}/k-{k0}"
                if os.path.exists(os.path.join(base_dir, work_subdir)):
                    print(f"Work directory {work_subdir} already exists, deleting it")
                    shutil.rmtree(os.path.join(base_dir, work_subdir))

                main(
                    time="01:00:00", mem=mem, ncpu=ncpu,
                    workdir=os.path.join(base_dir, work_subdir),
                    cmd=cmd, scr=script_path, import_pyscf_forge=False,
                )

            run("diamond")
