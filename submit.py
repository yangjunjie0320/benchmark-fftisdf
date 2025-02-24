import os, sys, shutil

def main(time="01:00:00", mem=200, ncpu=1, workdir=None, cmd=None, scr=None):
    pwd = os.path.abspath(os.path.dirname(__file__))

    assert cmd is not None
    assert os.path.exists(scr), f"Script {scr} does not exist"

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    os.system("cp %s %s/main.py" % (scr, workdir))

    # add few lines to the beginning of the run.sh file
    lines = []
    with open(os.path.join(pwd, "src", "script", "run.sh"), "r") as f:
        lines = f.readlines()
        lines.insert(1, "#SBATCH --time=%s\n" % time)
        lines.insert(1, "#SBATCH --mem=%dGB\n" % mem)
        lines.insert(1, "#SBATCH --cpus-per-task=%d\n" % ncpu)
        lines.insert(1, "#SBATCH --job-name=%s\n" % workdir.split("work")[-1][1:])

    with open(os.path.join(workdir, "run.sh"), "w") as f:
        f.writelines(lines)
        f.write("\n\n")
        f.write("export PREFIX=%s\n" % pwd)
        f.write("export DATA_PATH=$PREFIX/data/\n")
        f.write("export PYTHONPATH=$PREFIX/src/:$PYTHONPATH\n")
        f.write(cmd + "\n")

    os.chdir(workdir)
    os.system("sbatch run.sh")
    os.chdir(pwd)

if __name__ == "__main__":
    ncpu = 32
    mem = ncpu * 8

    # Define k-point meshes and system names
    meshes = [
        [1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2],
        [2, 2, 4], [2, 4, 4], [4, 4, 4]
    ]
    systems = ["nio", "cco", "diamond"]
    
    # Paths and parameters that remain constant across iterations
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    for kmesh in meshes:
        for system_name in systems:
            # Construct command and working directory
            cmd = (
                f"python main.py --name={system_name} "
                f"--kmesh={'-'.join(map(str, kmesh))}"
            )

            script = "time-vjk-fftisdf-jy"
            script_path = f"{base_dir}/src/script/{script}.py"

            work_subdir = f"work/{script}/{system_name}/{'-'.join(map(str, kmesh))}/"
            
            # Submit job with parameters
            main(
                time="04:00:00", mem=mem, ncpu=ncpu,
                workdir=os.path.join(base_dir, work_subdir),
                cmd=cmd, scr=script_path
            )

            ke_cutoff = 20.0
            cmd = (
                f"python main.py --name={system_name} "
                f"--kmesh={'-'.join(map(str, kmesh))} "
                f"--ke_cutoff={ke_cutoff:.2f}"
            )

            script = "check-vjk-fftisdf-jy"
            script_path = f"{base_dir}/src/script/{script}.py"

            work_subdir = f"work/{script}/{system_name}/{'-'.join(map(str, kmesh))}/"

            # Submit job with parameters
            main(
                time="04:00:00", mem=mem, ncpu=ncpu,
                workdir=os.path.join(base_dir, work_subdir),
                cmd=cmd, scr=script_path
            )