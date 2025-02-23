import os, sys, shutil

def main(name=None, time="01:00:00", mem=200, ncpu=1, cmd=None, scr=None):
    pwd = os.path.abspath(os.path.dirname(__file__))

    dir_name = os.path.join(pwd, "work/%s/c0-%.2f-k0-%.2f/" % (name, c0, k0))
    if os.path.exists(dir_name):
        print(f"Directory {dir_name} already exists, deleting ...")
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    assert cmd is not None
    assert os.path.exists(scr)

    os.system("cp %s %s/main.py" % (scr, dir_name))

    # add few lines to the beginning of the run.sh file
    lines = []
    with open("%s/src/run.sh" % pwd, "r") as f:
        lines = f.readlines()
        lines.insert(1, "#SBATCH --time=%s\n" % time)
        lines.insert(1, "#SBATCH --mem=%dGB\n" % mem)
        lines.insert(1, "#SBATCH --cpus-per-task=%d\n" % ncpu)
        lines.insert(1, "#SBATCH --job-name=%s-c0-%.2f-k0-%.2f\n" % (name, c0, k0))

    with open("%s/run.sh" % dir_name, "w") as f:
        f.writelines(lines)
        f.write("\n\n")
        f.write("export PREFIX=%s\n" % pwd)
        f.write("export DATA_PATH=$PREFIX/data/\n")
        f.write("export PYTHONPATH=$PREFIX/src/:$PYTHONPATH\n")
        f.write(cmd + "\n")

    os.chdir(dir_name)
    os.system("sbatch run.sh")
    os.chdir(pwd)

if __name__ == "__main__":
    ncpu = 20
    mem = ncpu * 8
    
    k0 = 60.0
    c0 = 20.0
    ke_cutoff = 10.0

    mm  = [[1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2]]
    mm += [[2, 2, 4], [2, 4, 4], [4, 4, 4]]

    for kmesh in mm:
        scr = "/home/junjiey/work/benchmark-fftisdf/src/check-vjk.py"
        cmd = "python main.py --name=%s --k0=%6.2f --c0=%6.2f "
        cmd += "--ke_cutoff=%6.2f --kmesh=%s" % (ke_cutoff, kmesh)
        main(name="nio", k0=k0, c0=c0, time="04:00:00", mem=mem, ncpu=ncpu, cmd=cmd, scr=scr)
