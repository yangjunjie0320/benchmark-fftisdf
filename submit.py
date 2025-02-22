import os, sys, shutil

def main(name=None, k0=None, c0=None, time="01:00:00", mem=2000, ncpu=1):
    pwd = os.path.abspath(os.path.dirname(__file__))

    dir_name = os.path.join(pwd, "%s/c0-%6.2f-k0-%6.2f/" % (name, c0, k0))
    if os.path.exists(dir_name):
        print(f"Directory {dir_name} already exists, deleting ...")
        shutil.rmtree(dir_name)

    os.makedirs(dir_name)

    os.system("cp %s/src/check-init-energy.py %s/main.py" % (pwd, dir_name))
    # os.system("cp %s/src/run.sh %s/run.sh" % (pwd, dir_name))

    # add few lines to the beginning of the run.sh file
    lines = []
    with open("%s/src/run.sh" % pwd, "r") as f:
        lines = f.readlines()
        lines.insert(1, "#SBATCH --time=%s\n" % time)
        lines.insert(1, "#SBATCH --mem=%dGB\n" % mem)
        lines.insert(1, "#SBATCH --cpus-per-task=%d\n" % ncpu)
        lines.insert(1, "#SBATCH --job-name=%s-c0-%6.2f-k0-%6.2f\n" % (name, c0, k0))

    with open("%s/run.sh" % dir_name, "w") as f:
        f.writelines(lines)
        f.write("python main.py --name %s --k0 %6.2f --c0 %6.2f\n" % (name, k0, c0))

    os.system("sbatch %s/run.sh" % dir_name)

if __name__ == "__main__":
    for k0 in [10.0, 20.0, 40.0]:
        for c0 in [5.0, 10.0, 15.0, 20.0]:
            main(name="diamond", k0=k0, c0=c0, time="04:00:00")
            main(name="nio", k0=k0, c0=c0, time="03:00:00")
            main(name="cco", k0=k0, c0=c0, time="02:00:00")
