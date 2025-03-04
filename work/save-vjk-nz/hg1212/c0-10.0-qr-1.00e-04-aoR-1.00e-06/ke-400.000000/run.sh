#!/bin/bash
#SBATCH --exclude=hpc-21-34
#SBATCH --job-name=save-vjk-nz/hg1212/c0-10.0-qr-1.00e-04-aoR-1.00e-06/ke-400.000000/
#SBATCH --cpus-per-task=64
#SBATCH --mem=384GB
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=changroup_standingres
#SBATCH --constraint=icelake

echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "Start time = $(date)"

# Load environment configuration
source /home/junjiey/anaconda3/bin/activate fftisdf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MKL_NUM_THREADS=1;

export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE;
echo OMP_NUM_THREADS = $OMP_NUM_THREADS
echo MKL_NUM_THREADS = $MKL_NUM_THREADS
echo PYSCF_MAX_MEMORY = $PYSCF_MAX_MEMORY

export TMP=/central/scratch/yangjunjie/
export TMPDIR=$TMP/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR

mkdir -p $TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
ln -s $PYSCF_TMPDIR tmp

echo ""; which python
python -c "import pyscf; print(pyscf.__version__)"
python -c "import scipy; print(scipy.__version__)"
python -c "import numpy; print(numpy.__version__)"


export PREFIX=/central/home/junjiey/work/benchmark-fftisdf
export DATA_PATH=$PREFIX/data/
export PYTHONPATH=$PREFIX/src/:$PYTHONPATH
export PYSCF_EXT_PATH=$HOME/packages/pyscf-forge/pyscf-forge-ning-isdf4/
python main.py --name hg1212 --c0=10.0 --rela_qr=0.0001 --aoR_cutoff=1e-06 --ke_cutoff=400.0
