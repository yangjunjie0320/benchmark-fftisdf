#!/bin/bash
#SBATCH --job-name=diamond-c0-15.00-k0-60.00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=changroup_standingres
#SBATCH --constraint=icelake

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
python main.py --name diamond --k0  60.00 --c0  15.00
