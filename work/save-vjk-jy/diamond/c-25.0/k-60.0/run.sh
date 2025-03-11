#!/bin/bash
#SBATCH --exclude=hpc-21-34
#SBATCH --job-name=save-vjk-jy/diamond/c-25.0/k-60.0
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
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

python -c "from pyscf import __config__; fft_engine = getattr(__config__, 'pbc_tools_pbc_fft_engine', 'NUMPY+BLAS'); print('fft_engine = %s' % fft_engine)"


export PREFIX=/central/home/junjiey/work/benchmark-fftisdf
export DATA_PATH=$PREFIX/data/
export PYTHONPATH=$PREFIX/src/:$PYTHONPATH
cp /central/home/junjiey/work/benchmark-fftisdf/src/script/save-vjk-jy.py /central/home/junjiey/work/benchmark-fftisdf/work/save-vjk-jy/diamond/c-25.0/k-60.0/main.py
python main.py --name diamond --c0=25.0 --k0=60.0 --ke_cutoff=70.0 --exxdiv=None 
