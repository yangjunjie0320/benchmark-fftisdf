# FFT-ISDF Benchmark

This repository contains the code for the FFT-ISDF benchmark.

## Dependencies

The dependencies are listed in `fftisdf.yml`. You can install them by running:

```bash
conda env create -f fftisdf.yml
conda activate fftisdf
python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; numpy.show_config()"
python -c "import scipy; print(scipy.__version__)"
python -c "import pyscf; print(pyscf.__version__)"
python -c "import torch; print(torch.__version__)"
```

and activate the environment by running:

```bash
conda activate fftisdf
```

On a `slurm` cluster, you can use the following command to activate the environment:

```bash
source submit.sh
```

## Crystal Structure
All the crystal structures used in this benchmark are stored in `data/crystal_structures`. They are all downloaded from the Materials Project
with the `mp_api` package (refer to `/src/build.py` for more details).

