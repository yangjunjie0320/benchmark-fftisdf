# FFT-ISDF Benchmark

There are two minimal working examples.

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

## Examples

### fftisdf-jy.py

This example only depends on `../fft_isdf.py`. The script has handled the path to `fft_isdf.py` by adding the path to `PYTHONPATH`. You can run the script by:

```bash
python fftisdf-jy.py
```

### fftisdf-nz.py

- Step 1: download Ning's ISDF code: https://github.com/NingZhang1/pyscf-forge/tree/ning_isdf4 and change to the branch
```bash
    git clone https://github.com/NingZhang1/pyscf-forge.git pyscf-forge-ning-isdf4    
    cd pyscf-forge-ning-isdf4
    git fetch; git checkout ning_isdf4
```

- Step 2: add the path to PYSCF_EXT
```bash
    export PYSCF_EXT_PATH=$(readlink -f pyscf-forge-ning-isdf4)
```

- Step 3: run the script
```bash
    python fftisdf-nz.py
```
