conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

conda install mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that mat
