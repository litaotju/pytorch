export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

#   CMAKE_C_COMPILER="ccache /usr/bin/gcc" \
#   CC="ccache /usr/bin/gcc" \
#   CXX="ccache /usr/bin/g++" \

DEBUG=1 MAX_JOBS=96 USE_CUDA=0 SHELL=/usr/bin/zsh \
   python setup.py develop
