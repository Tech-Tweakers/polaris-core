# 1. Recompila llama.cpp com CUDA 12.2
cd /home/atorres/dev/polaris/llama.cpp
rm -rf build-gpu && mkdir build-gpu && cd build-gpu
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.2
make -j$(nproc)

# 2. Recompila polaris_core
cd /home/atorres/dev/polaris/llama.cpp/examples/polaris
rm -rf build && mkdir build && cd build
cmake .. -DPOLARIS_ENABLE_CUDA=ON
make -j$(nproc)

# 3. Copia
DEST=/home/atorres/dev/polaris/polaris-v3-api/polaris_api/polaris_core/gpu
cp /home/atorres/dev/polaris/llama.cpp/build-gpu/bin/libllama.so $DEST/
cp /home/atorres/dev/polaris/llama.cpp/build-gpu/bin/libggml*.so $DEST/
cp /home/atorres/dev/polaris/llama.cpp/examples/polaris/build/polaris_core.cpython-*.so $DEST/