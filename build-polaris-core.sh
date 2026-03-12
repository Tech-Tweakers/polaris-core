#!/bin/bash
set -e

LLAMA_DIR=/home/atorres/dev/polaris/llama.cpp-latest
POLARIS_DIR=/home/atorres/dev/polaris/polaris-core
V3_API=/home/atorres/dev/polaris/polaris-v3-api/polaris_api/polaris_core

# ============================================================
# STEP 1: Compile llama.cpp-latest (GPU + CPU)
# ============================================================

echo "🔧 [1/4] Compilando llama.cpp-latest (GPU)..."
cd "$LLAMA_DIR"
rm -rf build-gpu && mkdir build-gpu && cd build-gpu
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF
make -j$(nproc)

echo "🔧 [2/4] Compilando llama.cpp-latest (CPU)..."
cd "$LLAMA_DIR"
rm -rf build-cpu && mkdir build-cpu && cd build-cpu
cmake .. \
  -DGGML_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF
make -j$(nproc)

# ============================================================
# STEP 2: Compile polaris_core (GPU + CPU)
# ============================================================

echo "🔧 [3/4] Compilando polaris_core (GPU + CPU)..."
cd "$POLARIS_DIR"

# GPU build
rm -rf build-gpu && mkdir build-gpu && cd build-gpu
cmake "$POLARIS_DIR" -DPOLARIS_ENABLE_CUDA=ON
make -j$(nproc)

# CPU build
cd "$POLARIS_DIR"
rm -rf build-cpu && mkdir build-cpu && cd build-cpu
cmake "$POLARIS_DIR" -DPOLARIS_ENABLE_CUDA=OFF
make -j$(nproc)

# ============================================================
# STEP 3: Copy artifacts to polaris-v3-api
# ============================================================

echo "📦 [4/4] Copiando artefatos para polaris-v3-api..."

# GPU
DEST_GPU="$V3_API/gpu"
mkdir -p "$DEST_GPU"
cp "$POLARIS_DIR/build-gpu/polaris_core.cpython-"*.so "$DEST_GPU/"
cp "$LLAMA_DIR"/build-gpu/bin/libllama.so*   "$DEST_GPU/"
cp "$LLAMA_DIR"/build-gpu/bin/libggml*.so*   "$DEST_GPU/"

# CPU
DEST_CPU="$V3_API/cpu"
mkdir -p "$DEST_CPU"
cp "$POLARIS_DIR/build-cpu/polaris_core.cpython-"*.so "$DEST_CPU/"
cp "$LLAMA_DIR"/build-cpu/bin/libllama.so*   "$DEST_CPU/"
cp "$LLAMA_DIR"/build-cpu/bin/libggml*.so*   "$DEST_CPU/"

echo ""
echo "✅ Build completo!"
echo "   GPU → $DEST_GPU"
echo "   CPU → $DEST_CPU"
