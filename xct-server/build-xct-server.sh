#!/bin/bash
# ============================================================
# Build script exclusivo para o xct-server
# Compila polaris_core contra llama.cpp-latest com CUDA 12.2
# ============================================================
set -e

POLARIS_CORE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="/home/atorres/dev/polaris/llama.cpp-latest"
XCT_SERVER_DIR="/home/atorres/dev/polaris/xct-server"
POLARIS_DIR="$LLAMA_ROOT/examples/polaris"

echo "============================================"
echo "  Polaris Core — xct-server build"
echo "============================================"
echo "  LLAMA_ROOT:      $LLAMA_ROOT"
echo "  POLARIS_CORE:    $POLARIS_CORE_DIR"
echo "  XCT_SERVER_DIR:  $XCT_SERVER_DIR"
echo "============================================"

# ── Step 1: Recompila llama.cpp-latest com CUDA 12.2 ──
echo ""
echo "[1/4] Compilando llama.cpp-latest (CUDA 12.2)..."
cd "$LLAMA_ROOT"
rm -rf build-gpu && mkdir build-gpu && cd build-gpu
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.2
make -j$(nproc)

# ── Step 2: Copia arquivos do polaris-core para examples/polaris ──
echo ""
echo "[2/4] Copiando polaris_bind.cpp e CMakeLists.txt para examples/polaris..."
mkdir -p "$POLARIS_DIR"
cp "$POLARIS_CORE_DIR/polaris_bind.cpp"        "$POLARIS_DIR/polaris_bind.cpp"
cp "$POLARIS_CORE_DIR/xct-server/CMakeLists.txt" "$POLARIS_DIR/CMakeLists.txt"

# ── Step 3: Compila polaris_core ──
echo ""
echo "[3/4] Compilando polaris_core (CUDA)..."
cd "$POLARIS_DIR"
rm -rf build && mkdir build && cd build
cmake .. -DPOLARIS_ENABLE_CUDA=ON
make -j$(nproc)

# ── Step 4: Copia .so + libs para xct-server ──
echo ""
echo "[4/4] Copiando artefatos para xct-server..."
DEST="$XCT_SERVER_DIR"
LIBS_DEST="$DEST/polaris_libs"
mkdir -p "$LIBS_DEST"

# modulo python
cp "$POLARIS_DIR/build/polaris_core.cpython-"*.so "$DEST/polaris_core.so"

# libs compartilhadas (inclui versionadas .so.0, .so.0.x.y)
cp "$LLAMA_ROOT/build-gpu/bin/libllama.so"*    "$LIBS_DEST/" 2>/dev/null || true
cp "$LLAMA_ROOT/build-gpu/bin/libggml"*.so*    "$LIBS_DEST/" 2>/dev/null || true

# configura RPATH no .so
if command -v patchelf &> /dev/null; then
  echo "  Configurando RPATH com patchelf..."
  patchelf --set-rpath '$ORIGIN/polaris_libs' "$DEST/polaris_core.so"
  for lib in "$LIBS_DEST"/lib*.so; do
    patchelf --set-rpath '$ORIGIN' "$lib" 2>/dev/null || true
  done
fi

echo ""
echo "============================================"
echo "  Build concluido!"
echo "============================================"
echo ""
echo "  Artefatos em: $DEST"
ls -lh "$DEST/polaris_core.so"
echo ""
echo "  Libs em: $LIBS_DEST"
ls -lh "$LIBS_DEST/"
echo ""
echo "  Para rodar o xct-server:"
echo "    cd $DEST"
echo "    python xct_server.py"
echo "============================================"
