#!/bin/bash
# 📦 Script para copiar arquivos necessários do polaris_core para um projeto Python
# Uso: ./copy-to-project.sh /caminho/para/seu/projeto

if [ -z "$1" ]; then
    echo "❌ Erro: forneça o caminho do projeto Python de destino"
    echo "Uso: $0 /caminho/para/seu/projeto"
    exit 1
fi

DEST_DIR="$1"
BUILD_DIR="$(cd "$(dirname "$0")/../../build/bin" && pwd)"

if [ ! -d "$DEST_DIR" ]; then
    echo "❌ Erro: diretório de destino não existe: $DEST_DIR"
    exit 1
fi

echo "📦 Copiando arquivos do polaris_core para: $DEST_DIR"
echo "   Origem: $BUILD_DIR"
echo ""

# Criar diretório para as bibliotecas
LIB_DIR="$DEST_DIR/polaris_libs"
mkdir -p "$LIB_DIR"

# Copiar módulo Python
echo "✅ Copiando módulo Python..."
cp "$BUILD_DIR/polaris_core.cpython-"*.so "$DEST_DIR/polaris_core.so" 2>/dev/null || {
    echo "❌ Erro: não foi possível encontrar polaris_core.cpython-*.so"
    exit 1
}

# Copiar bibliotecas compartilhadas
echo "✅ Copiando bibliotecas compartilhadas..."
LIBS=(
    "libllama.so"
    "libggml.so"
    "libggml-base.so"
    "libggml-cpu.so"
    "libggml-cuda.so"
)

for lib in "${LIBS[@]}"; do
    if [ -f "$BUILD_DIR/$lib" ]; then
        cp "$BUILD_DIR/$lib" "$LIB_DIR/"
        echo "   ✓ $lib"
    else
        echo "   ⚠️  $lib não encontrado (pode não ser necessário)"
    fi
done

echo ""
echo "✅ Arquivos copiados com sucesso!"
echo ""
echo "📋 Estrutura criada:"
echo "   $DEST_DIR/"
echo "   ├── polaris_core.so          (módulo Python)"
echo "   └── polaris_libs/            (bibliotecas compartilhadas)"
echo "       ├── libllama.so"
echo "       ├── libggml.so"
echo "       ├── libggml-base.so"
echo "       ├── libggml-cpu.so"
echo "       └── libggml-cuda.so"
echo ""
echo "💡 Para usar no Python, adicione ao seu código:"
echo ""
echo "   import sys"
echo "   import os"
echo "   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
echo "   os.environ['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polaris_libs')"
echo "   import polaris_core as pc"
echo ""

