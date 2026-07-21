# Deploying polaris_core to Your Python Project

This guide explains how to move the compiled `polaris_core` module and its shared libraries into a Python project so it can be imported at runtime.

## Required Files

### 1. Python Module

After building, the module is produced as:

```
build/polaris_core.cpython-<py_version>-<arch>-linux-gnu.so
```

Rename or copy it to `polaris_core.so` in your project root.

### 2. Shared Libraries (`.so`)

Copy all produced libraries to a directory such as `polaris_libs/`:

- `libllama.so`
- `libggml.so`
- `libggml-base.so`
- `libggml-cpu.so`
- `libggml-cuda.so` (if built with CUDA)

The exact filenames may include version suffixes (e.g., `.so.0`). Copy the versioned files as well.

## Recommended Structure

```
your_project/
├── polaris_core.so          # Python module
├── polaris_libs/            # Shared libraries
│   ├── libllama.so
│   ├── libggml.so
│   ├── libggml-base.so
│   ├── libggml-cpu.so
│   └── libggml-cuda.so
└── your_script.py
```

## Method 1: Using the Automatic Script

```bash
cd /path/to/polaris-core
./copy-to-project.sh /path/to/your/project
```

The script defaults to `build/` produced by pybind11 and falls back to the legacy `build/bin` layout. Override the build directory with:

```bash
POLARIS_BUILD_DIR=/path/to/build ./copy-to-project.sh /path/to/your/project
```

## Method 2: Manual Copy

```bash
SRC=/path/to/polaris-core
LLAMA_BUILD=$POLARIS_LLAMA_ROOT/build-cpu  # or build-gpu
DEST=/your/project

# 1. Copy Python module
cp "$SRC"/build/polaris_core.cpython-*.so "$DEST"/polaris_core.so

# 2. Create directory for libraries
mkdir -p "$DEST"/polaris_libs

# 3. Copy libraries
cp "$LLAMA_BUILD"/bin/libllama.so*   "$DEST"/polaris_libs/
cp "$LLAMA_BUILD"/bin/libggml*.so*   "$DEST"/polaris_libs/
```

## Method 3: Editable Install (Experimental)

If you are developing Polaris Core itself, you can install it in editable mode and the build backend will place the compiled extension into the source tree:

```bash
cd /path/to/polaris-core
pip install -e ".[dev]"
```

This requires `POLARIS_LLAMA_ROOT` to be set and a successful `cmake --build build --parallel` before importing.

## Python Code to Use

```python
import os
import sys

# Add the project directory to sys.path if needed
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Configure LD_LIBRARY_PATH to find libraries
lib_dir = os.path.join(project_dir, 'polaris_libs')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = lib_dir

# Import polaris_core
import polaris_core as pc

# Use the library
engine = pc.Engine(
    model_path="Qwen3-4B.gguf",
    n_ctx=4096,
    n_gpu_layers=-1
)

# Generate text with streaming
def on_token(data: bytes):
    print(data.decode('utf-8', errors='ignore'), end='', flush=True)

result = engine.generate(
    prompt="What is machine learning?",
    system_prompt="You are a helpful AI assistant.",
    n_predict=256,
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
    callback=on_token
)

print("\n\nGeneration complete!")
```

## Alternative: Using RPATH (Recommended)

If you prefer not to configure `LD_LIBRARY_PATH` in Python, use `patchelf` to set the RPATH in the `.so` and in the shared libraries:

```bash
# Install patchelf (if you don't have it)
sudo apt-get install patchelf

# Configure RPATH on the Python module
patchelf --set-rpath '$ORIGIN/polaris_libs' your_project/polaris_core.so

# Configure RPATH on the libraries
cd your_project/polaris_libs
patchelf --set-rpath '$ORIGIN' libllama.so
patchelf --set-rpath '$ORIGIN' libggml-base.so
patchelf --set-rpath '$ORIGIN' libggml-cpu.so
patchelf --set-rpath '$ORIGIN' libggml-cuda.so
```

With RPATH configured, you do not need to modify `LD_LIBRARY_PATH` in your Python code.

## Important Notes

1. **Python Version**: The module is compiled for the Python interpreter used during the build (e.g., Python 3.10). If you use another version, recompile with that interpreter active.

2. **Architecture**: The module is compiled for the architecture of the build machine (typically `x86_64` Linux). For other architectures, recompile.

3. **CUDA**: GPU libraries include CUDA support. Make sure the CUDA driver is installed on the target system and `nvidia-smi` works.

4. **Size**: `libggml-cuda.so` is large (~500 MB) because it includes compiled CUDA kernels.

5. **Library versions must match**: Copy the `.so` files from the same `llama.cpp` build that was used to compile `polaris_core`. Mismatched versions lead to `undefined symbol` errors at import time.

## Troubleshooting

### Error: "libllama.so: cannot open shared object file"

- Verify that `LD_LIBRARY_PATH` points to the `polaris_libs/` directory.
- Or use `patchelf` to set RPATH as shown above.
- Check that `polaris_libs/` contains `libllama.so` (possibly with version suffixes).

### Error: "undefined symbol"

- Make sure you copied **all** listed libraries.
- Verify that the `llama.cpp` build matches the `polaris_core` build.
- Rebuild `polaris_core` against the current `llama.cpp` tree if it was updated.

### Error: "CUDA not found"

- Install CUDA drivers on the target system.
- Verify with `nvidia-smi`.
- If you only need CPU inference, use the CPU build (`-DPOLARIS_ENABLE_CUDA=OFF`) and copy the CPU libraries.

---

**For more information, see:** [README.md](README.md)
