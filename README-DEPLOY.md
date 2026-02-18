# 📦 Deploying polaris_core to Your Python Project

## Required Files

To use `polaris_core` in your Python project, you need to copy the following files:

### 1. Python Module
- `polaris_core.cpython-310-x86_64-linux-gnu.so` → rename to `polaris_core.so`

### 2. Shared Libraries (`.so`)
Copy all of these libraries to a directory (ex: `polaris_libs/`):
- `libllama.so`
- `libggml.so`
- `libggml-base.so`
- `libggml-cpu.so`
- `libggml-cuda.so`

## 📋 Recommended Structure

```
your_project/
├── polaris_core.so          # Python module
├── polaris_libs/            # Directory with libraries
│   ├── libllama.so
│   ├── libggml.so
│   ├── libggml-base.so
│   ├── libggml-cpu.so
│   └── libggml-cuda.so
└── your_script.py
```

## 🚀 Method 1: Using the Automatic Script

```bash
cd /path/to/polaris-core-cpp/polaris
./copy-to-project.sh /path/to/your/project
```

## 🚀 Method 2: Manual Copy

```bash
# 1. Copy Python module
cp build/bin/polaris_core.cpython-*.so /your/project/polaris_core.so

# 2. Create directory for libraries
mkdir -p /your/project/polaris_libs

# 3. Copy libraries
cp build/bin/libllama.so /your/project/polaris_libs/
cp build/bin/libggml.so /your/project/polaris_libs/
cp build/bin/libggml-base.so /your/project/polaris_libs/
cp build/bin/libggml-cpu.so /your/project/polaris_libs/
cp build/bin/libggml-cuda.so /your/project/polaris_libs/
```

## 💻 Python Code to Use

```python
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure LD_LIBRARY_PATH to find libraries
lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polaris_libs')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = lib_dir + ':' + os.environ['LD_LIBRARY_PATH']
else:
    os.environ['LD_LIBRARY_PATH'] = lib_dir

# Import polaris_core
import polaris_core as pc

# Use the library
eng = pc.Engine("model.gguf", n_ctx=4096, n_gpu_layers=-1)
```

## 🔧 Alternative: Using RPATH (Recommended)

If you prefer not to configure `LD_LIBRARY_PATH`, you can use `patchelf` to set the RPATH:

```bash
# Install patchelf (if you don't have it)
sudo apt-get install patchelf

# Configure RPATH on the Python module
patchelf --set-rpath '$ORIGIN/polaris_libs' polaris_core.so

# Configure RPATH on the libraries too
cd polaris_libs
patchelf --set-rpath '$ORIGIN' libllama.so
patchelf --set-rpath '$ORIGIN' libggml.so
patchelf --set-rpath '$ORIGIN' libggml-base.so
patchelf --set-rpath '$ORIGIN' libggml-cpu.so
patchelf --set-rpath '$ORIGIN' libggml-cuda.so
```

With RPATH configured, you don't need to modify `LD_LIBRARY_PATH` in your Python code.

## 📝 Important Notes

1. **Python Version**: The module was compiled for Python 3.10. If you use another version, you may need to recompile.

2. **Architecture**: The module was compiled for x86_64 Linux. For other architectures, recompile.

3. **CUDA**: The libraries include CUDA support. Make sure the CUDA driver is installed on the target system.

4. **Size**: The `libggml-cuda.so` is large (~500MB) because it includes compiled CUDA kernels.

## 🐛 Troubleshooting

### Error: "libllama.so: cannot open shared object file"
- Verify that `LD_LIBRARY_PATH` is configured correctly
- Or use `patchelf` to configure RPATH

### Error: "undefined symbol"
- Make sure to copy ALL listed libraries
- Verify that library versions are compatible

### Error: "CUDA not found"
- Install CUDA drivers on the target system
- Verify with `nvidia-smi`

## Quick Start Example

```python
import polaris_core as pc
import os

# Setup library path (if not using RPATH)
lib_dir = os.path.join(os.path.dirname(__file__), 'polaris_libs')
os.environ['LD_LIBRARY_PATH'] = lib_dir

# Create engine
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

---

**For more information, see:** [README.md](README.md)
