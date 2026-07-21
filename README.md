# Polaris Core — C++ Binding for llama.cpp

> Deterministic LLM engine: a C++ binding for llama.cpp consumed by Python agents via pybind11.
> Supports the [XCT Protocol](https://github.com/Tech-Tweakers/xct) with JSON early-stop, grammar-constrained decoding and streaming callbacks.

🔗 **Protocol:** [XCT — Execution Control Transfer](https://github.com/Tech-Tweakers/xct)

🤖 **Model:** [XCT-Qwen3-4B on HuggingFace](https://huggingface.co/tech-tweakers/XCT-Qwen3-4B)

---

## 📋 Table of Contents

- [What is it?](#what-is-it)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Build & Deploy](#build--deploy)
- [How to Use](#how-to-use)
- [Legacy](#legacy)
- [Performance](#performance)
- [License](#license)

---

## What is it?

**Polaris Core** is a Python wrapper (`polaris_core.so`) that exposes the full power of **llama.cpp** directly to Python, with special support for:

- ✅ **GGUF Models** (Qwen, Mistral, Llama, etc)
- ✅ **GPU acceleration** (CUDA - optional)
- ✅ **Token streaming** via Python callbacks
- ✅ **JSON early-stop** (XCT-specific)
- ✅ **Intelligent batch backoff** (automatic retry)
- ✅ **Thread-safe** with mutex locks
- ✅ **GIL-aware** (Python Global Interpreter Lock management)

### Why custom vs official llama.cpp?

| Aspect | llama.cpp official | Polaris Core |
|--------|-------------------|--------------|
| Chat templates | ✅ (complex) | ✅ Manual (lean) |
| Python binding | ✅ (pybind11) | ✅ (pybind11, custom) |
| XCT-specific | ❌ | ✅ JSON early-stop |
| Batch backoff | ❌ | ✅ Automatic |
| Streaming callbacks | ❌ | ✅ (Python Bytes) |
| Size | Larger | **Smaller, lighter** |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Python Application (XCT, etc)           │
├─────────────────────────────────────────────────┤
│  import polaris_core                            │
│  eng = polaris_core.Engine(model_path)          │
│  result = eng.generate(prompt, ...)             │
└──────────────────┬──────────────────────────────┘
                   │ pybind11
┌──────────────────▼──────────────────────────────┐
│      PolarisEngine (C++) — polaris_bind.cpp     │
├─────────────────────────────────────────────────┤
│ • Tokenization (specials aware)                 │
│ • Prefill (batch optimization)                  │
│ • Decode loop (GIL release)                     │
│ • JSON early-stop (XCT)                         │
│ • Streaming callbacks (Bytes)                   │
│ • Batch backoff retry logic                     │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│    llama.cpp backend (common_*)                 │
├─────────────────────────────────────────────────┤
│ • Model loading                                 │
│ • Context management                            │
│ • Sampler (temperature, top_p, etc)             │
│ • GPU/CPU dispatch                              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  GGML backend (CPU / CUDA / Metal)              │
├─────────────────────────────────────────────────┤
│ • Matrix operations                             │
│ • Tensor computation                            │
│ • Device acceleration                           │
└─────────────────────────────────────────────────┘
```

---

## File Structure

```
polaris/
├── README.md                    ← You are here
├── polaris_bind.cpp             ← Main engine (XCT-optimized)
├── CMakeLists.txt               ← Build configuration
├── build-polaris-core.sh         ← Build script
├── copy-to-project.sh            ← Deploy script
├── README-DEPLOY.md              ← Deployment guide
├── example_usage.py              ← Usage example
└── legacy/                       ← Historical code
    ├── README-LEGACY.md
    ├── polaris_gen.cpp
    └── arg_compat.h
```

### Main Files

#### `polaris_bind.cpp` (18KB)
The **heart** of the project. Implements:
- `PolarisEngine` C++ struct
- Python binding via pybind11
- Token generation with streaming
- JSON early-stop for XCT
- Batch backoff with retry logic

#### `CMakeLists.txt`
Build configuration that:
- Detects CPU vs GPU build
- Links against compiled llama.cpp
- Configures pybind11 for Python module
- Defines RPATH for deployment

#### `README-DEPLOY.md`
Complete deployment guide:
- How to copy files
- Configure LD_LIBRARY_PATH
- Use patchelf for RPATH
- Troubleshooting

#### `legacy/`
Historical code (v1.0):
- `polaris_gen.cpp` — Generic CLI
- Kept as educational reference

---

## Build & Deploy

### Prerequisites

```bash
# Dependencies
sudo apt-get install -y \
    cmake \
    build-essential \
    python3-dev \
    pybind11-dev \
    libcurl4-openssl-dev

# CUDA (optional, for GPU)
# https://developer.nvidia.com/cuda-downloads
```

### Build

```bash
# 1. Configure (POLARIS_LLAMA_ROOT defaults to ../llama.cpp-latest)
cd /path/to/polaris-core
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Compile
cmake --build build --parallel

# 3. Result
ls -lh build/polaris_core.*.so
```

### Deploy

```bash
# Option 1: Automatic script
./copy-to-project.sh /your/python/project

# Option 2: Manual (see README-DEPLOY.md)
```

---

## How to Use

### Python API

```python
import polaris_core as pc
import sys

# 1. Create engine
eng = pc.Engine(
    model_path="xct-model.gguf",
    n_ctx=4096,           # Context size
    n_threads=8,          # CPU threads
    n_gpu_layers=-1       # -1 = max GPU layers
)

# 2. Generate with callback (streaming)
def on_chunk(data: bytes):
    print(data.decode('utf-8', errors='ignore'), end='', flush=True)

result = eng.generate(
    prompt="Hello, how are you?",
    system_prompt="You are a friendly assistant.",
    n_predict=256,
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
    callback=on_chunk  # Streaming!
)

print("\n" + "="*50)
print("Final result:", result)

# 3. Use without callback (full generation)
result_full = eng.generate(
    prompt="Another prompt here",
    n_predict=512
)
```

### Environment Variables

```bash
# Number of GPU layers
export POLARIS_N_GPU_LAYERS=999

# Batch size (prefill)
export POLARIS_BATCH=256

# Micro-batch size (decode)
export POLARIS_UBATCH=128

# Context safety margin
export POLARIS_SAFETY=16

# Include special tokens in token_to_piece output
export POLARIS_SPECIAL=true

# Streaming flush size (bytes)
export POLARIS_FLUSH=64

# Flush every N tokens
export POLARIS_TOKFLUSH=1

# Temporal flush (milliseconds)
export POLARIS_MS_FLUSH=100

# Debug stages (prompt/tokenize/prefill/sample/piece/push)
export POLARIS_STAGE=prompt

# Override the llama.cpp source tree used by CMake
export POLARIS_LLAMA_ROOT=/path/to/llama.cpp
```

### Diagnostics

```python
# See pipeline stages
import os
os.environ['POLARIS_STAGE'] = 'tokenize'
result = eng.generate("test")
# Output: "[OK] tokenize: 42 toks"

os.environ['POLARIS_STAGE'] = 'prefill'
result = eng.generate("test")
# Output: "[OK] prefill in 0.123s"
```

---

## Legacy

The `legacy/` folder contains the **first prototype** from 2 years ago.

### Why keep it?

- 📚 **Educational reference** — shows project evolution
- 🔧 **Base for customizations** — reusable for other bindings
- 🎯 **Quick prototyping** — generic CLI without Python overhead

### Learn more

👉 [legacy/README-LEGACY.md](legacy/README-LEGACY.md)

---

## Performance

### Benchmarks (Qwen3-4B-v0.1 on RTX 4090)

| Metric | Value |
|--------|-------|
| Prefill (256 tok prompt) | ~15ms (~17k tok/s) |
| Decode (per token) | ~2ms (500 tok/s) |
| Memory footprint | ~9GB VRAM |
| Batch backoff triggers | ~0.1% (rare) |
| JSON early-stop | -30% time (typical) |

### Implemented Optimizations

1. **GIL Release**
   ```cpp
   py::gil_scoped_release release;  // Allow other Python threads
   id = common_sampler_sample(smpl.get(), ctx, -1);
   ```

2. **Batch Backoff**
   ```cpp
   while (!ok) {
       int rc = llama_decode(ctx, batch);
       if (rc == 0) { ok = true; break; }
       try_ub = std::max(MIN_UB, try_ub/2);  // Reduce and retry
   }
   ```

3. **JSON Early-Stop (XCT)**
   ```cpp
   if (out.find("\"done\"") != npos && json_complete(out)) {
       break;  // Stop early if JSON is complete
   }
   ```

4. **Streaming Callbacks**
   ```cpp
   buf += piece;
   if (buf.size() >= FLUSH_BYTES) {
       flush_cb();  // Send bytes to Python in real-time
   }
   ```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'polaris_core'"

```bash
# 1. Copy .so to Python path
cp build/lib/polaris_core.*.so your_project/polaris_core.so

# 2. Configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=your_project/polaris_libs:$LD_LIBRARY_PATH

# 3. Run Python
python your_script.py
```

### Error: "libllama.so: cannot open shared object file"

```bash
# Use patchelf to configure RPATH
patchelf --set-rpath '$ORIGIN/polaris_libs' polaris_core.so
cd polaris_libs && patchelf --set-rpath '$ORIGIN' libllama.so
```

### Decode fails with "room<=0"

```bash
# Increase context
eng = pc.Engine(model_path, n_ctx=8192)

# Or reduce n_predict
result = eng.generate(prompt, n_predict=128)  # Smaller generation
```

---

## Contributing

### Add a feature?

1. Edit `polaris_bind.cpp`
2. Recompile: `cmake -B build && make -C build`
3. Test: `python example_usage.py`
4. Commit with clear message

### Bug fix?

1. Reproduce in `example_usage.py`
2. Fix the code
3. Validate with `example_usage.py`
4. Commit

---

## Credits

**Built on** llama.cpp by Georgi Gerganov

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT License)

---

## Useful Links

- 📖 [README-DEPLOY.md](README-DEPLOY.md) — Deployment guide
- 📚 [legacy/README-LEGACY.md](legacy/README-LEGACY.md) — History
- 🔧 [example_usage.py](example_usage.py) — Complete examples
- 🎯 [CMakeLists.txt](CMakeLists.txt) — Build configuration

