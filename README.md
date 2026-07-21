# Polaris Core — C++ Binding for llama.cpp

> Deterministic LLM engine: a C++ binding for llama.cpp consumed by Python agents via pybind11.
> Supports the [XCT Protocol](https://github.com/Tech-Tweakers/xct) with JSON early-stop, grammar-constrained decoding and streaming callbacks.

🔗 **Protocol:** [XCT — Execution Control Transfer](https://github.com/Tech-Tweakers/xct)  
🤖 **Model:** [XCT-Qwen3-4B on HuggingFace](https://huggingface.co/tech-tweakers/XCT-Qwen3-4B)  
📦 **PyPI-style build:** `pip install -e .` (requires a pre-compiled llama.cpp tree)

---

## Table of Contents

- [What is it?](#what-is-it)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Build & Deploy](#build--deploy)
- [How to Use](#how-to-use)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Legacy](#legacy)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## What is it?

**Polaris Core** is a Python wrapper (`polaris_core.*.so`) that exposes the full power of **llama.cpp** directly to Python, with special support for:

- ✅ **GGUF Models** (Qwen, Mistral, Llama, etc)
- ✅ **GPU acceleration** (CUDA - optional)
- ✅ **Token streaming** via Python callbacks
- ✅ **JSON early-stop** (XCT-specific)
- ✅ **Grammar-constrained decoding** (GBNF)
- ✅ **ChatML with preserved roles** (`generate_chat`)
- ✅ **Intelligent batch backoff** (automatic retry)
- ✅ **Thread-safe** with mutex locks
- ✅ **GIL-aware** (Python Global Interpreter Lock management)

### Why custom vs official llama.cpp?

| Aspect | llama.cpp official | Polaris Core |
|---|---|---|
| Chat templates | ✅ (complex) | ✅ Manual (lean) |
| Python binding | ✅ (pybind11) | ✅ (pybind11, custom) |
| XCT-specific | ❌ | ✅ JSON early-stop |
| Grammar / constrained decoding | ✅ | ✅ (GBNF) |
| Batch backoff | ❌ | ✅ Automatic |
| Streaming callbacks | ❌ | ✅ (Python `bytes`) |
| Size | Larger | **Smaller, lighter** |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Python Application (XCT, etc)           │
├─────────────────────────────────────────────────┤
│  import polaris_core as pc                       │
│  eng = pc.Engine(model_path)                     │
│  result = eng.generate(prompt, ...)              │
│  result = eng.generate_chat(messages, ...)     │
└──────────────────┬──────────────────────────────┘
                   │ pybind11
┌──────────────────▼──────────────────────────────┐
│      PolarisEngine (C++) — polaris_bind.cpp     │
├─────────────────────────────────────────────────┤
│ • Tokenization (specials aware)                 │
│ • Prefill (batch optimization)                  │
│ • Decode loop (GIL release)                     │
│ • JSON early-stop (XCT)                         │
│ • Grammar-constrained sampling                  │
│ • Streaming callbacks (bytes)                   │
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
polaris-core/
├── .github/workflows/ci.yml      ← GitHub Actions (lint / conditional build)
├── .pre-commit-config.yaml       ← Pre-commit hooks
├── AUDIT.md                      ← Full audit findings
├── CMakeLists.txt                ← Build configuration
├── LICENSE
├── README.md                     ← You are here
├── README-DEPLOY.md              ← Deployment guide
├── build-polaris-core.sh         ← Build script (GPU + CPU)
├── copy-to-project.sh            ← Copy artifacts to a Python project
├── example_usage.py              ← Usage examples
├── polaris_bind.cpp              ← Main C++ engine (XCT-optimized)
├── pyproject.toml                ← Python packaging / lint config
├── scripts/
│   ├── deploy.sh                 ← Deploy to remote host
│   └── tail-chat.sh              ← Tail remote Polaris logs
├── tests/
│   ├── test_import.py
│   └── test_json_complete.py
├── xct-server/
│   ├── CMakeLists.txt            ← CMake for xct-server layout
│   └── build-xct-server.sh       ← xct-server build script
└── legacy/                       ← Historical reference code
    ├── README-LEGACY.md
    ├── arg_compat.h
    └── polaris_gen.cpp
```

### Main Files

#### `polaris_bind.cpp`
The heart of the project. Implements:
- `PolarisEngine` C++ struct
- Python binding via pybind11
- Token generation with streaming
- JSON early-stop for XCT
- Grammar-constrained decoding
- Batch backoff with retry logic
- ChatML generation with preserved roles (`generate_chat`)

#### `CMakeLists.txt`
Build configuration that:
- Detects CPU vs GPU build (`-DPOLARIS_ENABLE_CUDA=ON`)
- Uses `POLARIS_LLAMA_ROOT` to locate the llama.cpp source tree
- Links against compiled llama.cpp
- Configures pybind11 for Python module
- Defines RPATH for deployment

#### `README-DEPLOY.md`
Complete deployment guide:
- How to copy files
- Configure `LD_LIBRARY_PATH`
- Use `patchelf` for RPATH
- Troubleshooting

#### `legacy/`
Historical code (v1.0). Kept as educational reference.

---

## Build & Deploy

### Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    python3-dev \
    pybind11-dev \
    libcurl4-openssl-dev \
    shellcheck

# CUDA (optional, for GPU)
# https://developer.nvidia.com/cuda-downloads
```

Polaris Core builds against an existing **llama.cpp** tree. By default it looks for `../llama.cpp-latest` relative to the repo root. Override with:

```bash
export POLARIS_LLAMA_ROOT=/path/to/llama.cpp
```

### Build

```bash
cd /path/to/polaris-core

# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# GPU (CUDA)
cmake -B build-gpu -DPOLARIS_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu --parallel
```

The compiled module is produced as `build/polaris_core.*.so`.

### Deploy

Copy the module and shared libraries into your Python project:

```bash
# Option 1: automatic script
./copy-to-project.sh /your/python/project

# Option 2: manual (see README-DEPLOY.md)
```

### Build everything (CPU + GPU) and copy to polaris-v3-api

```bash
export POLARIS_LLAMA_ROOT=/path/to/llama.cpp
export POLARIS_V3_API_DIR=/path/to/polaris-v3-api/polaris_api/polaris_core
./build-polaris-core.sh
```

For the xct-server layout:

```bash
export POLARIS_LLAMA_ROOT=/path/to/llama.cpp
export POLARIS_XCT_SERVER_DIR=/path/to/xct-server
./xct-server/build-xct-server.sh
```

---

## How to Use

### Python API

```python
import polaris_core as pc

# 1. Create engine
eng = pc.Engine(
    model_path="xct-model.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1  # -1 = max GPU layers
)

# 2. Generate with streaming
def on_chunk(data: bytes):
    print(data.decode("utf-8", errors="ignore"), end="", flush=True)

result = eng.generate(
    prompt="Hello, how are you?",
    system_prompt="You are a friendly assistant.",
    n_predict=256,
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
    callback=on_chunk
)

print("\n" + "=" * 50)
print("Final result:", result)

# 3. Conversation with preserved roles (recommended for XCT)
messages = [
    ("system", "You are an assistant that uses tools."),
    ("user", "What is the weather?"),
]
result = eng.generate_chat(
    messages=messages,
    n_predict=256,
    temperature=0.7,
    callback=on_chunk
)
```

### Generate vs Generate Chat

- `generate(prompt, system_prompt, ...)` — legacy single-turn helper.
- `generate_chat(messages, ...)` — preserves roles in individual ChatML blocks. Use this for multi-turn XCT conversations where the model must distinguish its own tool calls from user text.

### Grammar (GBNF)

Pass a grammar string to constrain the output format:

```python
grammar = r'''
root ::= "{" ws "\"done\"" ws ":" ws ("true" | "false") ws "}"
ws ::= " "?
'''

result = eng.generate(
    prompt="Are you done?",
    grammar=grammar,
    n_predict=64
)
```

---

## Environment Variables

All environment variables are read per call, which makes them useful for diagnostics but adds a small overhead. Set them before creating the engine when possible.

```bash
# Number of GPU layers (default: 999)
export POLARIS_N_GPU_LAYERS=999

# Batch size (prefill) and micro-batch size (decode)
export POLARIS_BATCH=256
export POLARIS_UBATCH=128

# Include special tokens in rendered token pieces (default: true)
export POLARIS_SPECIAL=true

# Context safety margin (default: 16)
export POLARIS_SAFETY=16

# Streaming flush size in bytes (default: 64)
export POLARIS_FLUSH=64

# Flush every N generated tokens (default: 1)
export POLARIS_TOKFLUSH=1

# Temporal flush in milliseconds (default: 100)
export POLARIS_MS_FLUSH=100

# Reset KV cache before every generation (default: true)
export POLARIS_RESET_KV=true

# Suppress the <think> block for Qwen3.5 models (default: false)
export POLARIS_SUPPRESS_THINK=false

# Debug stages: prompt, tokenize, prefill, sample, piece, push
export POLARIS_STAGE=prompt

# Override llama.cpp source tree for CMake
export POLARIS_LLAMA_ROOT=/path/to/llama.cpp
```

### Diagnostics

```python
import os
import polaris_core as pc

os.environ['POLARIS_STAGE'] = 'tokenize'
result = eng.generate("test")
# Output: "[OK] tokenize: 42 toks"

os.environ['POLARIS_STAGE'] = 'prefill'
result = eng.generate("test")
# Output: "[OK] prefill in 0.123s"
```

---

## Development

### Lint and test (no build required)

```bash
# Python
ruff check .
mypy --ignore-missing-imports .
pytest tests/

# Shell scripts
shellcheck build-polaris-core.sh copy-to-project.sh scripts/*.sh xct-server/*.sh

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Install in editable mode

```bash
pip install -e ".[dev]"
```

---

## Legacy

The `legacy/` folder contains the first prototype from two years ago.

### Why keep it?

- **Educational reference** — shows project evolution
- **Base for customizations** — reusable for other bindings
- **Quick prototyping** — generic CLI without Python overhead

Learn more in [legacy/README-LEGACY.md](legacy/README-LEGACY.md).

---

## Performance

### Benchmarks (Qwen3-4B-v0.1 on RTX 4090)

| Metric | Value |
|---|---|
| Prefill (256 tok prompt) | ~15 ms (~17 k tok/s) |
| Decode (per token) | ~2 ms (500 tok/s) |
| Memory footprint | ~9 GB VRAM |
| Batch backoff triggers | ~0.1% (rare) |
| JSON early-stop | ~-30% time (typical) |

### Implemented Optimizations

1. **GIL Release**
   ```cpp
   py::gil_scoped_release release;
   id = common_sampler_sample(smpl.get(), ctx, -1);
   ```

2. **Batch Backoff**
   ```cpp
   while (!ok) {
       int rc = llama_decode(ctx, batch);
       if (rc == 0) { ok = true; break; }

       llama_batch_free(batch);
       try_ub = std::max(MIN_UB, try_ub / 2);
       int next_n_eval = std::min(try_ub, room);
       if (next_n_eval >= n_eval) {
           throw std::runtime_error("llama_decode falhou (backoff esgotado)");
       }
       n_eval = next_n_eval;
       batch = make_batch(n_eval, i);
   }
   ```

3. **JSON Early-Stop (XCT)**
   ```cpp
   bool has_key =
       (out.find("\"done\"")      != std::string::npos) ||
       (out.find("\"next_step\"") != std::string::npos);

   if (has_key && json_complete(out)) {
       flush_cb(true);
       break;
   }
   ```

4. **Streaming Callbacks**
   ```cpp
   buf += piece;
   out += piece;

   bool by_bytes = buf.size() >= FLUSH_BYTES;
   bool by_toks  = (TOK_FLUSH > 0) && (++tok_since_flush >= (size_t)TOK_FLUSH);
   bool by_time  = ms_since_last_flush >= MS_FLUSH;

   if (by_bytes || by_toks || by_time) {
       flush_cb();
   }
   ```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'polaris_core'"

```bash
# 1. Copy .so to your project
cp build/polaris_core.*.so your_project/polaris_core.so

# 2. Copy the required shared libraries
mkdir -p your_project/polaris_libs
cp "$POLARIS_LLAMA_ROOT/build-cpu/bin"/libllama.so* your_project/polaris_libs/
cp "$POLARIS_LLAMA_ROOT/build-cpu/bin"/libggml*.so* your_project/polaris_libs/

# 3. Configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=your_project/polaris_libs:$LD_LIBRARY_PATH

# 4. Run Python
python your_script.py
```

### Error: "libllama.so: cannot open shared object file"

Use `patchelf` to configure RPATH:

```bash
patchelf --set-rpath '$ORIGIN/polaris_libs' your_project/polaris_core.so
cd your_project/polaris_libs
patchelf --set-rpath '$ORIGIN' libllama.so
patchelf --set-rpath '$ORIGIN' libggml-base.so
patchelf --set-rpath '$ORIGIN' libggml-cpu.so
```

### Error: "llama.cpp root not found"

Set `POLARIS_LLAMA_ROOT` to the absolute path of the compiled llama.cpp source tree:

```bash
export POLARIS_LLAMA_ROOT=/home/you/dev/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### Decode fails with "room<=0"

```python
# Increase context
eng = pc.Engine(model_path, n_ctx=8192)

# Or reduce n_predict
result = eng.generate(prompt, n_predict=128)
```

### Error: "undefined symbol"

- Make sure you copied **all** listed libraries
- Verify that the llama.cpp version matches the one Polaris Core was compiled against

### Error: "CUDA not found"

- Install CUDA drivers on the target system
- Verify with `nvidia-smi`
- Override the CUDA toolkit path when building:
  ```bash
  export CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  export CUDAToolkit_ROOT=/usr/local/cuda
  cmake -B build-gpu -DPOLARIS_ENABLE_CUDA=ON
  ```

---

## Contributing

### Add a feature?

1. Edit `polaris_bind.cpp`
2. Build: `cmake -B build && cmake --build build --parallel`
3. Run lint and tests: `ruff check . && mypy . && pytest tests/`
4. Test locally with `python example_usage.py`
5. Commit with a clear message

### Bug fix?

1. Reproduce in `example_usage.py` or add a test in `tests/`
2. Fix the code
3. Validate with `ruff check . && pytest tests/`
4. Commit

---

## Credits

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov.

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT License).

---

## Useful Links

- [README-DEPLOY.md](README-DEPLOY.md) — Deployment guide
- [AUDIT.md](AUDIT.md) — Full audit findings
- [legacy/README-LEGACY.md](legacy/README-LEGACY.md) — History
- [example_usage.py](example_usage.py) — Complete examples
- [CMakeLists.txt](CMakeLists.txt) — Build configuration
