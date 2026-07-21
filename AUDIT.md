# Polaris Core — Full Audit Findings

This document records the findings of the full audit requested for `Tech-Tweakers/polaris-core` and the fixes applied to make the repository safe and reproducible for automated agents.

## Scope

- `polaris_bind.cpp` — main pybind11 / llama.cpp integration
- `CMakeLists.txt` and `xct-server/CMakeLists.txt` — build configuration
- `build-polaris-core.sh`, `xct-server/build-xct-server.sh`, `copy-to-project.sh` — build/deploy scripts
- `example_usage.py` — Python example
- Repository infrastructure: CI, lint, tests, packaging metadata

## Critical Fixes

### 1. Infinite loop in batch backoff (`polaris_bind.cpp`)

The backoff logic in `push_tokens` could loop forever when `llama_decode` failed with a small `room` value. The old check `if (try_ub == n_eval)` did not catch the case where `n_eval` got stuck below `MIN_UB`.

**Fix:** Track the previous `n_eval` and abort when the next `n_eval` is not smaller than the previous one, indicating backoff is exhausted.

```cpp
int next_n_eval = std::min(try_ub, room);
if (next_n_eval >= n_eval) {
    throw std::runtime_error("llama_decode falhou (backoff esgotado)");
}
n_eval = next_n_eval;
```

### 2. Dead code / unused variables in tool-call handling

`in_tool_call`, `tool_calls_seen` and `tool_buf` were declared and modified but never read. This produced compiler warnings under `-Wall -Wextra` and confused the protocol implementation.

**Fix:** Removed the unused state. Token detection for `<tool_call>` / `</tool_call>` remains in place; a comment explains that full structured tool-call streaming requires protocol work on the client side.

### 3. Incomplete JSON early-stop

`json_complete` only counted `{` / `}`, so array-only JSON responses could cause incorrect early termination or missed termination.

**Fix:** `json_complete` now also balances `[` / `]` and validates that all braces and brackets are closed and no string is left open.

### 4. Streaming flush not honoring `force`, token or time triggers

The `flush_cb` lambda accepted a `force` parameter but ignored it, and `TOK_FLUSH` / `MS_FLUSH` were parsed but never used.

**Fix:** `flush_cb` now respects `force`; the decode loop flushes on bytes, token count and elapsed milliseconds.

### 5. `params.special` was left at default

`common_token_to_piece` was called with `params.special`, which defaulted to `false`. For Qwen-style special / tool tokens this can render special pieces as empty text.

**Fix:** Added `POLARIS_SPECIAL` environment variable (default `true`) and set `params.special` accordingly. `POLARIS_USE_SPECIALS` is no longer used.

### 6. Hard-coded build paths

`CMakeLists.txt`, `build-polaris-core.sh`, `xct-server/build-xct-server.sh` and `copy-to-project.sh` contained hard-coded paths to `/home/atorres/dev/polaris/...` or `../llama.cpp-latest`, making the build fail on any other machine or agent.

**Fix:**
- CMake now honors `POLARIS_LLAMA_ROOT` and fails with a clear message when the directory is missing.
- Build scripts accept `POLARIS_LLAMA_ROOT`, `POLARIS_CORE_DIR`, `POLARIS_V3_API_DIR`, `POLARIS_XCT_SERVER_DIR`, `POLARIS_BUILD_DIR`, `CUDA_COMPILER` and `CUDAToolkit_ROOT`.
- `copy-to-project.sh` defaults to the standard `build/` directory produced by pybind11 and falls back to the old `build/bin` layout.

### 7. Wrong `LLAMA_ROOT` in `xct-server/CMakeLists.txt`

The xct-server CMake pointed `LLAMA_ROOT` to `${CMAKE_CURRENT_SOURCE_DIR}/../..` (i.e., `polaris-core` itself) and sourced `polaris_bind.cpp` from a non-existent relative path.

**Fix:** `xct-server/CMakeLists.txt` now uses `POLARIS_LLAMA_ROOT` with a sensible `../../llama.cpp-latest` fallback and references `../polaris_bind.cpp`.

## Infrastructure Added

- `.github/workflows/ci.yml` — GitHub Actions workflow running shellcheck, ruff, mypy and pytest.
- `.pre-commit-config.yaml` — pre-commit hooks for whitespace, YAML, shellcheck and ruff.
- `pyproject.toml` — packaging metadata, build-system requirements, ruff/mypy configuration.
- `tests/test_import.py` — checks that `example_usage.py` compiles and documents the optional `polaris_core` import.
- `tests/test_json_complete.py` — mirrors the C++ JSON-balancing heuristic in Python so regressions can be caught without compiling.

## Remaining Risks / Next Steps

1. **No local build verification in this PR.** `polaris_core` cannot be compiled without a pre-built `llama.cpp` tree. A follow-up should add a self-hosted runner or a container image with `POLARIS_LLAMA_ROOT` pre-installed.
2. **Tool-call protocol is not complete.** Tool tokens are detected but still emitted as plain text. A future change should stream tool-call payloads through a separate channel.
3. **`llama_get_memory` / `llama_memory_clear` API.** These calls depend on the exact llama.cpp version being built against. CI should pin a known-good `llama.cpp` commit.
4. **CUDA 12.2 is still the default** in build scripts, but can now be overridden via `CUDA_COMPILER` and `CUDAToolkit_ROOT`.

## Verification

```bash
# Lint
ruff check .
mypy --ignore-missing-imports .
pytest tests/

# Build (requires llama.cpp compiled at the location pointed by POLARIS_LLAMA_ROOT)
export POLARIS_LLAMA_ROOT=/path/to/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```
