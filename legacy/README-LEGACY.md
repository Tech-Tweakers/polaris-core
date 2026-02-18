# 📦 Legacy: polaris_gen.cpp (Generic CLI Binding)

## History

This directory contains the **first C++ binding prototype for llama.cpp** developed ~2 years ago, when the Polaris project began its journey with AI model integration.

### Timeline

| Date | Version | Description |
|------|---------|-----------|
| ~2022 | v1.0 | First generic binding with standard chat templates |
| ~2023 | v1.5 | Improvements in tokenization and sampling |
| 2024+ | v2.0+ | **polaris_bind.cpp** — Absurdly optimized rewrite for XCT |

---

## What is `polaris_gen.cpp`?

A **generic and lightweight CLI** that:

- ✅ Loads GGUF models via llama.cpp
- ✅ Supports **automatic chat templates** (Qwen, Mistral, etc)
- ✅ Does **simple streaming** to stdout
- ✅ **No overhead** from callbacks or JSON early-stop
- ✅ Perfect for prototyping and quick testing

### Architecture

```c++
┌─────────────────────────────────────┐
│       polaris_gen.cpp (Main)        │
├─────────────────────────────────────┤
│ 1. Parse args (common_params)       │
│ 2. Load model + context             │
│ 3. Detect chat template             │
│ 4. Build prompt (text/ChatML)       │
│ 5. Tokenize + Prefill               │
│ 6. Decode + stream stdout           │
│ 7. Print metrics + cleanup          │
└─────────────────────────────────────┘
         ↓
   llama.cpp backend
```

---

## Why is it "legacy"?

### `polaris_gen.cpp` (Generic)
- ❌ No Python callbacks
- ❌ No JSON early-stop (XCT-specific)
- ❌ No intelligent batch backoff
- ❌ No GIL management
- ✅ **But**: Lighter weight, no pybind11 overhead

### `polaris_bind.cpp` (New, XCT-optimized)
- ✅ Python binding via pybind11
- ✅ Automatic JSON early-stop
- ✅ Streaming callbacks in Python
- ✅ Batch backoff with retry logic
- ✅ GIL properly managed
- ❌ **But**: pybind11 overhead (minimal, acceptable)

---

## When to use `polaris_gen.cpp`?

### Use this generic binding if you:

1. **Don't need to call from Python** — want a pure C++ CLI
2. **Want quick prototyping** — test new models without Python setup
3. **Need maximum lightness** — super constrained environment (edge devices)
4. **Want a "hello world" you can customize** — base for your own binding

### Usage Example

```bash
# Build (inside root folder with CMake)
cmake -B build -DCMAKE_BUILD_TYPE=Release
make -C build

# Run
./build/polaris_gen \
    -m model.gguf \
    -n 256 \
    -p "Hello, how are you?"
```

---

## Comparison: `polaris_gen` vs `polaris_bind`

| Aspect | polaris_gen | polaris_bind |
|--------|------------|--------------|
| **Type** | CLI (main) | Python module (.so) |
| **Chat Templates** | Auto-detect | Manual (ChatML XCT) |
| **Callbacks** | ❌ | ✅ (Python) |
| **JSON Early-Stop** | ❌ | ✅ (XCT-specific) |
| **Batch Backoff** | ❌ | ✅ (intelligent) |
| **GIL Management** | ❌ | ✅ (release/acquire) |
| **Overhead** | Minimal | Minimal (pybind11) |
| **Production** | Testing | ✅ XCT production |

---

## Detailed Implementation

### 1. Argument Parsing
```cpp
common_params params;
common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, nullptr);
params.conversation_mode = COMMON_CONVERSATION_MODE_AUTO;
```

### 2. Initialization
```cpp
common_init();
common_init_result init = common_init_from_params(params);
llama_model  * model = init.model.get();
llama_context* ctx   = init.context.get();
```

### 3. Chat Template Auto-detect
```cpp
auto chat_templates = common_chat_templates_init(model, params.chat_template);

if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
    params.conversation_mode = common_chat_templates_was_explicit(chat_templates.get())
        ? COMMON_CONVERSATION_MODE_ENABLED
        : COMMON_CONVERSATION_MODE_DISABLED;
}
```

### 4. Prompt Building (ChatML or plain text)
```cpp
std::string prompt_text;
if (using_chat) {
    common_chat_templates_inputs inputs;
    inputs.messages = chat_msgs;
    prompt_text = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
} else {
    prompt_text = params.prompt;  // raw text
}
```

### 5. Tokenization with Specials
```cpp
const bool use_specials = using_chat;
const bool add_bos_tok  = llama_vocab_get_add_bos(vocab) && !use_specials;
std::vector<llama_token> embd_inp = common_tokenize(ctx, prompt_text, add_bos_tok, use_specials);
```

### 6. Prefill (Push prompt tokens)
```cpp
for (int i = 0; i < (int)toks.size(); i += n_batch) {
    int n_eval = std::min((int)toks.size() - i, n_batch);
    llama_decode(ctx, llama_batch_get_one(&toks[i], n_eval));
    n_past += n_eval;
}
```

### 7. Decode + Streaming
```cpp
while (n_remain != 0) {
    llama_token id = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, id, false);

    if (llama_vocab_is_eog(vocab, id)) break;

    std::string piece = common_token_to_piece(ctx, id, params.special);
    std::fwrite(piece.data(), 1, piece.size(), stdout);
    std::fflush(stdout);

    // Push generated token back
    push_tokens({id});
    --n_remain;
}
```

### 8. Cleanup
```cpp
common_perf_print(ctx, smpl);
common_sampler_free(smpl);
llama_backend_free();
```

---

## Key Differences vs `polaris_bind.cpp`

### JSON Early-Stop (XCT-specific)
`polaris_gen.cpp` **doesn't have** this — runs until `n_predict` or EOG token.

`polaris_bind.cpp` **detects automatically**:
```cpp
bool has_key = (out.find("\"done\"") != std::string::npos) ||
               (out.find("\"next_step\"") != std::string::npos);
if (has_key && json_complete(out)) break;  // stop early!
```

### Batch Backoff (Resilience)
`polaris_gen.cpp` — simple failure:
```cpp
if (llama_decode(ctx, batch)) {
    LOG_ERR("llama_decode failed\n");
    return false;
}
```

`polaris_bind.cpp` — **intelligent retry with gradual reduction**:
```cpp
while (!ok) {
    int rc = llama_decode(ctx, batch);
    if (rc == 0) { ok = true; break; }

    try_ub = std::max(MIN_UB, try_ub/2);  // reduce and retry
    // retry with smaller batch...
}
```

### Python Callbacks
`polaris_gen.cpp` — nothing, output directly to stdout.

`polaris_bind.cpp`:
```cpp
py::object py_callback;  // Python callable
if (!py_callback.is_none() && !buf.empty()) {
    py::gil_scoped_acquire acquire;
    py::bytes b(buf.data(), buf.size());
    py_callback(b);  // stream to Python!
}
```

---

## Possible Future: Generic Binding v2

If you decide to make a **second generic binding** (non-XCT), you could:

1. **Reuse most of `polaris_gen.cpp`**
2. **Add Python callbacks** (like `polaris_bind`, but without JSON early-stop)
3. **Keep it lightweight** (no XCT-specific logic)
4. Create a `polaris_bind_generic.cpp` (indicative name)

This would be a **middle ground** between:
- Lightweight of the generic
- Power of Python callbacks

---

## Conclusion

`polaris_gen.cpp` is a **valuable historical artifact** that shows:

✅ Project evolution (generic → XCT-specific)
✅ How to do simple C++ → llama.cpp binding
✅ Solid base for customizations
✅ "Hello world" for devs who want to understand the stack

**Kept here for:**
- 📚 Educational reference
- 🔄 Base for new custom bindings
- 🎯 Quick prototyping

**Use `polaris_bind.cpp` for XCT production!**

---

## Credits

- **Architecture**: Polaris Team (you + comadre)
- **Polaris v1 (2022)**: First experiments
- **Polaris v2 (2024+)**: Optimized rewrite for XCT
- **Llama.cpp**: Base backend

---

## Related Links

- [polaris_bind.cpp](../polaris_bind.cpp) — Production XCT binding
- [README-DEPLOY.md](../README-DEPLOY.md) — Deploy guide
- [CMakeLists.txt](../CMakeLists.txt) — Build configuration
- [README.md](../README.md) — Main documentation
