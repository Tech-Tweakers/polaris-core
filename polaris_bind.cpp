// pybind: engine embutido no llama.cpp (reusa common.*)
#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>   // getenv
#include <chrono>    // métricas / timer de flush
#include <cctype>    // tolower

namespace py = pybind11;

struct PolarisEngine {
    common_params          params;
    common_init_result     init;
    size_t n_past = 0;
    llama_model          * model = nullptr;
    llama_context        * ctx   = nullptr;
    const llama_vocab    * vocab = nullptr;
    int safety_margin = 16;

    std::unique_ptr<common_sampler, void(*)(common_sampler*)> smpl{nullptr, common_sampler_free};
    common_chat_templates_ptr chat_tmpl;

    std::mutex mtx;

    // helper env
    static int env_int(const char *k, int defv) {
        if (const char *v = std::getenv(k)) { try { return std::max(1, std::stoi(v)); } catch (...) {} }
        return defv;
    }

    PolarisEngine(const std::string & model_path,
                int n_ctx = 4096,
                int n_threads = 0,
                int n_gpu_layers = -1) {

        params = common_params{};
        params.model.path = model_path;

        if (n_ctx > 0) params.n_ctx = n_ctx;

        if (n_threads > 0) {
            params.cpuparams.n_threads       = n_threads;
            params.cpuparams_batch.n_threads = n_threads;
        }

        // ================================
        // XCT MODE: COMPLETADOR CRU
        // ================================
        params.conversation_mode     = COMMON_CONVERSATION_MODE_DISABLED;
        params.enable_chat_template  = false;
        params.use_jinja             = false;
        params.chat_template         = "";

        // GPU layers
        if (n_gpu_layers == -1) {
            const char *env_val = std::getenv("POLARIS_N_GPU_LAYERS");
            if (env_val) {
                try { params.n_gpu_layers = std::stoi(env_val); }
                catch (...) { params.n_gpu_layers = 999; }
            } else {
                params.n_gpu_layers = 999;
            }
        } else {
            params.n_gpu_layers = n_gpu_layers;
        }

        // batch & ubatch
        params.n_batch  = env_int("POLARIS_BATCH", 256);
        params.n_ubatch = env_int("POLARIS_UBATCH", 128);
        safety_margin   = env_int("POLARIS_SAFETY", 16);

        // init llama.cpp backend
        common_init();
        init  = common_init_from_params(params);

        model = init.model.get();
        ctx   = init.context.get();

        if (!model || !ctx)
            throw std::runtime_error("Falha ao carregar modelo/contexto");

        vocab = llama_model_get_vocab(model);

        // ================================
        // NUNCA inicializa chat templates
        // ================================
        chat_tmpl = nullptr;

        // sampler normal
        smpl.reset(common_sampler_init(model, params.sampling));
        if (!smpl)
            throw std::runtime_error("Falha ao inicializar sampler");
    }

    ~PolarisEngine() {
        llama_backend_free();
    }

    struct SamplerCfg { float temp, top_p, rep; };
    SamplerCfg last_cfg{ -1.f, -1.f, -1.f };

    std::string generate(const std::string & prompt,
                         const std::string & system_prompt,
                         int n_predict,
                         double temperature,
                         double top_p,
                         double repeat_penalty,
                         py::object py_callback) {
        std::lock_guard<std::mutex> lock(mtx);

        // --- helpers ENV / flags de diagnóstico ---
        auto getenv_str = [](const char* k) -> std::string {
            const char* v = std::getenv(k);
            return v ? std::string(v) : std::string();
        };
        auto getenv_bool = [&](const char* k, bool defv) -> bool {
            const char* v = std::getenv(k);
            if (!v) return defv;
            std::string s(v);
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return (s=="1"||s=="true"||s=="yes"||s=="on");
        };

        const bool reset_kv = getenv_bool("POLARIS_RESET_KV", true);

        if (reset_kv) {
            auto * mem = llama_get_memory(ctx);
            llama_memory_clear(mem, /*keep_meta=*/false);
            n_past = 0;
        }

        const bool disable_tmpl  = getenv_bool("POLARIS_DISABLE_TEMPLATE", false);
        const bool force_no_spec = !getenv_bool("POLARIS_USE_SPECIALS", true);
        const std::string STAGE  = getenv_str("POLARIS_STAGE"); // "", "prompt","tokenize","prefill","sample","piece","push"

        params.n_predict               = n_predict > 0 ? n_predict : 256;
        params.sampling.temp           = (float) (temperature    > 0.0 ? temperature    : 0.7);
        params.sampling.top_p          = (float) (top_p          > 0.0 ? top_p          : 0.9);
        params.sampling.penalty_repeat = (float) (repeat_penalty > 0.0 ? repeat_penalty : 1.1);

        SamplerCfg cfg{ params.sampling.temp, params.sampling.top_p, params.sampling.penalty_repeat };
        if (!smpl || cfg.temp != last_cfg.temp || cfg.top_p != last_cfg.top_p || cfg.rep != last_cfg.rep) {
            smpl.reset(common_sampler_init(model, params.sampling));
            if (!smpl) throw std::runtime_error("Falha ao (re)configurar sampler");
            last_cfg = cfg;
        }

        // ================================
        // XCT MODE: ChatML mínimo manual
        // - sem template apply
        // - sem jinja
        // - só trilho pro Qwen
        // ================================

        std::string prompt_text;

        if (!system_prompt.empty()) {
            prompt_text += "<|im_start|>system\n";
            prompt_text += system_prompt;
            prompt_text += "\n<|im_end|>\n";
        }

        prompt_text += "<|im_start|>user\n";
        prompt_text += prompt;
        prompt_text += "\n<|im_end|>\n";

        prompt_text += "<|im_start|>assistant\n";

        if (STAGE == "prompt")
            return prompt_text;

        // Tokenização correta pro Qwen3
        // - specials ON
        // - BOS conforme vocab (false)

        const bool use_specials = true;        // Qwen precisa disso
        const bool add_bos_tok = llama_vocab_get_add_bos(vocab); // vai dar false

        LOG_INF("tokenize: use_specials=%d | add_bos=%d\n",
        (int)use_specials, (int)add_bos_tok);


        std::vector<llama_token> embd_inp = common_tokenize(ctx, prompt_text, add_bos_tok, use_specials);
        LOG_INF("tokenize: produced %zu tokens\n", embd_inp.size());
        if (embd_inp.empty()) {
            if (add_bos_tok) embd_inp.push_back(llama_vocab_bos(vocab));
            else throw std::runtime_error("Entrada vazia após tokenização");
        }
        if (STAGE == "tokenize") return std::string("[OK] tokenize: ") + std::to_string(embd_inp.size()) + " toks";

        // limites de contexto e aparo preventivo do prompt para caber com margem
        const int n_ctx_local = llama_n_ctx(ctx);
        if ((int) embd_inp.size() > n_ctx_local - safety_margin) {
            const int keep = n_ctx_local - safety_margin;
            embd_inp.erase(embd_inp.begin(), embd_inp.end() - keep);
            LOG_WRN("prompt aparado para %d tokens para caber no contexto\n", keep);
        }

        // ---- push_tokens SEGURO com llama_batch_init/free ----
        auto push_tokens = [&](const std::vector<llama_token> &toks) {
            const int n_ctx_here  = llama_n_ctx(ctx);
            int       ubatch      = params.n_ubatch > 0 ? params.n_ubatch : 128;
            const int MIN_UB      = 16;

            py::gil_scoped_release release;

            for (int i = 0; i < (int) toks.size(); ) {
                int room = n_ctx_here - safety_margin - (int) n_past;
                if (room <= 0) {
                    LOG_WRN("room<=0 ao empurrar %d; n_ctx=%d safety=%d n_past=%zu\n",
                            (int)toks.size() - i, n_ctx_here, safety_margin, n_past);
                    throw std::runtime_error("Sem espaço no contexto (room<=0)");
                }

                int n_eval = std::min({ (int) toks.size() - i, ubatch, room });

                // cria batch com capacidade n_eval, 0 emb, 1 seq
                llama_batch batch = llama_batch_init(n_eval, 0, 1);
                batch.n_tokens = n_eval;
                for (int k = 0; k < n_eval; ++k) {
                    batch.token[k]     = toks[i + k];
                    batch.pos[k]       = (int) (n_past + k);
                    batch.logits[k]    = (k == n_eval - 1);
                    batch.n_seq_id[k]  = 1;
                    batch.seq_id[k][0] = 0; // seq única
                }

                bool ok = false;
                int  try_ub = n_eval;
                while (!ok) {
                    int rc = llama_decode(ctx, batch);
                    if (rc == 0) {
                        ok = true;
                        break;
                    }
                    // backoff: diminui o tamanho do batch
                    llama_batch_free(batch);

                    try_ub = std::max(MIN_UB, try_ub/2);
                    if (try_ub == n_eval) {
                        throw std::runtime_error("llama_decode falhou (mesmo após backoff)");
                    }

                    n_eval = std::min(try_ub, room);
                    batch  = llama_batch_init(n_eval, 0, 1);
                    batch.n_tokens = n_eval;
                    for (int k = 0; k < n_eval; ++k) {
                        batch.token[k]     = toks[i + k];
                        batch.pos[k]       = (int) (n_past + k);
                        batch.logits[k]    = (k == n_eval - 1);
                        batch.n_seq_id[k]  = 1;
                        batch.seq_id[k][0] = 0;
                    }
                }

                llama_batch_free(batch);

                n_past += n_eval;
                i      += n_eval;
            }
        };

        // ---- PREFILL ----
        auto t_prefill0 = std::chrono::steady_clock::now();
        push_tokens(embd_inp);
        auto t_prefill1 = std::chrono::steady_clock::now();
        double prefill_sec = std::chrono::duration<double>(t_prefill1 - t_prefill0).count();
        LOG_INF("prefill: %zu toks em %.3fs (%.1f tok/s)\n",
                embd_inp.size(), prefill_sec,
                embd_inp.size() ? (embd_inp.size()/std::max(1e-9, prefill_sec)) : 0.0);
        for (auto t : embd_inp) common_sampler_accept(smpl.get(), t, /*grammar*/false);
        if (STAGE == "prefill") return std::string("[OK] prefill in ") + std::to_string(prefill_sec) + "s";

        // ---- ROOM PÓS-PREFILL ----
        int room = n_ctx_local - safety_margin - (int) n_past;
        if (room <= 0) {
            LOG_WRN("sem espaço para decodificar (room<=0) após prefill; n_ctx=%d safety=%d n_past=%zu embd=%zu\n",
                    n_ctx_local, safety_margin, n_past, embd_inp.size());
            return std::string{};
        }

        // ---- estágios de diagnóstico (sample/piece/push) ----
        if (STAGE == "sample" || STAGE == "piece" || STAGE == "push") {
            llama_token sid;
            {
                py::gil_scoped_release release;
                sid = common_sampler_sample(smpl.get(), ctx, -1);
            }
            if (STAGE == "sample") return std::string("[OK] sample id=") + std::to_string(sid);

            std::string piece = common_token_to_piece(ctx, sid, params.special);
            if (STAGE == "piece") return std::string("[OK] piece len=") + std::to_string(piece.size());

            {
                std::vector<llama_token> one{ sid };
                push_tokens(one);
            }
            return std::string("[OK] push one; piece len=") + std::to_string(piece.size());
        }

        // ---- clamp n_remain ----
        int n_remain = (n_predict > 0 ? n_predict : 256);
        if (room < n_remain) {
            n_remain = room;
            LOG_WRN("reduzindo n_predict para %d para não estourar contexto\n", n_remain);
        }

        // ---- geração ----
        std::string out;
        std::string buf;

        const size_t FLUSH_BYTES  = (size_t) env_int("POLARIS_FLUSH",   64);  // bytes
        const int    TOK_FLUSH    = env_int("POLARIS_TOKFLUSH",          1);  // a cada N tokens
        const int    MS_FLUSH     = env_int("POLARIS_MS_FLUSH",        100);  // flush temporal (ms)

        auto flush_cb = [&](bool /*force*/=false) {
            if (!py_callback.is_none() && (!buf.empty())) {
                py::gil_scoped_acquire acquire;
                py::bytes b(buf.data(), (py::ssize_t) buf.size());
                py_callback(b);
                buf.clear();
            }
        };

        size_t toks_generated   = 0;
        size_t tok_since_flush  = 0;
        auto   t_decode0        = std::chrono::steady_clock::now();
        auto   t_last50         = t_decode0;
        auto   t_last_flush     = t_decode0;

        const int MAX_STEPS = std::max(1, n_remain); // limite duro
        int steps = 0;

        auto json_complete = [](const std::string &s) -> bool {
            int braces = 0;
            bool in_str = false;
            bool esc = false;
            bool started = false;

            for (char c : s) {
                if (esc) { esc = false; continue; }
                if (c == '\\') { esc = true; continue; }

                if (c == '"') {
                    in_str = !in_str;
                    continue;
                }
                if (in_str) continue;

                if (c == '{') {
                    braces++;
                    started = true;
                }
                if (c == '}') {
                    braces--;
                }
            }

            return started && braces == 0 && !in_str;
        };


        while (n_remain > 0 && steps < MAX_STEPS) {

            // --- sample next token ---
            llama_token id;
            {
                py::gil_scoped_release release;
                id = common_sampler_sample(smpl.get(), ctx, -1);
            }

            // accept sampled token
            common_sampler_accept(smpl.get(), id, /*grammar*/false);

            // stop if end-of-generation token
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }

            // convert token -> text piece
            std::string piece = common_token_to_piece(ctx, id, params.special);

            if (!py_callback.is_none()) {

                buf += piece;
                out += piece;

                // STOP cedo do XCT
                bool has_key =
                    (out.find("\"done\"")      != std::string::npos) ||
                    (out.find("\"next_step\"") != std::string::npos);

                if (has_key && json_complete(out)) {
                    flush_cb(true);
                    break;
                }

            } else {

                out += piece;

                // STOP cedo do XCT
                bool has_key =
                    (out.find("\"done\"")      != std::string::npos) ||
                    (out.find("\"next_step\"") != std::string::npos);

                bool closed_json =
                    (out.find("}") != std::string::npos);

                if (has_key && json_complete(out)) {
                    break;
                }
            }


            // ============================================================
            // push generated token back into context
            // ============================================================
            {
                std::vector<llama_token> one{ id };
                push_tokens(one);
            }

            // bookkeeping
            --n_remain;
            ++toks_generated;
            ++steps;

            // perf log every 50 tokens
            if (toks_generated % 50 == 0) {
                auto now = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(now - t_last50).count();

                LOG_INF("decode: +50 toks em %.3fs (%.1f tok/s)\n",
                        dt, 50.0 / std::max(1e-9, dt));

                t_last50 = now;
            }
        }


        if (!py_callback.is_none() && !buf.empty()) {
            py::gil_scoped_acquire acquire;
            py::bytes b(buf.data(), (py::ssize_t) buf.size());
            py_callback(b);
            buf.clear();
        }

        auto t_decode1 = std::chrono::steady_clock::now();
        double decode_sec = std::chrono::duration<double>(t_decode1 - t_decode0).count();
        LOG_INF("decode: %zu toks em %.3fs (%.1f tok/s)\n",
                toks_generated, decode_sec,
                toks_generated ? (toks_generated/std::max(1e-9, decode_sec)) : 0.0);

        return out;
    }
};

PYBIND11_MODULE(polaris_core, m) {
    py::class_<PolarisEngine>(m, "Engine")
        .def(py::init<const std::string&, int, int, int>(),
             py::arg("model_path"),
             py::arg("n_ctx") = 4096,
             py::arg("n_threads") = 0,
             py::arg("n_gpu_layers") = -1)
        .def("generate",
             &PolarisEngine::generate,
             py::arg("prompt"),
             py::arg("system_prompt") = "",
             py::arg("n_predict") = 256,
             py::arg("temperature") = 0.7,
             py::arg("top_p") = 0.9,
             py::arg("repeat_penalty") = 1.1,
             py::arg("callback") = py::none(),
             "Gera texto; se callback for passado, faz streaming por chunk.");
}
