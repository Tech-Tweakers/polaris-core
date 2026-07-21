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

#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <cstdlib>   // getenv
#include <chrono>    // métricas / timer de flush
#include <cctype>    // tolower

namespace py = pybind11;

struct PolarisEngine {
    common_params              params;
    common_init_result_ptr     init;
    size_t n_past = 0;
    llama_model          * model = nullptr;
    llama_context        * ctx   = nullptr;
    const llama_vocab    * vocab = nullptr;
    int safety_margin = 16;

    common_sampler_ptr smpl;
    common_chat_templates_ptr chat_tmpl;

    std::mutex mtx;

    // helper env
    static int env_int(const char *k, int defv) {
        if (const char *v = std::getenv(k)) { try { return std::max(1, std::stoi(v)); } catch (...) {} }
        return defv;
    }

    static bool env_bool(const char *k, bool defv) {
        const char *v = std::getenv(k);
        if (!v) return defv;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s == "1" || s == "true" || s == "yes" || s == "on";
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
        params.special  = env_bool("POLARIS_SPECIAL", true);
        safety_margin   = env_int("POLARIS_SAFETY", 16);

        // init llama.cpp backend
        common_init();
        init  = common_init_from_params(params);

        model = init->model();
        ctx   = init->context();

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

    struct SamplerCfg { float temp, top_p, rep, topk, minp, freq, pres; int seed; std::string grammar; };
    SamplerCfg last_cfg{ -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1 };

    // generate: a forma antiga (system + um unico user). Mantida porque e assim
    // que metade das chamadas ja existe. Por baixo delega pro caminho novo.
    // ChatMsg: uma mensagem com o papel preservado. E um std::pair de proposito:
    // pybind11/stl.h converte tupla/lista do Python direto, sem registrar tipo.
    using ChatMsg = std::pair<std::string, std::string>;  // (role, content)

    std::string generate(const std::string & prompt,
                         const std::string & system_prompt,
                         int n_predict,
                         double temperature,
                         double top_p,
                         double repeat_penalty,
                         int    top_k,
                         double min_p,
                         double penalty_freq,
                         double penalty_present,
                         int    seed,
                         const std::string & grammar,
                         py::object py_callback) {
        std::vector<ChatMsg> msgs;
        if (!system_prompt.empty()) msgs.emplace_back("system", system_prompt);
        msgs.emplace_back("user", prompt);
        return generate_chat(msgs, n_predict, temperature, top_p, repeat_penalty,
                             top_k, min_p, penalty_freq, penalty_present, seed,
                             grammar, py_callback);
    }

    // generate_chat: a conversa com os PAPEIS preservados.
    //
    // Por que existe: antes o historico inteiro — os turnos anteriores, as
    // chamadas que o proprio modelo fez, os resultados que voltaram — era
    // concatenado DENTRO de um unico <|im_start|>user. O modelo nao tinha como
    // saber que ELE havia feito aquelas chamadas: pra ele o usuario escreveu um
    // texto gigante que por acaso continha tool_calls. Consequencia medida em
    // 20-21/07: ele nao percebia que ja tinha investigado e seguia pedindo tool
    // pra sempre, e a regra do system afundava a tres turnos de distancia
    // dentro do mesmo bloco. Um bloco ChatML por mensagem devolve o turno.
    std::string generate_chat(const std::vector<ChatMsg> & messages,
                              int n_predict,
                              double temperature,
                              double top_p,
                              double repeat_penalty,
                              int    top_k,
                              double min_p,
                              double penalty_freq,
                              double penalty_present,
                              int    seed,
                              const std::string & grammar,
                              py::object py_callback) {
        std::lock_guard<std::mutex> lock(mtx);

        // --- helpers ENV / flags de diagnóstico ---
        auto getenv_str = [](const char* k) -> std::string {
            const char* v = std::getenv(k);
            return v ? std::string(v) : std::string();
        };

        const bool reset_kv = env_bool("POLARIS_RESET_KV", true);

        if (reset_kv) {
            auto * mem = llama_get_memory(ctx);
            llama_memory_clear(mem, /*keep_meta=*/false);
            n_past = 0;
        }

        const std::string STAGE  = getenv_str("POLARIS_STAGE"); // "", "prompt","tokenize","prefill","sample","piece","push"

        params.n_predict                 = n_predict > 0 ? n_predict : 256;
        params.sampling.temp             = (float)(temperature     > 0.0  ? temperature     : 0.7);
        params.sampling.top_p            = (float)(top_p           > 0.0  ? top_p           : 0.9);
        params.sampling.penalty_repeat   = (float)(repeat_penalty  > 0.0  ? repeat_penalty  : 1.1);
        params.sampling.top_k            = top_k > 0 ? top_k : 40;
        params.sampling.min_p            = (float)(min_p           >= 0.0 ? min_p           : 0.05);
        params.sampling.penalty_freq     = (float)(penalty_freq    >= 0.0 ? penalty_freq    : 0.0);
        params.sampling.penalty_present  = (float)(penalty_present >= 0.0 ? penalty_present : 0.0);
        if (seed >= 0) params.sampling.seed = (uint32_t)seed;

        // Grammar (GBNF): quando o cliente manda, o sampler ZERA a probabilidade
        // de qualquer token que quebre a gramática — o modelo fica IMPEDIDO de
        // emitir formato inválido, em vez de a gente pedir por favor no prompt e
        // torcer. É assim que os provedores de nuvem garantem tool-call bem
        // formado. A Polaris não sabe o que a gramática significa (pode ser
        // tool-call, JSON, o que for): ela só constrange. Quem conhece as tools
        // é o cliente (o XCT vive lá).
        params.sampling.grammar = grammar;

        SamplerCfg cfg{
            params.sampling.temp,
            params.sampling.top_p,
            params.sampling.penalty_repeat,
            (float)params.sampling.top_k,
            params.sampling.min_p,
            params.sampling.penalty_freq,
            params.sampling.penalty_present,
            seed,
            grammar
        };
        if (!smpl || cfg.temp != last_cfg.temp || cfg.top_p != last_cfg.top_p ||
            cfg.rep != last_cfg.rep || cfg.topk != last_cfg.topk ||
            cfg.minp != last_cfg.minp || cfg.freq != last_cfg.freq ||
            cfg.pres != last_cfg.pres || cfg.seed != last_cfg.seed ||
            cfg.grammar != last_cfg.grammar) {
            smpl.reset(common_sampler_init(model, params.sampling));
            if (!smpl) throw std::runtime_error("Falha ao (re)configurar sampler");
            last_cfg = cfg;
        } else {
            // Config igual à da chamada anterior: o sampler é REUSADO — e ele
            // carrega estado. A janela de penalidade (repeat/freq/presence)
            // guarda os últimos N tokens gerados, e mais abaixo alimentamos o
            // prompt inteiro com common_sampler_accept(). Sem limpar, a geração
            // nova começa penalizando tokens do turno ANTERIOR.
            //
            // O sintoma é característico: uma resposta boa, a seguinte ruim,
            // alternando — porque a config só muda de vez em quando e, quando
            // muda, o sampler é recriado limpo por acaso.
            common_sampler_reset(smpl.get());
        }

        // ================================
        // XCT MODE: ChatML mínimo manual
        // - sem template apply
        // - sem jinja
        // - só trilho pro Qwen
        // ================================

        std::string prompt_text;

        for (const auto & m : messages) {
            if (m.second.empty()) continue;
            // Papel fora do ChatML vira "user": o Qwen so conhece
            // system/user/assistant, e um papel inventado quebra o trilho.
            std::string role = m.first;
            if (role != "system" && role != "user" && role != "assistant") role = "user";
            prompt_text += "<|im_start|>";
            prompt_text += role;
            prompt_text += "\n";
            prompt_text += m.second;
            prompt_text += "\n<|im_end|>\n";
        }

        prompt_text += "<|im_start|>assistant\n";

        // Dial por MODELO, não global: a família Qwen3.5 emite bloco
        // <think> espontaneamente; injetar um <think></think> vazio e já
        // fechado suprime isso. Mas o Qwen3 NÃO usa think — nele o bloco
        // pré-fechado faz o modelo concluir que o turno acabou e emitir
        // EOS de cara (decode: 0 toks, resposta vazia). Por isso é opt-in:
        // ligue POLARIS_SUPPRESS_THINK=1 só ao servir um modelo 3.5.
        if (env_bool("POLARIS_SUPPRESS_THINK", false)) {
            prompt_text += "<think>\n\n</think>\n";
        }

        if (STAGE == "prompt")
            return prompt_text;

        // Tokenização correta pro Qwen3
        // - specials ON
        // - BOS conforme vocab (false)

        const bool use_specials = params.special;
        const bool add_bos_tok = llama_vocab_get_add_bos(vocab);

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

            auto make_batch = [&](int n, int offset) {
                llama_batch batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (int k = 0; k < n; ++k) {
                    batch.token[k]     = toks[offset + k];
                    batch.pos[k]       = (int) (n_past + k);
                    batch.logits[k]    = (k == n - 1);
                    batch.n_seq_id[k]  = 1;
                    batch.seq_id[k][0] = 0; // seq única
                }
                return batch;
            };

            for (int i = 0; i < (int) toks.size(); ) {
                int room = n_ctx_here - safety_margin - (int) n_past;
                if (room <= 0) {
                    LOG_WRN("room<=0 ao empurrar %d; n_ctx=%d safety=%d n_past=%zu\n",
                            (int)toks.size() - i, n_ctx_here, safety_margin, n_past);
                    throw std::runtime_error("Sem espaço no contexto (room<=0)");
                }

                int n_eval = std::min({ (int) toks.size() - i, ubatch, room });

                llama_batch batch = make_batch(n_eval, i);

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

                    try_ub = std::max(MIN_UB, try_ub / 2);
                    int next_n_eval = std::min(try_ub, room);
                    if (next_n_eval >= n_eval) {
                        throw std::runtime_error("llama_decode falhou (backoff esgotado)");
                    }
                    n_eval = next_n_eval;

                    batch = make_batch(n_eval, i);
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
        // accept_grammar: precisa ser true quando há gramática, senão o estado dela
        // não avança e o constraint não vale. No prompt (embd_inp) segue false —
        // a gramática vale para o que o MODELO gera, não para o que ele leu.
        const bool use_grammar = !grammar.empty();

        // Tokens ESPECIAIS de tool-call. No Qwen, <tool_call> e </tool_call> são
        // tokens ÚNICOS do vocabulário — não strings montadas caractere a
        // caractere. Detectá-los aqui, no momento em que o modelo os emite, é o
        // que os provedores de nuvem fazem: a chamada NUNCA vira texto, então
        // nunca "vaza" no chat nem depende de regex pra ser reconhecida.
        // -1 = o modelo não tem esses tokens (aí seguimos no modo texto).
        llama_token tok_call_start = -1, tok_call_end = -1;
        {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (int i = 0; i < n_vocab; i++) {
                const char * tp = llama_vocab_get_text(vocab, i);
                if (!tp) continue;
                if (tok_call_start < 0 && std::strcmp(tp, "<tool_call>") == 0)  tok_call_start = i;
                if (tok_call_end   < 0 && std::strcmp(tp, "</tool_call>") == 0) tok_call_end   = i;
                if (tok_call_start >= 0 && tok_call_end >= 0) break;
            }
        }
        // Tool-call tokens são detectados mas, por enquanto, seguem no fluxo de
        // texto normal: separá-los de verdade exige protocolar o canal de
        // streaming. A detecção por token continua aqui para logs futuros.
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

        auto flush_cb = [&](bool force=false) {
            if (!py_callback.is_none() && (!buf.empty() || force)) {
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
            int braces = 0, brackets = 0;
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

                if (c == '{') { braces++; started = true; }
                else if (c == '}') { braces--; }
                else if (c == '[') { brackets++; started = true; }
                else if (c == ']') { brackets--; }
            }

            return started && braces == 0 && brackets == 0 && !in_str;
        };


        while (n_remain > 0 && steps < MAX_STEPS) {

            // --- sample next token ---
            llama_token id;
            {
                py::gil_scoped_release release;
                id = common_sampler_sample(smpl.get(), ctx, -1);
            }

            // accept sampled token
            common_sampler_accept(smpl.get(), id, /*grammar*/use_grammar);

            // stop if end-of-generation token
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }

            // Tool-call por TOKEN: detectamos os tokens especiais se existirem
            // no vocabulário. Por hora o conteúdo segue no fluxo de texto; a
            // separação estruturada exige protocolar o stream no cliente.
            (void)tok_call_start; (void)tok_call_end;

            // convert token -> text piece
            std::string piece = common_token_to_piece(ctx, id, params.special);

            // NOTA: houve aqui um "canal separado" que desviava o conteúdo do
            // tool_call para um buffer próprio, anexado ao final da resposta. A
            // intenção era não deixar a chamada virar texto (como fazem os
            // provedores), mas ficou pela metade: o STREAM não via a chamada e o
            // resultado final a recebia grudada no fim — fora da posição em que
            // o modelo a emitiu. Isso desalinhava o texto e produzia fragmentos
            // soltos quando o stream cortava no meio. Revertido: o piece segue o
            // fluxo normal, na ordem gerada. A separação real exige tratar o
            // protocolo inteiro (stream + final), não só a saída.

            buf += piece;
            out += piece;

            // STOP cedo do XCT: procura sinalizadores e só para quando o JSON
            // estiver balanceado. Isso evita parada no meio de uma string
            // que por acaso contenha "done".
            bool has_key =
                (out.find("\"done\"")      != std::string::npos) ||
                (out.find("\"next_step\"") != std::string::npos);

            if (has_key && json_complete(out)) {
                flush_cb(true);
                break;
            }

            // flushing streaming
            if (!py_callback.is_none()) {
                bool by_bytes = buf.size() >= FLUSH_BYTES;
                bool by_toks  = (TOK_FLUSH > 0) && (++tok_since_flush >= (size_t)TOK_FLUSH);
                bool by_time  = false;
                if (MS_FLUSH > 0) {
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - t_last_flush).count() >= MS_FLUSH) {
                        by_time = true;
                        t_last_flush = now;
                    }
                }

                if (by_bytes || by_toks || by_time) {
                    flush_cb();
                    tok_since_flush = 0;
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
             py::arg("system_prompt")    = "",
             py::arg("n_predict")        = 256,
             py::arg("temperature")      = 0.7,
             py::arg("top_p")            = 0.9,
             py::arg("repeat_penalty")   = 1.1,
             py::arg("top_k")            = 40,
             py::arg("min_p")            = 0.05,
             py::arg("penalty_freq")     = 0.0,
             py::arg("penalty_present")  = 0.0,
             py::arg("seed")             = -1,
             py::arg("grammar")          = "",
             py::arg("callback")         = py::none(),
             "Gera texto; se callback for passado, faz streaming por chunk.")
        .def("generate_chat",
             &PolarisEngine::generate_chat,
             py::arg("messages"),
             py::arg("n_predict")        = 256,
             py::arg("temperature")      = 0.7,
             py::arg("top_p")            = 0.9,
             py::arg("repeat_penalty")   = 1.1,
             py::arg("top_k")            = 40,
             py::arg("min_p")            = 0.05,
             py::arg("penalty_freq")     = 0.0,
             py::arg("penalty_present")  = 0.0,
             py::arg("seed")             = -1,
             py::arg("grammar")          = "",
             py::arg("callback")         = py::none(),
             "Gera a partir da conversa com PAPEIS preservados: messages e uma "
             "lista de (role, content), role em {system,user,assistant}. Cada "
             "mensagem vira seu proprio bloco ChatML em vez de tudo virar um "
             "unico <|im_start|>user.");
}
