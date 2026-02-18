#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    // 1) Parse de args padrão do repo
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, /*print_usage=*/nullptr)) {
        return 1;
    }

    // queremos comportamento igual ao cli: template liga conversa automaticamente
    params.conversation_mode = COMMON_CONVERSATION_MODE_AUTO;

    // 2) Inicialização padrão (backend, modelo, contexto, LoRA, etc.)
    common_init();
    common_init_result init = common_init_from_params(params);
    llama_model  * model = init.model.get();
    llama_context* ctx   = init.context.get();
    if (!model || !ctx) {
        LOG_ERR("erro: não carregou modelo/contexto\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 3) Chat templates (iguais ao cli)
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    // 4) Construção do prompt final
    std::vector<common_chat_msg> chat_msgs;
    std::string prompt_text;

    bool using_chat = false;
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        // auto-detecta template e liga conversa
        params.conversation_mode = common_chat_templates_was_explicit(chat_templates.get())
            ? COMMON_CONVERSATION_MODE_ENABLED
            : COMMON_CONVERSATION_MODE_DISABLED;
    }
    if (params.conversation_mode && params.enable_chat_template) {
        using_chat = true;
        if (!params.system_prompt.empty()) {
            chat_msgs.push_back({"system", params.system_prompt});
        }
        if (!params.prompt.empty()) {
            chat_msgs.push_back({"user", params.prompt});
        }
        common_chat_templates_inputs inputs;
        inputs.use_jinja = params.use_jinja;
        inputs.messages  = chat_msgs;
        inputs.add_generation_prompt = !params.prompt.empty();
        prompt_text = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
    } else {
        // texto plano
        prompt_text = params.prompt;
    }

    // 5) Tokenização — via common_tokenize (faz o certo p/ specials/BOS)
    const bool use_specials = using_chat; // só usa specials se estamos com template
    const bool add_bos_tok  = llama_vocab_get_add_bos(vocab) && !use_specials;
    std::vector<llama_token> embd_inp = common_tokenize(ctx, prompt_text, add_bos_tok, use_specials);

    if (embd_inp.empty()) {
        // fallback idêntico ao cli: se vazio, tenta BOS
        if (add_bos_tok) {
            embd_inp.push_back(llama_vocab_bos(vocab));
        } else {
            LOG_ERR("input vazio após tokenização\n");
            return 1;
        }
    }

    // 6) Sampler (usa a cadeia do repo)
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("erro: sampler\n");
        return 1;
    }

    // 7) Console igual ao cli (cores/opções)
    console::init(params.simple_io, params.use_color);
    atexit([](){ console::cleanup(); });

    // 8) Empurra o prompt
    int n_past   = 0;
    int n_ctx    = llama_n_ctx(ctx);
    int n_batch  = params.n_batch;
    auto push_tokens = [&](const std::vector<llama_token> &toks) {
        for (int i = 0; i < (int)toks.size(); i += n_batch) {
            int n_eval = std::min((int)toks.size() - i, n_batch);
            if (llama_decode(ctx, llama_batch_get_one((llama_token*)&toks[i], n_eval))) {
                LOG_ERR("llama_decode falhou\n"); return false;
            }
            n_past += n_eval;
        }
        return true;
    };
    if (!push_tokens(embd_inp)) { common_sampler_free(smpl); return 1; }
    for (llama_token t : embd_inp) common_sampler_accept(smpl, t, /*grammar*/false);

    // 9) Geração (streaming simples)
    int n_remain = params.n_predict;
    while (n_remain != 0) {
        // amostra próximo id
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, /*grammar*/true);

        // quebra com EOG
        if (llama_vocab_is_eog(vocab, id)) break;

        // imprime token
        std::string piece = common_token_to_piece(ctx, id, params.special);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);

        // empurra de volta
        std::vector<llama_token> one{ id };
        if (!push_tokens(one)) break;

        --n_remain;
        if (n_past >= n_ctx - 4) break; // igual ao cli: margem p/ logits
    }

    std::puts("");

    // 10) métricas e limpeza
    common_perf_print(ctx, smpl);
    common_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
