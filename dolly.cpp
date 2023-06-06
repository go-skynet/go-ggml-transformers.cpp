#include "ggml.h"
#include "dolly.h"

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include "ggml.cpp/examples/dolly-v2/main.cpp"

struct dolly_state {
    gpt_vocab vocab;
    dollyv2_model model;
    struct {
        int64_t t_load_us = -1;
        int64_t t_sample_us = -1;
        int64_t t_predict_us = -1;
    } timing;
};

int dolly_predict(void* params_ptr, void* state_pr, char* result) {
    gpt_params params = *(gpt_params*) params_ptr;
    dolly_state state = *(dolly_state*) state_pr;
    gpt_vocab vocab = state.vocab;
    dollyv2_model model = state.model;

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
  
    const std::string prompt = params.prompt;

    int64_t t_load_us = 0;

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());


    std::vector<gpt_vocab::id> embd;
    std::string res = "";

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    dollyv2_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    const int32_t end_token = vocab.token_to_id["### End"];

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!dollyv2_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);

        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
                        res += vocab.id_to_token[id].c_str();

        }

        // end of text token
        if (embd.back() == 0 || (end_token > 0 && embd.back() == end_token)) {
            break;
        }
    }

    // report timing
    /*
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }
    */

    strcpy(result, res.c_str()); 

    return 0;
}



int dolly_bootstrap(const char *model_path, void* state_pr)
// load the model
{
    ggml_time_init();
    dolly_state* state = (dolly_state*) state_pr;

    const int64_t t_start_us = ggml_time_us();
    if (!dollyv2_model_load(model_path, state->model, state->vocab)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model_path);
        return 1;
    }

    state->timing.t_load_us = ggml_time_us() - t_start_us;
    return 0;
}

void* dolly_allocate_state() {
    return new dolly_state;
}

void dolly_free_model(void *state_ptr) {
    dolly_state* state = (dolly_state*) state_ptr;
    ggml_free(state->model.ctx);
}

void dolly_free_params(void* params_ptr) {
    gpt_params* params = (gpt_params*) params_ptr;
    delete params;
}

void* dolly_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, int n_batch) {
    gpt_params* params = new gpt_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_predict = tokens;

    params->top_k = top_k;
    params->top_p = top_p;
    params->temp = temp;
    params->n_batch = n_batch;

    params->prompt = prompt;
    
    return params;
}
