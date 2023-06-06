// https://github.com/ggerganov/ggml/pull/139

#include "ggml.h"

#include "common.h"
#include "common-ggml.h"
#include "mpt.h"
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
#include "ggml.cpp/examples/mpt/main.cpp"


struct mpt_state {
    gpt_vocab vocab;
    mpt_model model;
    struct {
        int64_t t_load_us = -1;
        int64_t t_sample_us = -1;
        int64_t t_predict_us = -1;
    } timing;
};

int mpt_predict(void* params_ptr, void* state_pr, char* result) {
    mpt_params params = *(mpt_params*) params_ptr;
    mpt_state state = *(mpt_state*) state_pr;
    gpt_vocab vocab = state.vocab;
    mpt_model model = state.model;
    const int64_t t_main_start_us = ggml_time_us();

    
    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
   

    int64_t t_load_us = 0;


    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    if ( params.n_predict <= 0 ) {
        params.n_predict = 200;
    }

    model.hparams.n_ctx = params.n_ctx;
    
    if (params.top_k == 0) {
        params.top_k = model.hparams.n_vocab;
    }

    if (params.repeat_last_n == -1) {
        params.repeat_last_n = params.n_ctx;
    }

    std::vector<int32_t> last_n_tokens(params.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    // tokenize the prompt
    std::vector<int> embd_inp = ::gpt_tokenize(vocab, params.prompt);



    std::string res = "";

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    mpt_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, false, mem_per_token);

    int n_past     = 0;
    int n_consumed = 0;
    int n_sampled  = 0;

    while (n_sampled < params.n_predict) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!mpt_eval(model, params.n_threads, n_past, embd, logits, false, mem_per_token)) {
                printf("%s: failed to predict\n", __func__);
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
            n_past += embd.size();
            embd.clear();
        }

        if ((int)embd_inp.size() <= n_consumed) {
            // sample next token
            const int top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;
            const int repeat_last_n = params.repeat_last_n;
            const float repeat_penalty = params.repeat_penalty;


            gpt_vocab::id id = 0;

          {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p_repeat(vocab, logits.data() + (logits.size() - model.hparams.n_vocab), last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_last_n, repeat_penalty, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }


            // add it to the context
            embd.push_back(id);
            ++n_sampled;
        } else {
         // if here, it means we are still processing the input prompt
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        for (auto id : embd) {
            res += vocab.id_to_token[id].c_str();
        }
        // end of text token
        if (embd.back() == 0) {
            break;
        }
    }

    strcpy(result, res.c_str()); 

    return 0;
}


int mpt_bootstrap(const char *model_path, void* state_pr)
// load the model
{
    ggml_time_init();
     mpt_state* state = ( mpt_state*) state_pr;

    const int64_t t_start_us = ggml_time_us();
    if (! mpt_model_load(model_path, state->model, state->vocab)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model_path);
        return 1;
    }

    state->timing.t_load_us = ggml_time_us() - t_start_us;
    return 0;
}

void*  mpt_allocate_state() {
    return new  mpt_state;
}

void mpt_free_model(void *state_ptr) {
     mpt_state* state = ( mpt_state*) state_ptr;
    ggml_free(state->model.ctx);
}

void mpt_free_params(void* params_ptr) {
    mpt_params* params = (mpt_params*) params_ptr;
    delete params;
}

void* mpt_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, int n_batch) {
    mpt_params* params = new mpt_params;
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
