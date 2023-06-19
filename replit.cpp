// https://github.com/ggerganov/ggml/pull/145
#include "ggml.h"

#include "common-ggml.h"
#include "common.h"
#include "replit.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "ggml.cpp/examples/replit/main.cpp"


struct replit_state {
    replit_tokenizer vocab;
    replit_model model;
    struct {
        int64_t t_load_us = -1;
        int64_t t_sample_us = -1;
        int64_t t_predict_us = -1;
    } timing;
};

int replit_predict(void* params_ptr, void* state_pr, char* result) {
    gpt_params params = *(gpt_params*) params_ptr;
    replit_state state = *(replit_state*) state_pr;
    replit_tokenizer vocab = state.vocab;
    replit_model model = state.model;
  const int64_t t_main_start_us = ggml_time_us();


  if (params.seed < 0) {
    params.seed = time(NULL);
  }

  std::mt19937 rng(params.seed);
  

  int64_t t_load_us = 0;

  int n_past = 0;

  int64_t t_sample_us = 0;
  int64_t t_predict_us = 0;

  std::vector<float> logits;

  // tokenize the prompt
  std::vector<std::size_t> embd_inp =
      replit_tokenizer_tokenize(vocab, params.prompt);

  printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());

  for (int i = 0; i < embd_inp.size(); i++) {
    printf("%s: token[%d] = %6d\n", __func__, i, embd_inp[i]);
    // vocab.id_to_token.at(embd_inp[i]).c_str()
  }
  printf("\n");

  params.n_predict = std::min(params.n_predict,
                              model.hparams.max_seq_len - (int)embd_inp.size());

  std::vector<gpt_vocab::id> embd;
    std::string res = "";

  // determine the required inference memory per token:
  size_t mem_per_token = 0;
  replit_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, false, mem_per_token);

  for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
    // predict
    if (embd.size() > 0) {
      const int64_t t_start_us = ggml_time_us();
      
      if (!replit_eval(model, params.n_threads, n_past, embd, logits, false, mem_per_token)) {
        printf("Failed to predict\n");
        return 1;
      }

      t_predict_us += ggml_time_us() - t_start_us;
    }

    n_past += embd.size();
    embd.clear();

    if (i >= embd_inp.size()) {
      // sample next token
      const int top_k = params.top_k;
      const float top_p = params.top_p;
      const float temp = params.temp;

      const int n_vocab = model.hparams.n_vocab;

      gpt_vocab::id id = 0;

      {
        const int64_t t_start_sample_us = ggml_time_us();

        id = gpt_sample_top_k_top_p(vocab.raw_vocab,
                                    logits.data() + (logits.size() - n_vocab),
                                    top_k, top_p, temp, rng);

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
        res += replit_tokenizer_detokenize(vocab, {static_cast<std::size_t>(id)})
                 .c_str();
    }

    // end of text token
    if (embd.back() == 0) {
      break;
    }
  }
strcpy(result, res.c_str()); 
  return 0;
}


int replit_bootstrap(const char *model_path, void* state_pr)
// load the model
{
    ggml_time_init();
    replit_state* state = (replit_state*) state_pr;

    const int64_t t_start_us = ggml_time_us();
    if (!replit_model_load(model_path, state->model, state->vocab)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model_path);
        return 1;
    }

    state->timing.t_load_us = ggml_time_us() - t_start_us;
    return 0;
}

void* replit_allocate_state() {
    return new replit_state;
}

void replit_free_model(void *state_ptr) {
    replit_state* state = (replit_state*) state_ptr;
    ggml_free(state->model.ctx);
}

void replit_free_params(void* params_ptr) {
    gpt_params* params = (gpt_params*) params_ptr;
    delete params;
}

void* replit_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
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
