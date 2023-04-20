#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void *gpt2_allocate_state();

int gpt2_bootstrap(const char *model_path, void *state_pr);

void* gpt2_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, int n_batch);
void gpt2_free_params(void* params_ptr);
void gpt2_free_model(void* state_ptr);

int gpt2_predict(void* params_ptr, void* state_pr, char* result);

#ifdef __cplusplus
}
#endif