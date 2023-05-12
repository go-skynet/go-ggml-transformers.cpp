#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void * gpt_neox_allocate_state();

int  gpt_neox_bootstrap(const char *model_path, void *state_pr);

void*  gpt_neox_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, int n_batch);
void  gpt_neox_free_params(void* params_ptr);
void  gpt_neox_free_model(void* state_ptr);

int  gpt_neox_predict(void* params_ptr, void* state_pr, char* result);

#ifdef __cplusplus
}
#endif