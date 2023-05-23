#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void * gptj_allocate_state();

int  gptj_bootstrap(const char *model_path, void *state_pr);

void*  gptj_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, int n_batch);
void  gptj_free_params(void* params_ptr);
void  gptj_free_model(void* state_ptr);

int  gptj_predict(void* params_ptr, void* state_pr, char* result);

#ifdef __cplusplus
}
#endif