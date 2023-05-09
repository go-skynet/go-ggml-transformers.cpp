#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void *dolly_allocate_state();

int dolly_bootstrap(const char *model_path, void *state_pr);

void* dolly_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, int n_batch);
void dolly_free_params(void* params_ptr);
void dolly_free_model(void* state_ptr);

int dolly_predict(void* params_ptr, void* state_pr, char* result);

#ifdef __cplusplus
}
#endif