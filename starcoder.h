#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void *starcoder_allocate_state();

int starcoder_bootstrap(const char *model_path, void *state_pr);

void* starcoder_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, int n_batch);
void starcoder_free_params(void* params_ptr);
void starcoder_free_model(void* state_ptr);

int starcoder_predict(void* params_ptr, void* state_pr, char* result);

#ifdef __cplusplus
}
#endif