#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void *replit_allocate_state();

int replit_bootstrap(const char *model_path, void *state_pr);

void* replit_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, int n_batch);
void replit_free_params(void* params_ptr);
void replit_free_model(void* state_ptr);

int replit_predict(void* params_ptr, void* state_pr, char* result);

#ifdef __cplusplus
}
#endif