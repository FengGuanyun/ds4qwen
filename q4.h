#ifndef Q4_H
#define Q4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat q4_engine as the loaded model and
 * q4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * q4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    Q4_BACKEND_METAL,
    Q4_BACKEND_CUDA,
    Q4_BACKEND_CPU,
} q4_backend;

typedef enum {
    Q4_LOG_DEFAULT,
    Q4_LOG_PREFILL,
    Q4_LOG_GENERATION,
    Q4_LOG_KVCACHE,
    Q4_LOG_WARNING,
    Q4_LOG_TIMING,
    Q4_LOG_OK,
    Q4_LOG_ERROR,
} q4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} q4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} q4_token_score;

#define Q4_DEFAULT_TEMPERATURE 0.6f
#define Q4_DEFAULT_TOP_P       0.95f
#define Q4_DEFAULT_MIN_P       0.01f
#define Q4_DEFAULT_TOP_K       20

typedef struct q4_engine q4_engine;
typedef struct q4_session q4_session;

typedef struct {
    const char *model_path;
    q4_backend backend;
    int n_threads;
    bool warm_weights;
} q4_engine_options;

typedef void (*q4_token_emit_fn)(void *ud, int token);
typedef void (*q4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
} q4_context_memory;

int q4_engine_open(q4_engine **out, const q4_engine_options *opt);
void q4_engine_close(q4_engine *e);
void q4_engine_summary(q4_engine *e);
const char *q4_backend_name(q4_backend backend);
q4_context_memory q4_context_memory_estimate(q4_backend backend, int ctx_size);

bool q4_log_is_tty(FILE *fp);
void q4_log(FILE *fp, q4_log_type type, const char *fmt, ...);

int q4_engine_generate_argmax(q4_engine *e, const q4_tokens *prompt,
                               int n_predict, int ctx_size,
                               q4_token_emit_fn emit,
                               q4_generation_done_fn done,
                               void *emit_ud);

int q4_engine_tokenize(q4_engine *e, const char *text, int text_len,
                        q4_tokens **out_tokens);
void q4_tokens_free(q4_tokens *tokens);

/* Session API: incremental inference with KV cache reuse. */
int q4_session_create(q4_session **out, q4_engine *e, int ctx_size);
void q4_session_free(q4_session *s);
int q4_session_sync(q4_session *s, const q4_tokens *prompt, char *err, size_t errlen);
int q4_session_eval(q4_session *s, int token, char *err, size_t errlen);
int q4_session_argmax(q4_session *s, char *err, size_t errlen);
int q4_session_sample(q4_session *s, float temperature, int top_k,
                       float top_p, float min_p, uint64_t *rng,
                       char *err, size_t errlen);
int q4_session_n_tokens(q4_session *s);
void q4_session_checkpoint(q4_session *s);
const float *q4_session_logits(q4_session *s);

/* Token utilities. */
char *q4_token_text(q4_engine *e, int token, size_t *out_len);
int q4_token_eos(q4_engine *e);
void q4_tokens_push(q4_tokens *tokens, int token);

/* Session position/context helpers. */
int q4_session_pos(q4_session *s);
int q4_session_ctx(q4_session *s);

#endif /* Q4_H */
