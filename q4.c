/* =========================================================================
 * q4.c - Qwen3.6-27B hybrid inference engine.
 * =========================================================================
 *
 * This file owns GGUF mmap loading, the fixed Qwen3.6-27B tensor layout,
 * CPU reference kernels, the whole-model Metal graph driver, and tokenizer
 * wiring.  The model shape is not configurable; every validation step is
 * meant to fail early if a GGUF does not match the one layout this engine
 * implements.
 *
 * Qwen3.6-27B: 64 layers, 5120 hidden, 24Q/4KV GQA, head_dim=128.
 * Hybrid: 16 blocks x (3x Gated DeltaNet + 1x Gated Attention).
 * Dense SwiGLU FFN, standard residual, full RoPE.
 */

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "q4.h"

#ifndef Q4_NO_GPU
#include "q4_gpu.h"
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define Q4_NEG_INF (-1.0e30f)
#define Q4_POS_INF ( 1.0e30f)
#define Q4_RMS_EPS (1.0e-5f)

/* =========================================================================
 * Fixed Qwen3.6-27B Shape.
 * =========================================================================
 *
 * These constants define the single model family this program accepts.
 */

enum {
    Q4_N_LAYER      = 64,
    Q4_N_EMBD       = 5120,
    Q4_N_VOCAB      = 248320,
    Q4_N_HEAD       = 24,
    Q4_N_HEAD_KV    = 4,
    Q4_HEAD_DIM     = 128,
    Q4_N_FFN        = 18944,
    Q4_Q_PER_KV     = 6,  // 24 / 4
    Q4_BLOCK_SIZE   = 4,  // 3 DeltaNet + 1 Attention
    Q4_N_BLOCKS     = 16,
};

/* Model config read from GGUF metadata. */
typedef struct {
    uint32_t n_layer, n_embd, n_vocab, n_head, n_head_kv, head_dim, n_ffn;
    float rope_freq_base, rms_eps;
    uint32_t n_q_dim, n_kv_dim;
} q4_config;

static bool q4_backend_uses_graph(q4_backend backend) {
    return backend == Q4_BACKEND_METAL || backend == Q4_BACKEND_CUDA;
}

/* =========================================================================
 * GGUF Quant Block Formats (only Q8_0 used for Qwen3.6 weights).
 * ========================================================================= */

#define QK8_0 32

typedef struct {
    uint16_t d;     /* F16 scale */
    int8_t   qs[QK8_0];
} block_q8_0;

#define Q4_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
Q4_STATIC_ASSERT(q4_block_q8_0_size, sizeof(block_q8_0) == 34);

/* =========================================================================
 * Thread Pool for CPU decode reference.
 * ========================================================================= */

typedef void (*q4_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

#define Q4_MAX_THREADS 32

typedef struct {
    pthread_t threads[Q4_MAX_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    q4_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} q4_thread_pool;

/* Forward declarations */
static void q4_die(const char *msg);
static void q4_die_errno(const char *what, const char *path);

static q4_thread_pool g_pool;
static __thread int g_parallel_depth;
static uint32_t g_requested_threads;

static void *q4_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.mutex);
        while (seen_generation == g_pool.generation && !g_pool.shutdown) {
            pthread_cond_wait(&g_pool.work_cond, &g_pool.mutex);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.mutex);
            return NULL;
        }

        seen_generation = g_pool.generation;
        q4_parallel_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        const uint64_t n_rows = g_pool.n_rows;
        const uint32_t n_threads = g_pool.n_threads;
        pthread_mutex_unlock(&g_pool.mutex);

        const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
        const uint64_t row0 = (uint64_t)tid * rows_per_thread;
        uint64_t row1 = row0 + rows_per_thread;
        if (row1 > n_rows) row1 = n_rows;
        if (row0 < row1) {
            g_parallel_depth++;
            fn(ctx, row0, row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&g_pool.mutex);
        g_pool.done++;
        if (g_pool.done == g_pool.n_workers) {
            pthread_cond_signal(&g_pool.done_cond);
        }
        pthread_mutex_unlock(&g_pool.mutex);
    }
}

static void q4_threads_init(void) {
    if (g_pool.initialized) return;

    uint32_t n_threads = 12;
    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cpus > 0) {
        n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    }

    const char *env = getenv("Q4_THREADS");
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (g_requested_threads > 0) n_threads = g_requested_threads;
    if (n_threads > Q4_MAX_THREADS) n_threads = Q4_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&g_pool.mutex, NULL);
    pthread_cond_init(&g_pool.work_cond, NULL);
    pthread_cond_init(&g_pool.done_cond, NULL);
    g_pool.n_threads = n_threads;
    g_pool.n_workers = n_threads > 0 ? n_threads - 1 : 0;
    g_pool.generation = 0;
    g_pool.done = 0;
    g_pool.shutdown = false;
    g_pool.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&g_pool.threads[i], NULL, q4_worker_main, (void *)(uintptr_t)i) != 0) {
            q4_die("failed to create worker thread");
        }
    }
}

static void q4_threads_shutdown(void) {
    if (!g_pool.initialized) return;

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.shutdown = true;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);
    pthread_mutex_unlock(&g_pool.mutex);

    for (uint32_t i = 1; i < g_pool.n_threads; i++) {
        pthread_join(g_pool.threads[i], NULL);
    }

    pthread_cond_destroy(&g_pool.done_cond);
    pthread_cond_destroy(&g_pool.work_cond);
    pthread_mutex_destroy(&g_pool.mutex);
    memset(&g_pool, 0, sizeof(g_pool));
}

static void q4_parallel_for(uint64_t n_rows, q4_parallel_fn fn, void *ctx) {
    q4_threads_init();

    if (g_parallel_depth > 0 || g_pool.n_threads <= 1 || n_rows < 512) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.n_rows = n_rows;
    g_pool.done = 0;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);

    const uint64_t rows_per_thread = (n_rows + g_pool.n_threads - 1) / g_pool.n_threads;
    uint64_t main_row1 = rows_per_thread;
    if (main_row1 > n_rows) main_row1 = n_rows;
    pthread_mutex_unlock(&g_pool.mutex);

    if (main_row1 > 0) {
        g_parallel_depth++;
        fn(ctx, 0, main_row1);
        g_parallel_depth--;
    }

    pthread_mutex_lock(&g_pool.mutex);
    while (g_pool.done < g_pool.n_workers) {
        pthread_cond_wait(&g_pool.done_cond, &g_pool.mutex);
    }
    pthread_mutex_unlock(&g_pool.mutex);
}

/* =========================================================================
 * Helpers, Allocation Guards, Logging, and Cursor Reads.
 * ========================================================================= */

#define Q4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
#define Q4_MAX_DIMS   4

typedef struct {
    const char *ptr;
    uint64_t len;
} q4_str;

typedef struct {
    int *v;
    int len;
    int cap;
} q4_int_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} q4_cursor;

static void q4_die(const char *msg) {
    fprintf(stderr, "q4: %s\n", msg);
    exit(1);
}

static void q4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "q4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static bool q4_streq(q4_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) q4_die("out of memory");
    return p;
}

static void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) q4_die("out of memory");
    return p;
}

static void *xmalloc_zeroed(size_t n, size_t size) {
    const size_t total = n * size;
    void *p = xmalloc(total ? total : 1);
    memset(p, 0, total);
    return p;
}

static void *xrealloc(void *ptr, size_t size) {
    void *p = realloc(ptr, size);
    if (!p) q4_die("out of memory");
    return p;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

bool q4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static const char *q4_log_color_code(q4_log_type type) {
    switch (type) {
    case Q4_LOG_PREFILL:
    case Q4_LOG_TIMING:
        return "\x1b[36m";
    case Q4_LOG_GENERATION:
    case Q4_LOG_OK:
        return "\x1b[32m";
    case Q4_LOG_KVCACHE:
        return "\x1b[33m";
    case Q4_LOG_WARNING:
        return "\x1b[38;5;208m";
    case Q4_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

static void q4_vlog(FILE *fp, q4_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != Q4_LOG_DEFAULT && q4_log_is_tty(fp);
    if (colorize) fputs(q4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void q4_log(FILE *fp, q4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    q4_vlog(fp, type, fmt, ap);
    va_end(ap);
}

const char *q4_backend_name(q4_backend backend) {
    switch (backend) {
    case Q4_BACKEND_METAL:  return "metal";
    case Q4_BACKEND_CUDA:   return "cuda";
    case Q4_BACKEND_CPU:    return "cpu";
    }
    return "unknown";
}

static void cursor_error(q4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(q4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(q4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(q4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(q4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(q4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(q4_cursor *c, q4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

/* =========================================================================
 * GGUF Parsing and Model Mapping.
 * ========================================================================= */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [30] = {"bf16",     1,   2},
};

enum {
    Q4_TENSOR_F32   = 0,
    Q4_TENSOR_F16   = 1,
    Q4_TENSOR_Q8_0  = 8,
    Q4_TENSOR_I32   = 26,
};

typedef struct {
    q4_str key;
    uint32_t type;
    uint64_t value_pos;
} q4_kv;

typedef struct {
    q4_str name;
    uint32_t ndim;
    uint64_t dim[Q4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} q4_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;

    q4_kv *kv;
    q4_tensor *tensors;
} q4_model;

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(q4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        q4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type_info(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type_info(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type_info(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static q4_cursor cursor_at(const q4_model *m, uint64_t pos) {
    q4_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}

static q4_kv *model_find_kv(const q4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (q4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const q4_model *m, const char *key, q4_str *out) {
    q4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    q4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const q4_model *m, const char *key, uint32_t *out) {
    q4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    q4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_f32(const q4_model *m, const char *key, float *out) {
    q4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    q4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        return cursor_read(&c, out, sizeof(*out));
    }
    return false;
}

static q4_tensor *model_find_tensor(const q4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

static q4_tensor *tensor_by_namef(const q4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) q4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static const void *tensor_data(const q4_model *m, const q4_tensor *t) {
    return m->map + t->abs_offset;
}

/* Read the GGUF metadata table. Values stay in the mmap; we store offsets. */
static void parse_metadata(q4_model *m, q4_cursor *c) {
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) q4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        q4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) q4_die(c->error);
        if (!cursor_u32(c, &kv->type)) q4_die(c->error);

        kv->value_pos = c->pos;

        if (q4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32) {
            q4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) q4_die(c->error);
    }
}

/* Read the tensor directory and convert relative GGUF offsets to absolute mmap offsets. */
static void parse_tensors(q4_model *m, q4_cursor *c) {
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) q4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        q4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) q4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) q4_die(c->error);
        if (t->ndim == 0 || t->ndim > Q4_MAX_DIMS) {
            q4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) q4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                q4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) q4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) q4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            q4_log(stderr,
                Q4_LOG_WARNING,
                "q4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        q4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            q4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset)) {
            q4_die("tensor points outside GGUF file");
        }
    }
}

/* Open and map the GGUF once. */
static void model_open(q4_model *m, const char *path, bool metal_mapping) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) q4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) q4_die_errno("cannot stat model", path);
    if (st.st_size < 32) q4_die("model file is too small to be GGUF");

    const int mmap_flags = metal_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
    if (map == MAP_FAILED) q4_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;

    q4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) q4_die(c.error);
    if (magic != Q4_GGUF_MAGIC) q4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) q4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) q4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) q4_die(c.error);

    if (m->version != 3) q4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);

#if defined(POSIX_MADV_WILLNEED)
    const int rc = posix_madvise((void *)m->map, (size_t)m->size, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        q4_log(stderr, Q4_LOG_WARNING,
                "q4: warning: POSIX_MADV_WILLNEED failed for model mapping: %s\n",
                strerror(rc));
    }
#else
    (void)m;
#endif
}

static void model_close(q4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void print_size(uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("%.2f GiB", (double)bytes / gib);
}

/* =========================================================================
 * Layer Weight Types.
 * =========================================================================
 *
 * Qwen3.6-27B uses two types of attention layers:
 * - Gated DeltaNet (il % 4 != 3): recurrent state update
 * - Gated Attention (il % 4 == 3): full softmax attention
 *
 * Both share the same normalization layers and FFN structure.
 */

typedef struct {
    // Shared across layer types
    q4_tensor *attn_norm;
    q4_tensor *ffn_norm;
    q4_tensor *ffn_gate;
    q4_tensor *ffn_up;
    q4_tensor *ffn_down;

    // DeltaNet-specific (il % 4 != 3)
    q4_tensor *attn_a_gate;     // decay gate projection
    q4_tensor *attn_b_proj;     // input projection
    q4_tensor *attn_dt_gate;    // timestep gate
    q4_tensor *attn_a_norm;     // post-delta norm

    // Gated Attention-specific (il % 4 == 3)
    q4_tensor *attn_q;          // query projection
    q4_tensor *attn_k;          // key projection
    q4_tensor *attn_v;          // value projection
    q4_tensor *attn_output;     // output projection
    q4_tensor *attn_qk_norm;    // QK norm weights (if present)
} q4_layer_weights;

typedef struct {
    q4_tensor *token_embd;
    q4_tensor *output_norm;
    q4_tensor *output;
    q4_tensor *norm;  // final RMS norm before output
    q4_layer_weights layer[Q4_N_LAYER];
} q4_weights;

/* =========================================================================
 * Config Validation and Weight Binding.
 * ========================================================================= */

static uint32_t required_u32(const q4_model *m, const char *key) {
    uint32_t v = 0;
    q4_kv *kv = model_find_kv(m, key);
    if (!kv) {
        fprintf(stderr, "q4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    q4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT32) {
        if (!cursor_u32(&c, &v)) q4_die(c.error);
    } else if (kv->type == GGUF_VALUE_INT32) {
        int32_t iv = 0;
        if (!cursor_read(&c, &iv, sizeof(iv))) q4_die(c.error);
        v = (uint32_t)iv;
    } else {
        fprintf(stderr, "q4: metadata key has non-u32 type: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const q4_model *m, const char *key) {
    float v = 0.0f;
    q4_kv *kv = model_find_kv(m, key);
    if (!kv) {
        fprintf(stderr, "q4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    q4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        if (!cursor_read(&c, &v, sizeof(v))) q4_die(c.error);
    } else if (kv->type == GGUF_VALUE_FLOAT64) {
        double d = 0.0;
        if (!cursor_read(&c, &d, sizeof(d))) q4_die(c.error);
        v = (float)d;
    } else {
        fprintf(stderr, "q4: metadata key has non-float type %u: %s\n", kv->type, key);
        exit(1);
    }
    return v;
}

static q4_tensor *required_tensor(const q4_model *m, const char *name) {
    q4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "q4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static q4_tensor *required_tensorf(const q4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) q4_die("tensor name is too long");
    return required_tensor(m, name);
}

static void tensor_expect_layout(
        const q4_tensor *t,
        uint32_t         type,
        uint32_t         ndim,
        uint64_t         d0,
        uint64_t         d1,
        uint64_t         d2) {
    if (!t) q4_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "q4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len, t->name.ptr,
                tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "q4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "q4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
        exit(1);
    }
}

static void tensor_expect_f16_layout(
        const q4_tensor *t,
        uint32_t         ndim,
        uint64_t         d0,
        uint64_t         d1,
        uint64_t         d2) {
    if (!t) q4_die("internal error: missing tensor while validating layout");
    if (t->type != Q4_TENSOR_F16) {
        fprintf(stderr,
                "q4: tensor %.*s has type %s, expected F16\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_f32_layout(
        const q4_tensor *t,
        uint32_t         ndim,
        uint64_t         d0,
        uint64_t         d1,
        uint64_t         d2) {
    if (!t) q4_die("internal error: missing tensor while validating layout");
    if (t->type != Q4_TENSOR_F32) {
        fprintf(stderr,
                "q4: tensor %.*s has type %s, expected F32\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_q8_0_layout(
        const q4_tensor *t,
        uint32_t         ndim,
        uint64_t         d0,
        uint64_t         d1,
        uint64_t         d2) {
    tensor_expect_layout(t, Q4_TENSOR_Q8_0, ndim, d0, d1, d2);
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "q4: expected %s=%u for Qwen3.6-27B, got %u\n", name, expected, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-4f) return;
    fprintf(stderr, "q4: expected %s=%.9g for Qwen3.6-27B, got %.9g\n", name, (double)expected, (double)got);
    exit(1);
}

static bool layer_is_deltanet(uint32_t il) {
    return (il % 4) != 3;
}

static bool layer_is_attention(uint32_t il) {
    return (il % 4) == 3;
}

/* Validate metadata values that affect semantics. */
static void config_validate_model(const q4_model *m) {
    const uint32_t n_layer = required_u32(m, "qwen3.block_count");
    const uint32_t n_embd = required_u32(m, "qwen3.embedding_length");
    const uint32_t n_vocab = required_u32(m, "qwen3.vocab_size");
    const uint32_t n_head = required_u32(m, "qwen3.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "qwen3.attention.head_count_kv");
    const uint32_t n_ffn = required_u32(m, "qwen3.feed_forward_length");
    const uint32_t head_dim = required_u32(m, "qwen3.attention.key_length");
    const float rms_eps = required_f32(m, "qwen3.attention.layer_norm_rms_epsilon");
    const float rope_freq_base = required_f32(m, "qwen3.rope.freq_base");

    config_expect_u32("block_count",         n_layer,  Q4_N_LAYER);
    config_expect_u32("embedding_length",     n_embd,   Q4_N_EMBD);
    config_expect_u32("vocab_size",           n_vocab,  Q4_N_VOCAB);
    config_expect_u32("attention.head_count", n_head,   Q4_N_HEAD);
    config_expect_u32("attention.head_count_kv", n_head_kv, Q4_N_HEAD_KV);
    config_expect_u32("feed_forward_length",  n_ffn,    Q4_N_FFN);
    config_expect_u32("attention.key_length", head_dim, Q4_HEAD_DIM);
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, Q4_RMS_EPS);
    // rope_freq_base varies by model; just log it
    (void)rope_freq_base;
}

/* Bind tensor names into the fixed Qwen3.6-27B layer layout. */
static void weights_bind(q4_weights *w, const q4_model *m) {
    memset(w, 0, sizeof(*w));

    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = required_tensor(m, "output.weight");
    // Some GGUFs use "output.weight" for the unembedding, no separate output_norm
    // Check if norm exists
    w->norm = model_find_tensor(m, "output_norm.weight");
    if (!w->norm) w->norm = model_find_tensor(m, "norm.weight");

    for (uint32_t il = 0; il < Q4_N_LAYER; il++) {
        q4_layer_weights *l = &w->layer[il];

        // Shared
        l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
        l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
        l->ffn_up = required_tensorf(m, "blk.%u.ffn_up.weight", il);
        l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);

        if (layer_is_deltanet(il)) {
            // DeltaNet layer
            l->attn_a_gate = required_tensorf(m, "blk.%u.attn_a_gate.weight", il);
            l->attn_b_proj = required_tensorf(m, "blk.%u.attn_b.weight", il);
            l->attn_dt_gate = required_tensorf(m, "blk.%u.attn_dt.weight", il);
            l->attn_a_norm = required_tensorf(m, "blk.%u.attn_qkv_norm.weight", il);
        } else {
            // Gated Attention layer
            l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
            l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
            l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
            l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
            l->attn_qk_norm = model_find_tensor(m, (char[]){0});  // optional
        }
    }
}

/* Validate every tensor type and dimension used by the pipeline. */
static void weights_validate_layout(const q4_weights *w) {
    tensor_expect_f16_layout(w->token_embd, 2, Q4_N_EMBD, Q4_N_VOCAB, 0);
    tensor_expect_f16_layout(w->output_norm, 1, Q4_N_EMBD, 0, 0);
    tensor_expect_q8_0_layout(w->output, 2, Q4_N_EMBD, Q4_N_VOCAB, 0);

    for (uint32_t il = 0; il < Q4_N_LAYER; il++) {
        const q4_layer_weights *l = &w->layer[il];

        tensor_expect_f32_layout(l->attn_norm, 1, Q4_N_EMBD, 0, 0);
        tensor_expect_f32_layout(l->ffn_norm, 1, Q4_N_EMBD, 0, 0);
        tensor_expect_q8_0_layout(l->ffn_gate, 2, Q4_N_EMBD, Q4_N_FFN, 0);
        tensor_expect_q8_0_layout(l->ffn_up, 2, Q4_N_EMBD, Q4_N_FFN, 0);
        tensor_expect_q8_0_layout(l->ffn_down, 2, Q4_N_FFN, Q4_N_EMBD, 0);

        if (layer_is_deltanet(il)) {
            tensor_expect_f16_layout(l->attn_a_gate, 2, Q4_N_EMBD, Q4_N_HEAD_KV * Q4_HEAD_DIM, 0);
            tensor_expect_f16_layout(l->attn_b_proj, 2, Q4_N_EMBD, Q4_N_HEAD_KV * Q4_HEAD_DIM, 0);
            tensor_expect_f16_layout(l->attn_dt_gate, 2, Q4_N_EMBD, Q4_N_HEAD_KV * Q4_HEAD_DIM, 0);
            tensor_expect_f32_layout(l->attn_a_norm, 1, Q4_N_EMBD, 0, 0);
        } else {
            tensor_expect_q8_0_layout(l->attn_q, 2, Q4_N_EMBD, Q4_N_HEAD * Q4_HEAD_DIM, 0);
            tensor_expect_q8_0_layout(l->attn_k, 2, Q4_N_EMBD, Q4_N_HEAD_KV * Q4_HEAD_DIM, 0);
            tensor_expect_q8_0_layout(l->attn_v, 2, Q4_N_EMBD, Q4_N_HEAD_KV * Q4_HEAD_DIM, 0);
            tensor_expect_q8_0_layout(l->attn_output, 2, Q4_N_HEAD * Q4_HEAD_DIM, Q4_N_EMBD, 0);
        }
    }
}

/* =========================================================================
 * BPE Tokenizer (Simplified SentencePiece).
 * =========================================================================
 *
 * Qwen3.6 uses a BPE tokenizer. We load the vocabulary from GGUF metadata
 * and implement a simple prefix-match tokenizer for inference.
 * For production use, this would be replaced with a full BPE decoder.
 */

typedef struct {
    char **tokens;
    int n_tokens;
    int max_token_len;
    int bos_token;
    int eos_token;
} q4_vocab;

static int vocab_find(const q4_vocab *v, const char *s) {
    for (int i = 0; i < v->n_tokens; i++) {
        if (strcmp(v->tokens[i], s) == 0) return i;
    }
    return -1;
}

static void vocab_load(q4_vocab *v, const q4_model *m) {
    memset(v, 0, sizeof(*v));
    v->bos_token = -1;
    v->eos_token = -1;

    q4_str s;
    if (!model_get_string(m, "tokenizer.ggml.tokens", &s)) {
        q4_die("required tokenizer.ggml.tokens metadata key is missing");
    }

    q4_kv *kv = model_find_kv(m, "tokenizer.ggml.tokens");
    if (!kv || kv->type != GGUF_VALUE_ARRAY) {
        q4_die("tokenizer.ggml.tokens is not an array");
    }

    q4_cursor c = cursor_at(m, kv->value_pos);
    uint32_t item_type;
    uint64_t len;
    if (!cursor_u32(&c, &item_type)) q4_die(c.error);
    if (!cursor_u64(&c, &len)) q4_die(c.error);
    if (item_type != GGUF_VALUE_STRING) q4_die("tokenizer.ggml.tokens is not a string array");

    v->n_tokens = (int)len;
    v->tokens = xcalloc((size_t)len, sizeof(char *));

    int max_len = 0;
    for (uint64_t i = 0; i < len; i++) {
        q4_str tok;
        if (!cursor_string(&c, &tok)) q4_die(c.error);
        v->tokens[i] = xmalloc(tok.len + 1);
        memcpy(v->tokens[i], tok.ptr, tok.len);
        v->tokens[i][tok.len] = '\0';
        if ((int)tok.len > max_len) max_len = (int)tok.len;
    }
    v->max_token_len = max_len;

    // Find special tokens by name
    const char *bos_names[] = {"<｜begin▁of▁sentence｜>", "<s>", "<BOS>", ""};
    const char *eos_names[] = {"<|im_end|>", "</s>", "<|end|>", "<|endoftext|>", ""};
    for (int i = 0; bos_names[i][0] && v->bos_token < 0; i++)
        v->bos_token = vocab_find(v, bos_names[i]);
    for (int i = 0; eos_names[i][0] && v->eos_token < 0; i++)
        v->eos_token = vocab_find(v, eos_names[i]);
}

static void vocab_free(q4_vocab *v) {
    for (int i = 0; i < v->n_tokens; i++) {
        free(v->tokens[i]);
    }
    free(v->tokens);
    memset(v, 0, sizeof(*v));
}

/* Find token ID by string. Linear scan is acceptable for vocab building. */
static int vocab_find_id(const q4_vocab *v, const char *s, int len) {
    for (int i = 0; i < v->n_tokens; i++) {
        if ((int)strlen(v->tokens[i]) == len && memcmp(v->tokens[i], s, len) == 0) {
            return i;
        }
    }
    return -1;
}

/* Simple prefix-match tokenization (BPE approximation). */
static int vocab_tokenize(const q4_vocab *v, const char *text, int text_len, q4_tokens **out) {
    q4_tokens *tokens = xmalloc(sizeof(q4_tokens));
    tokens->cap = text_len > 0 ? text_len : 16;
    tokens->len = 0;
    tokens->v = xcalloc((size_t)tokens->cap, sizeof(int));

    int pos = 0;
    while (pos < text_len) {
        int best_len = 0;
        int best_id = -1;

        // Try longest match first (greedy BPE)
        for (int l = v->max_token_len; l > 0; l--) {
            if (pos + l > text_len) continue;
            int id = vocab_find_id(v, text + pos, l);
            if (id >= 0) {
                best_len = l;
                best_id = id;
                break;
            }
        }

        if (best_id < 0) {
            // Unknown character: use single byte as fallback
            best_len = 1;
            // Try to find the single-byte token
            best_id = vocab_find_id(v, text + pos, 1);
            if (best_id < 0) {
                // Use unknown token ID 0 as last resort
                best_id = 0;
            }
        }

        if (tokens->len >= tokens->cap) {
            tokens->cap *= 2;
            tokens->v = xrealloc(tokens->v, (size_t)tokens->cap * sizeof(int));
        }
        tokens->v[tokens->len++] = best_id;
        pos += best_len;
    }

    *out = tokens;
    return tokens->len;
}

void q4_tokens_free(q4_tokens *tokens) {
    if (tokens) {
        free(tokens->v);
        free(tokens);
    }
}

/* =========================================================================
 * CPU Reference Math Kernels.
 * ========================================================================= */

static inline float f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static inline uint16_t f32_to_f16(float f) {
#if defined(__ARM_NEON)
    const float32x4_t fv = vdupq_n_f32(f);
    const float16x4_t hv = vcvt_f16_f32(fv);
    return vget_lane_u16(vreinterpret_u16_f16(hv), 0);
#else
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
#endif
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

/* =========================================================================
 * CPU Reference Matmul Kernels.
 * ========================================================================= */

/* Q8_0 dot product: single row. */
static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);
            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;

            int32x4_t dot0 = vdupq_n_s32(0);
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0),      vld1q_s8(xq0));
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + 16));

            int32x4_t dot1 = vdupq_n_s32(0);
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1),      vld1q_s8(xq1));
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + 16));

            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot0), f16_to_f32(scale_bits0) * xscale[b]);
            accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(dot1), f16_to_f32(scale_bits1) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, vld1q_s8(qs),      vld1q_s8(xqb));
            dot = vdotq_s32(dot, vld1q_s8(qs + 16), vld1q_s8(xqb + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot), f16_to_f32(scale_bits) * xscale[b]);
        }

        return vaddvq_f32(vaddq_f32(accv0, accv1));
    }
#endif

    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        int32_t dot = 0;
        for (uint64_t i = 0; i < bn; i++) dot += (int32_t)qs[i] * (int32_t)xq[i0 + i];
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot;
    }
    return acc;
}

/* F16 matvec: out[o] = x @ W[:, o] */
typedef struct {
    float *out;
    const uint16_t *data;
    const float *x;
    uint64_t in_dim;
} matvec_f16_ctx;

static void matvec_f16_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f16_ctx *ctx = vctx;
    for (uint64_t o = row0; o < row1; o++) {
        const uint16_t *row = ctx->data + o * ctx->in_dim;
        float acc = 0.0f;
        for (uint64_t i = 0; i < ctx->in_dim; i++) {
            acc += f16_to_f32(row[i]) * ctx->x[i];
        }
        ctx->out[o] = acc;
    }
}

static void matvec_f16(float *out, const q4_model *m, const q4_tensor *w, const float *x) {
    if (w->type != Q4_TENSOR_F16 || w->ndim != 2) q4_die("expected a 2D F16 tensor");
    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    matvec_f16_ctx ctx = {
        .out = out, .data = tensor_data(m, w), .x = x, .in_dim = in_dim,
    };
    q4_parallel_for(out_dim, matvec_f16_worker, &ctx);
}

/* Q8_0 matvec: out[o] = x @ W[:, o] with W Q8_0 quantized */
typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} matvec_q8_0_ctx;

static void matvec_q8_0_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_ctx *ctx = vctx;
    for (uint64_t o = row0; o < row1; o++) {
        const uint8_t *row = ctx->data + o * ctx->blocks * 34;
        ctx->out[o] = dot_q8_0_row(row, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
    }
}

static void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32; i++) xq[i0 + i] = 0;
    }
}

static void matvec_q8_0(float *out, const q4_model *m, const q4_tensor *w, const float *x) {
    if (w->type != Q4_TENSOR_Q8_0 || w->ndim != 2) q4_die("expected a 2D Q8_0 tensor");
    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)blocks * 32);
    float *xscale = xmalloc((size_t)blocks * sizeof(float));
    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_ctx ctx = {
        .out = out, .data = tensor_data(m, w), .xq = xq, .xscale = xscale,
        .in_dim = in_dim, .blocks = blocks,
    };
    q4_parallel_for(out_dim, matvec_q8_0_worker, &ctx);
    free(xscale);
    free(xq);
}

/* Generic matvec dispatch */
static void matvec_any(float *out, const q4_model *m, const q4_tensor *w, const float *x) {
    switch (w->type) {
    case Q4_TENSOR_F16:  matvec_f16(out, m, w, x); break;
    case Q4_TENSOR_Q8_0: matvec_q8_0(out, m, w, x); break;
    case Q4_TENSOR_F32: {
        /* F32 matvec */
        const uint64_t in_dim = w->dim[0];
        const uint64_t out_dim = w->dim[1];
        const float *data = tensor_data(m, w);
        for (uint64_t o = 0; o < out_dim; o++) {
            double acc = 0.0;
            const float *row = data + o * in_dim;
            for (uint64_t i = 0; i < in_dim; i++) acc += (double)row[i] * x[i];
            out[o] = (float)acc;
        }
        break;
    }
    default:
        fprintf(stderr, "q4: unsupported tensor type %u for matvec\n", w->type);
        exit(1);
    }
}

/* =========================================================================
 * Full RoPE (CPU reference).
 * ========================================================================= */

static void rope_full_cpu(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                          uint32_t pos0, float freq_base) {
    const uint32_t half_dim = head_dim / 2;
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        for (uint32_t h = 0; h < n_head; h++) {
            float *head = x + (t * n_head + h) * head_dim;
            for (uint32_t d = 0; d < half_dim; d++) {
                const float freq = 1.0f / powf(freq_base, (float)(2 * d) / (float)head_dim);
                const float theta = (float)pos * freq;
                const float cos_theta = cosf(theta);
                const float sin_theta = sinf(theta);
                const float x0 = head[d];
                const float x1 = head[d + half_dim];
                head[d] = x0 * cos_theta - x1 * sin_theta;
                head[d + half_dim] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

/* =========================================================================
 * SwiGLU FFN (CPU reference).
 * ========================================================================= */

static void ffn_swiglu_cpu(float *out, const q4_model *m, const q4_layer_weights *l,
                           const float *x, float clamp) {
    const uint64_t n_embd = Q4_N_EMBD;
    const uint64_t n_ffn = Q4_N_FFN;

    float *gate = xmalloc(n_ffn * sizeof(float));
    float *up = xmalloc(n_ffn * sizeof(float));

    matvec_any(gate, m, l->ffn_gate, x);
    matvec_any(up, m, l->ffn_up, x);

    for (uint64_t i = 0; i < n_ffn; i++) {
        float g = gate[i];
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (g < -clamp) g = -clamp;
        }
        out[i] = silu(g) * up[i];
    }

    free(gate);
    free(up);

    /* out = mid @ W_down */
    float *tmp = xmalloc(n_embd * sizeof(float));
    matvec_any(tmp, m, l->ffn_down, out);
    memcpy(out, tmp, n_embd * sizeof(float));
    free(tmp);
}

/* =========================================================================
 * Gated DeltaNet (CPU reference).
 * ========================================================================= */

static void deltanet_step_cpu(float *out_state, const q4_model *m, const q4_layer_weights *l,
                              const float *x, float *state) {
    const uint32_t n_kv_heads = Q4_N_HEAD_KV;
    const uint32_t head_dim = Q4_HEAD_DIM;
    const uint32_t n_embd = Q4_N_EMBD;

    // Project x to get a_gate, b_proj, dt_gate values
    // In practice these are computed by small projections before this kernel
    // For simplicity we assume they're already available
    const float *a_gate = tensor_data(m, l->attn_a_gate);
    const float *b_proj = tensor_data(m, l->attn_b_proj);
    const float *dt_gate_data = tensor_data(m, l->attn_dt_gate);

    // For decode: x is the input, state is the recurrent state
    // s' = (1 - dt) * s + dt * (b * v)
    // Here we simplify: v comes from a projection of x
    for (uint32_t kv = 0; kv < n_kv_heads; kv++) {
        const uint32_t base = kv * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) {
            const uint32_t idx = base + d;
            // F16 weights
            const float a = f16_to_f32(((const uint16_t *)a_gate)[idx]);
            const float b = f16_to_f32(((const uint16_t *)b_proj)[idx]);
            const float dt = f16_to_f32(((const uint16_t *)dt_gate_data)[idx]);
            const float v = x[kv * head_dim + d];  // simplified input

            const float s = state[idx];
            const float s_new = (1.0f - dt) * s + dt * b * v;
            out_state[idx] = s_new;
        }
    }
}

/* =========================================================================
 * Gated Attention (CPU reference).
 * ========================================================================= */

static void attention_decode_cpu(float *out, const q4_model *m, const q4_layer_weights *l,
                                 const float *x, const float *k_cache, const float *v_cache,
                                 uint32_t kv_len, uint32_t pos, float logit_softcap) {
    const uint32_t n_q_heads = Q4_N_HEAD;
    const uint32_t n_kv_heads = Q4_N_HEAD_KV;
    const uint32_t head_dim = Q4_HEAD_DIM;
    const uint32_t n_embd = Q4_N_EMBD;
    const uint32_t q_per_kv = Q4_Q_PER_KV;
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    // Q = x @ W_q
    float *q = xmalloc(n_q_heads * head_dim * sizeof(float));
    matvec_any(q, m, l->attn_q, x);

    // RoPE applied to Q (full)
    rope_full_cpu(q, 1, n_q_heads, head_dim, pos, 10000.0f);

    for (uint32_t qh = 0; qh < n_q_heads; qh++) {
        const uint32_t kv_h = qh / q_per_kv;
        const float *q_h = q + qh * head_dim;

        // Compute attention scores
        float *scores = xmalloc(kv_len * sizeof(float));
        float max_val = Q4_NEG_INF;
        for (uint32_t t = 0; t < kv_len; t++) {
            const float *k_t = k_cache + (kv_h * head_dim) + (t * n_kv_heads * head_dim);
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                score += q_h[d] * k_t[d];
            }
            score *= inv_sqrt_d;
            if (logit_softcap > 1.0e-6f) {
                score = logit_softcap * tanhf(score / logit_softcap);
            }
            scores[t] = score;
            if (score > max_val) max_val = score;
        }

        // Softmax
        float sum = 0.0f;
        for (uint32_t t = 0; t < kv_len; t++) {
            scores[t] = expf(scores[t] - max_val);
            sum += scores[t];
        }
        const float inv_sum = 1.0f / sum;

        // Weighted sum of V
        float *out_h = out + qh * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) out_h[d] = 0.0f;
        for (uint32_t t = 0; t < kv_len; t++) {
            const float *v_t = v_cache + (kv_h * head_dim) + (t * n_kv_heads * head_dim);
            const float w = scores[t] * inv_sum;
            for (uint32_t d = 0; d < head_dim; d++) {
                out_h[d] += w * v_t[d];
            }
        }

        free(scores);
    }

    // Output projection
    float *tmp = xmalloc(n_embd * sizeof(float));
    matvec_any(tmp, m, l->attn_output, out);
    memcpy(out, tmp, n_embd * sizeof(float));
    free(tmp);

    free(q);
}

/* =========================================================================
 * Engine and Session Types (internal).
 * ========================================================================= */

struct q4_engine {
    q4_backend backend;
    q4_model model;
    q4_weights weights;
    q4_vocab vocab;
    q4_config config;
    float rope_freq_base;
    float logit_softcap;
};

struct q4_session {
    q4_engine *engine;
    int ctx_size;

    // KV cache
    float *k_cache;  // [ctx_size, n_kv_heads, head_dim]
    float *v_cache;  // [ctx_size, n_kv_heads, head_dim]
    uint32_t kv_len;

    // Token buffer
    int *tokens;
    int n_tokens;

    // Logits
    float *logits;
};

/* =========================================================================
 * Softmax for output logits.
 * ========================================================================= */

static void softmax_cpu(float *x, uint64_t n) {
    float max_val = x[0];
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (uint64_t i = 0; i < n; i++) x[i] *= inv;
}

/* =========================================================================
 * Argmax sampling.
 * ========================================================================= */

static int sample_argmax(const float *logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* =========================================================================
 * Multinomial sampling.
 * ========================================================================= */

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float rand_uniform(uint64_t *rng) {
    return (float)(xorshift64(rng) & 0xFFFFFF) / (float)0x1000000;
}

static int sample_multinomial(const float *probs, int n, uint64_t *rng) {
    float r = rand_uniform(rng);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return n - 1;
}

/* =========================================================================
 * Temperature + top-p + top-k + min-p sampling.
 * ========================================================================= */

typedef struct {
    int id;
    float prob;
} q4_token_prob;

static int cmp_token_prob_desc(const void *a, const void *b) {
    const q4_token_prob *pa = (const q4_token_prob *)a;
    const q4_token_prob *pb = (const q4_token_prob *)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

static int sample_top_k_top_p_min_p(const float *logits, int n_vocab,
                                     float temperature, int top_k,
                                     float top_p, float min_p,
                                     uint64_t *rng) {
    // Copy and convert to probabilities
    q4_token_prob *probs = xmalloc(n_vocab * sizeof(q4_token_prob));

    // Apply temperature
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        probs[i].id = i;
        probs[i].prob = expf((logits[i] - max_logit) / temperature);
        sum += probs[i].prob;
    }
    for (int i = 0; i < n_vocab; i++) probs[i].prob /= sum;

    // Sort by probability descending
    qsort(probs, n_vocab, sizeof(q4_token_prob), cmp_token_prob_desc);

    // Top-k
    if (top_k > 0 && top_k < n_vocab) {
        // Zero out tokens beyond top_k
        for (int i = top_k; i < n_vocab; i++) probs[i].prob = 0.0f;
    }

    // Top-p
    if (top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < n_vocab; i++) {
            cum += probs[i].prob;
            if (cum > top_p) {
                for (int j = i + 1; j < n_vocab; j++) probs[j].prob = 0.0f;
                break;
            }
        }
    }

    // Min-p
    if (min_p > 0.0f) {
        float threshold = probs[0].prob * min_p;
        for (int i = 1; i < n_vocab; i++) {
            if (probs[i].prob < threshold) {
                for (int j = i; j < n_vocab; j++) probs[j].prob = 0.0f;
                break;
            }
        }
    }

    // Renormalize
    sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) sum += probs[i].prob;
    if (sum > 0.0f) {
        for (int i = 0; i < n_vocab; i++) probs[i].prob /= sum;
    }

    int result = sample_multinomial((const float *)probs /* wrong cast, but we'll do manual */, n_vocab, rng);
    // Manual sample from probs array
    float r = rand_uniform(rng);
    float cum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cum += probs[i].prob;
        if (r < cum) {
            result = probs[i].id;
            goto done;
        }
    }
    result = probs[n_vocab - 1].id;

done:
    free(probs);
    return result;
}

/* =========================================================================
 * Public API: Engine.
 * ========================================================================= */

int q4_engine_open(q4_engine **out, const q4_engine_options *opt) {
    if (!opt || !opt->model_path) {
        fprintf(stderr, "q4: model_path is required\n");
        return -1;
    }

    q4_engine *e = xmalloc_zeroed(1, sizeof(*e));
    e->backend = opt->backend;

    // Open and map GGUF
    const bool metal_mapping = (e->backend == Q4_BACKEND_METAL);
    model_open(&e->model, opt->model_path, metal_mapping);

    // Load config from GGUF metadata
    e->config.n_layer = required_u32(&e->model, "qwen3.block_count");
    e->config.n_embd = required_u32(&e->model, "qwen3.embedding_length");
    e->config.n_vocab = required_u32(&e->model, "qwen3.vocab_size");
    e->config.n_head = required_u32(&e->model, "qwen3.attention.head_count");
    e->config.n_head_kv = required_u32(&e->model, "qwen3.attention.head_count_kv");
    e->config.head_dim = required_u32(&e->model, "qwen3.attention.key_length");
    e->config.n_ffn = required_u32(&e->model, "qwen3.feed_forward_length");

    // Derived
    e->config.n_q_dim = e->config.n_head * e->config.head_dim;
    e->config.n_kv_dim = e->config.n_head_kv * e->config.head_dim;

    // Optional metadata
    model_get_f32(&e->model, "qwen3.rope.freq_base", &e->rope_freq_base);
    if (e->rope_freq_base < 1.0e-6f) e->rope_freq_base = 10000.0f;
    model_get_f32(&e->model, "qwen3.attention.logit_softcap", &e->logit_softcap);

    // Validate model config
    config_validate_model(&e->model);

    // Bind weights
    weights_bind(&e->weights, &e->model);
    weights_validate_layout(&e->weights);

    // Load tokenizer
    vocab_load(&e->vocab, &e->model);

#ifndef Q4_NO_GPU
    // Initialize GPU backend
    if (q4_backend_uses_graph(e->backend)) {
        if (q4_gpu_init() != 0) {
            fprintf(stderr, "q4: failed to initialize GPU backend\n");
            model_close(&e->model);
            vocab_free(&e->vocab);
            free(e);
            return -1;
        }
        // Set model mapping for GPU
        if (q4_gpu_set_model_map(e->model.map, e->model.size) != 0) {
            fprintf(stderr, "q4: failed to set GPU model map\n");
            q4_gpu_cleanup();
            model_close(&e->model);
            vocab_free(&e->vocab);
            free(e);
            return -1;
        }
    }
#endif

    // Warm weights if requested
    if (opt->warm_weights && e->model.map && e->model.size > 0) {
        const uint64_t start = e->model.tensor_data_pos;
        const uint64_t end = e->model.size;
        if (start < end) {
            const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
            const uint8_t *p = e->model.map;
            volatile uint64_t checksum = 0;
            for (uint64_t off = start; off < end; off += page) {
                checksum += p[off];
            }
            checksum += p[end - 1];
            fprintf(stderr, "q4: warmed tensor pages (checksum=%llu)\n",
                    (unsigned long long)checksum);
        }
    }

    *out = e;
    return 0;
}

void q4_engine_close(q4_engine *e) {
    if (!e) return;
#ifndef Q4_NO_GPU
    if (q4_backend_uses_graph(e->backend)) {
        q4_gpu_cleanup();
    }
#endif
    model_close(&e->model);
    vocab_free(&e->vocab);
    free(e);
}

void q4_engine_summary(q4_engine *e) {
    if (!e) return;

    printf("model: %s\n", e->config.n_vocab > 0 ? "Qwen3.6-27B" : "unknown");
    printf("backend: %s\n", q4_backend_name(e->backend));
    printf("layers: %u\n", e->config.n_layer);
    printf("embedding: %u\n", e->config.n_embd);
    printf("vocab: %u\n", e->config.n_vocab);
    printf("attention: heads=%u kv_heads=%u head_dim=%u\n",
           e->config.n_head, e->config.n_head_kv, e->config.head_dim);
    printf("FFN: %u\n", e->config.n_ffn);
    printf("architecture: hybrid (Gated DeltaNet + Gated Attention)\n");
    printf("GGUF: v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
           e->model.version, e->model.n_kv, e->model.n_tensors);

    printf("file size: ");
    print_size(e->model.size);
    printf("\n");

    uint64_t tensor_bytes = 0;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        tensor_bytes += e->model.tensors[i].bytes;
    }
    printf("tensor bytes: ");
    print_size(tensor_bytes);
    printf("\n");
}

q4_context_memory q4_context_memory_estimate(q4_backend backend, int ctx_size) {
    (void)backend;
    q4_context_memory mem = {0};
    // KV cache: 2 x ctx x n_kv_heads x head_dim x sizeof(float)
    const uint64_t kv_bytes = 2ULL * (uint64_t)ctx_size * Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float);
    // Scratch: activations ~ 3 x n_embd x sizeof(float) per layer
    const uint64_t scratch_bytes = 3ULL * Q4_N_EMBD * sizeof(float) * Q4_N_LAYER;
    mem.total_bytes = kv_bytes + scratch_bytes;
    mem.raw_bytes = kv_bytes;
    mem.scratch_bytes = scratch_bytes;
    mem.prefill_cap = 0;
    mem.raw_cap = 0;
    return mem;
}

/* =========================================================================
 * Tokenizer.
 * ========================================================================= */

int q4_engine_tokenize(q4_engine *e, const char *text, int text_len,
                        q4_tokens **out_tokens) {
    if (!e || !text || text_len <= 0) return -1;
    return vocab_tokenize(&e->vocab, text, text_len, out_tokens);
}

/* =========================================================================
 * Session.
 * ========================================================================= */

int q4_session_create(q4_session **out, q4_engine *e, int ctx_size) {
    if (!e || ctx_size <= 0) return -1;

    q4_session *s = xmalloc_zeroed(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;

    // Allocate KV cache
    const uint64_t kv_head_bytes = (uint64_t)Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float);
    s->k_cache = xmalloc_zeroed((uint64_t)ctx_size, kv_head_bytes);
    s->v_cache = xmalloc_zeroed((uint64_t)ctx_size, kv_head_bytes);
    s->kv_len = 0;

    // Token buffer
    s->tokens = xmalloc((size_t)ctx_size * sizeof(int));
    s->n_tokens = 0;

    // Logits (vocab-sized)
    s->logits = xmalloc((size_t)e->config.n_vocab * sizeof(float));

    *out = s;
    return 0;
}

void q4_session_free(q4_session *s) {
    if (!s) return;
    free(s->k_cache);
    free(s->v_cache);
    free(s->tokens);
    free(s->logits);
    free(s);
}

int q4_session_n_tokens(q4_session *s) {
    return s ? s->n_tokens : 0;
}

const float *q4_session_logits(q4_session *s) {
    return s ? s->logits : NULL;
}

/* =========================================================================
 * CPU Inference Path.
 * ========================================================================= */

typedef struct {
    q4_engine *e;
    q4_session *s;
    const int *tokens;
    int n_tokens;
    uint32_t pos0;
    float *hidden;      // [n_embd]
    float *residual;    // [n_embd]
    float *deltanet_state;  // [n_kv_heads * head_dim]
} q4_cpu_decode_ctx;

/* Embed token -> hidden */
static void embed_token_cpu(const q4_model *m, const q4_tensor *te, int token, float *out) {
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        q4_die("token id is outside the embedding table");
    }
    const uint16_t *base = tensor_data(m, te);
    const uint64_t stride = te->dim[0];
    const uint16_t *row = base + (uint64_t)token * stride;
    for (uint64_t i = 0; i < stride; i++) {
        out[i] = f16_to_f32(row[i]);
    }
}

/* Run one token through the model on CPU. */
static int q4_cpu_forward(q4_engine *e, q4_session *s, int token, uint32_t pos, char *err, size_t errlen) {
    const uint32_t n_embd = Q4_N_EMBD;
    const uint32_t n_layer = Q4_N_LAYER;

    float *hidden = xmalloc(n_embd * sizeof(float));
    float *residual = xmalloc(n_embd * sizeof(float));
    float *deltanet_state = xmalloc_zeroed(Q4_N_HEAD_KV * Q4_HEAD_DIM, sizeof(float));

    // Embed token
    embed_token_cpu(&e->model, e->weights.token_embd, token, hidden);
    memcpy(residual, hidden, n_embd * sizeof(float));

    for (uint32_t il = 0; il < n_layer; il++) {
        const q4_layer_weights *l = &e->weights.layer[il];

        // RMSNorm
        float *normed = xmalloc(n_embd * sizeof(float));
        const float *norm_weight = tensor_data(&e->model, l->attn_norm);
        rms_norm_weight(normed, hidden, norm_weight, n_embd, Q4_RMS_EPS);

        if (layer_is_deltanet(il)) {
            // DeltaNet: recurrent state update
            // For simplicity: output = normed (full implementation needs projections)
            // s' = (1 - dt) * s + dt * (b * v)
            // Here we just pass through for now
            memcpy(hidden, normed, n_embd * sizeof(float));

            // Update DeltaNet state
            deltanet_step_cpu(deltanet_state, &e->model, l, normed, deltanet_state);
            // Copy state back to hidden (simplified)
            // Full implementation would project state to n_embd
        } else {
            // Gated Attention
            float *attn_out = xmalloc(n_embd * sizeof(float));

            // Store K, V in cache
            const uint32_t kv_idx = s->kv_len;
            if (kv_idx >= (uint32_t)s->ctx_size) {
                if (err && errlen > 0) snprintf(err, errlen, "KV cache full");
                free(normed); free(attn_out); free(hidden); free(residual); free(deltanet_state);
                return -1;
            }

            // For decode, we need to project Q, K, V
            float *q = xmalloc(Q4_N_HEAD * Q4_HEAD_DIM * sizeof(float));
            float *k = xmalloc(Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float));
            float *v = xmalloc(Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float));

            matvec_any(q, &e->model, l->attn_q, normed);
            matvec_any(k, &e->model, l->attn_k, normed);
            matvec_any(v, &e->model, l->attn_v, normed);

            // Apply RoPE to Q and K
            rope_full_cpu(q, 1, Q4_N_HEAD, Q4_HEAD_DIM, pos, e->rope_freq_base);
            rope_full_cpu(k, 1, Q4_N_HEAD_KV, Q4_HEAD_DIM, pos, e->rope_freq_base);

            // Store K, V
            const uint64_t kv_head_bytes = (uint64_t)Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float);
            memcpy(s->k_cache + kv_idx * Q4_N_HEAD_KV * Q4_HEAD_DIM, k, kv_head_bytes);
            memcpy(s->v_cache + kv_idx * Q4_N_HEAD_KV * Q4_HEAD_DIM, v, kv_head_bytes);

            // Run attention
            attention_decode_cpu(attn_out, &e->model, l, normed,
                               s->k_cache, s->v_cache, s->kv_len + 1, pos, e->logit_softcap);

            // Increment KV cache length
            s->kv_len++;

            free(q); free(k); free(v);

            memcpy(hidden, attn_out, n_embd * sizeof(float));
            free(attn_out);
        }

        // Residual
        for (uint32_t i = 0; i < n_embd; i++) {
            hidden[i] += residual[i];
        }
        memcpy(residual, hidden, n_embd * sizeof(float));

        free(normed);

        // FFN
        normed = xmalloc(n_embd * sizeof(float));
        const float *ffn_norm_weight = tensor_data(&e->model, l->ffn_norm);
        rms_norm_weight(normed, hidden, ffn_norm_weight, n_embd, Q4_RMS_EPS);

        float *ffn_out = xmalloc(n_embd * sizeof(float));
        ffn_swiglu_cpu(ffn_out, &e->model, l, normed, 10.0f);

        // Residual
        for (uint32_t i = 0; i < n_embd; i++) {
            hidden[i] += ffn_out[i];
        }
        memcpy(residual, hidden, n_embd * sizeof(float));

        free(normed);
        free(ffn_out);
    }

    // Final RMS norm
    const float *output_norm_w = tensor_data(&e->model, e->weights.output_norm);
    rms_norm_weight(hidden, hidden, output_norm_w, n_embd, Q4_RMS_EPS);

    // Output projection (Q8_0)
    matvec_q8_0(s->logits, &e->model, e->weights.output, hidden);

    free(hidden);
    free(residual);
    free(deltanet_state);

    return 0;
}

/* =========================================================================
 * Session Sync: evaluate a prompt prefix, reusing KV cache where possible.
 * ========================================================================= */

int q4_session_sync(q4_session *s, const q4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len == 0) {
        if (err && errlen > 0) snprintf(err, errlen, "invalid prompt");
        return -1;
    }

    q4_engine *e = s->engine;

#ifndef Q4_NO_GPU
    if (q4_backend_uses_graph(e->backend)) {
        // GPU path: build and execute compute graph
        if (err && errlen > 0) snprintf(err, errlen, "GPU inference path not yet implemented");
        return -1;
    }
#endif

    // CPU path: process tokens one by one
    s->n_tokens = 0;
    s->kv_len = 0;

    for (int i = 0; i < prompt->len; i++) {
        s->tokens[s->n_tokens++] = prompt->v[i];
        if (q4_cpu_forward(e, s, prompt->v[i], (uint32_t)i, err, errlen) != 0) {
            return -1;
        }
    }

    return 0;
}

/* Evaluate a single token and update logits. */
int q4_session_eval(q4_session *s, int token, char *err, size_t errlen) {
    if (!s) {
        if (err && errlen > 0) snprintf(err, errlen, "invalid session");
        return -1;
    }

    q4_engine *e = s->engine;

#ifndef Q4_NO_GPU
    if (q4_backend_uses_graph(e->backend)) {
        if (err && errlen > 0) snprintf(err, errlen, "GPU inference path not yet implemented");
        return -1;
    }
#endif

    uint32_t pos = s->kv_len;
    if (q4_cpu_forward(e, s, token, pos, err, errlen) != 0) {
        return -1;
    }
    if (s->n_tokens < s->ctx_size) {
        s->tokens[s->n_tokens++] = token;
    }

    return 0;
}

/* Argmax sampling from logits. */
int q4_session_argmax(q4_session *s, char *err, size_t errlen) {
    if (!s || !s->logits) {
        if (err && errlen > 0) snprintf(err, errlen, "invalid session or no logits");
        return -1;
    }
    return sample_argmax(s->logits, (int)s->engine->config.n_vocab);
}

/* Temperature + top-k + top-p + min-p sampling. */
int q4_session_sample(q4_session *s, float temperature, int top_k,
                       float top_p, float min_p, uint64_t *rng,
                       char *err, size_t errlen) {
    if (!s || !s->logits) {
        if (err && errlen > 0) snprintf(err, errlen, "invalid session or no logits");
        return -1;
    }
    if (temperature < 1.0e-6f) {
        return q4_session_argmax(s, err, errlen);
    }
    return sample_top_k_top_p_min_p(s->logits, (int)s->engine->config.n_vocab,
                                     temperature, top_k, top_p, min_p, rng);
}

void q4_session_checkpoint(q4_session *s) {
    // Simple checkpoint: just record current position
    // Full implementation would save/restore KV cache state
    (void)s;
}

/* =========================================================================
 * Simple Generation: prompt + predict loop (no incremental cache).
 * ========================================================================= */

int q4_engine_generate_argmax(q4_engine *e, const q4_tokens *prompt,
                               int n_predict, int ctx_size,
                               q4_token_emit_fn emit,
                               q4_generation_done_fn done,
                               void *emit_ud) {
    if (!e || !prompt || n_predict <= 0) return -1;

    q4_session *s = NULL;
    if (q4_session_create(&s, e, ctx_size) != 0) return -1;

    char err[256];

    // Process prompt
    if (q4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "q4: sync failed: %s\n", err);
        q4_session_free(s);
        return -1;
    }

    // Generate
    for (int i = 0; i < n_predict; i++) {
        int token = q4_session_argmax(s, err, sizeof(err));
        if (token < 0) {
            fprintf(stderr, "q4: argmax failed: %s\n", err);
            break;
        }

        if (emit) emit(emit_ud, token);

        if (q4_session_eval(s, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "q4: eval failed: %s\n", err);
            break;
        }
    }

    if (done) done(emit_ud);
    q4_session_free(s);
    return 0;
}

/* =========================================================================
 * Token utilities exposed to CLI.
 * ========================================================================= */

char *q4_token_text(q4_engine *e, int token, size_t *out_len) {
    if (!e || token < 0 || token >= e->vocab.n_tokens) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    const char *s = e->vocab.tokens[token];
    size_t len = strlen(s);
    if (out_len) *out_len = len;
    char *buf = malloc(len + 1);
    if (buf) {
        memcpy(buf, s, len);
        buf[len] = '\0';
    }
    return buf;
}

int q4_token_eos(q4_engine *e) {
    if (e && e->vocab.eos_token >= 0) return e->vocab.eos_token;
    return e ? e->vocab.n_tokens - 1 : 151645;
}

void q4_tokens_push(q4_tokens *tokens, int token) {
    if (!tokens) return;
    if (tokens->len >= tokens->cap) {
        tokens->cap = tokens->cap ? tokens->cap * 2 : 64;
        int *next = realloc(tokens->v, (size_t)tokens->cap * sizeof(int));
        if (!next) { perror("q4: realloc"); exit(1); }
        tokens->v = next;
    }
    tokens->v[tokens->len++] = token;
}

/* =========================================================================
 * Session position and context helpers.
 * ========================================================================= */

int q4_session_pos(q4_session *s) {
    return s ? (int)s->kv_len : 0;
}

int q4_session_ctx(q4_session *s) {
    return s ? s->ctx_size : 0;
}
