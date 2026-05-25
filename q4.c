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
    Q4_HEAD_DIM     = 256,    /* K/V head dimension */
    Q4_Q_HEAD_DIM   = 512,    /* Q head dimension (24 * 512 = 12288) */
    Q4_N_FFN        = 17408,
    Q4_Q_PER_KV     = 6,  /* 24 / 4 */
    Q4_BLOCK_SIZE   = 4,  /* 3 DeltaNet + 1 Attention */
    Q4_N_BLOCKS     = 16,

    /* Gated DeltaNet dimensions */
    Q4_N_V_HEADS    = 48,   /* ssm_time_step_rank = num value heads */
    Q4_N_K_GROUPS   = 16,   /* ssm_group_count = num key heads */
    Q4_HEAD_K_DIM   = 128,  /* ssm_state_size = key dim per group */
    Q4_HEAD_V_DIM   = 128,  /* head_v_dim = ssm_inner_size / n_v_heads */
    Q4_QKV_DIM      = 10240,/* n_embd + n_k_dim + n_v_dim = 5120+2048+6144 (no: qkv_dim) */
    Q4_CONV_KERNEL  = 4,    /* ssm_conv_kernel */
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

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    uint16_t d;
} block_q6_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];   /* 32 bytes: 1 high bit per element (bit 4), 256/8=32 */
    uint8_t  qs[QK_K / 2];   /* 128 bytes: low 4 bits per element */
} block_q5_K;
/* Total: 2+2+12+32+128 = 176 bytes */

#define Q4_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
Q4_STATIC_ASSERT(q4_block_q8_0_size, sizeof(block_q8_0) == 34);
Q4_STATIC_ASSERT(q4_block_q4_k_size, sizeof(block_q4_K) == 144);
Q4_STATIC_ASSERT(q4_block_q5_k_size, sizeof(block_q5_K) == 176);
Q4_STATIC_ASSERT(q4_block_q6_k_size, sizeof(block_q6_K) == 210);

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
    Q4_TENSOR_Q4_K  = 12,
    Q4_TENSOR_Q5_K  = 13,
    Q4_TENSOR_Q6_K  = 14,
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
    q4_tensor *attn_norm;        /* pre-layer norm */
    q4_tensor *post_attn_norm;   /* post-layer norm, also FFN pre-norm */
    q4_tensor *ffn_gate;
    q4_tensor *ffn_up;
    q4_tensor *ffn_down;

    // DeltaNet (il % 4 != 3)
    q4_tensor *attn_qkv;      /* [n_embd, qkv_dim=10240] Q6_K: q+k+v combined */
    q4_tensor *ssm_conv1d;    /* [conv_kernel=4, qkv_dim=10240] F32 */
    q4_tensor *ssm_alpha;     /* [n_embd, n_v_heads=48] F32 */
    q4_tensor *ssm_beta;      /* [n_embd, n_v_heads=48] F32 */
    q4_tensor *ssm_dt_bias;   /* [n_v_heads=48] F32 */
    q4_tensor *ssm_a;         /* [n_v_heads=48] F32, gating factor */
    q4_tensor *ssm_norm;      /* [head_v_dim=128] F32 */
    q4_tensor *ssm_out;       /* [n_v_heads*head_v_dim=6144, n_embd] Q5_K */
    q4_tensor *attn_gate;     /* [n_embd, n_v_heads*head_v_dim=6144] Q4_K, z gate */

    // Gated Attention (il % 4 == 3)
    q4_tensor *attn_q;        /* [n_embd, n_head*q_head_dim=12288] Q4_K */
    q4_tensor *attn_k;        /* [n_embd, n_head_kv*head_dim=1024] Q4_K */
    q4_tensor *attn_v;        /* [n_embd, n_head_kv*head_dim=1024] Q6_K */
    q4_tensor *attn_output;   /* [n_head*q_head_dim=6144, n_embd] Q4_K */
    q4_tensor *attn_q_norm;   /* [256] F32 - QK norm over first 256 dims */
    q4_tensor *attn_k_norm;   /* [256] F32 */
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

static void tensor_expect_q4_k_layout(
        const q4_tensor *t,
        uint32_t         ndim,
        uint64_t         d0,
        uint64_t         d1,
        uint64_t         d2) {
    tensor_expect_layout(t, Q4_TENSOR_Q4_K, ndim, d0, d1, d2);
}

static void tensor_expect_q5_k_layout(
        const q4_tensor *t,
        uint32_t         ndim,
        uint64_t         d0,
        uint64_t         d1,
        uint64_t         d2) {
    tensor_expect_layout(t, Q4_TENSOR_Q5_K, ndim, d0, d1, d2);
}

static void tensor_expect_q6_k_layout(
        const q4_tensor *t,
        uint32_t         ndim,
        uint64_t         d0,
        uint64_t         d1,
        uint64_t         d2) {
    tensor_expect_layout(t, Q4_TENSOR_Q6_K, ndim, d0, d1, d2);
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
    const uint32_t n_layer = required_u32(m, "qwen35.block_count");
    const uint32_t n_embd = required_u32(m, "qwen35.embedding_length");
    const uint32_t n_head = required_u32(m, "qwen35.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "qwen35.attention.head_count_kv");
    const uint32_t n_ffn = required_u32(m, "qwen35.feed_forward_length");
    const uint32_t head_dim = required_u32(m, "qwen35.attention.key_length");
    const float rms_eps = required_f32(m, "qwen35.attention.layer_norm_rms_epsilon");
    const float rope_freq_base = required_f32(m, "qwen35.rope.freq_base");

    config_expect_u32("block_count",         n_layer,  Q4_N_LAYER);
    config_expect_u32("embedding_length",     n_embd,   Q4_N_EMBD);
    // vocab_size not in metadata; inferred from token_embd tensor
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
        l->post_attn_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
        l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
        l->ffn_up = required_tensorf(m, "blk.%u.ffn_up.weight", il);
        l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);

        if (layer_is_deltanet(il)) {
            // DeltaNet layer
            l->attn_qkv = required_tensorf(m, "blk.%u.attn_qkv.weight", il);
            l->ssm_conv1d = required_tensorf(m, "blk.%u.ssm_conv1d.weight", il);
            l->ssm_alpha = required_tensorf(m, "blk.%u.ssm_alpha.weight", il);
            l->ssm_beta = required_tensorf(m, "blk.%u.ssm_beta.weight", il);
            l->ssm_dt_bias = required_tensorf(m, "blk.%u.ssm_dt.bias", il);
            l->ssm_a = required_tensorf(m, "blk.%u.ssm_a", il);
            l->ssm_norm = required_tensorf(m, "blk.%u.ssm_norm.weight", il);
            l->ssm_out = required_tensorf(m, "blk.%u.ssm_out.weight", il);
            l->attn_gate = required_tensorf(m, "blk.%u.attn_gate.weight", il);
            l->post_attn_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
        } else {
            // Gated Attention layer
            l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
            l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
            l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
            l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
            l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
            l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
            l->post_attn_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
        }
    }
}

/* Validate every tensor type and dimension used by the pipeline. */
static void weights_validate_layout(const q4_weights *w) {
    tensor_expect_q4_k_layout(w->token_embd, 2, Q4_N_EMBD, Q4_N_VOCAB, 0);
    tensor_expect_f32_layout(w->output_norm, 1, Q4_N_EMBD, 0, 0);
    tensor_expect_q6_k_layout(w->output, 2, Q4_N_EMBD, Q4_N_VOCAB, 0);

    for (uint32_t il = 0; il < Q4_N_LAYER; il++) {
        const q4_layer_weights *l = &w->layer[il];

        tensor_expect_f32_layout(l->attn_norm, 1, Q4_N_EMBD, 0, 0);
        tensor_expect_f32_layout(l->post_attn_norm, 1, Q4_N_EMBD, 0, 0);
        tensor_expect_q4_k_layout(l->ffn_gate, 2, Q4_N_EMBD, Q4_N_FFN, 0);
        tensor_expect_q4_k_layout(l->ffn_up, 2, Q4_N_EMBD, Q4_N_FFN, 0);
        /* ffn_down can be Q4_K or Q6_K depending on layer */
        if (l->ffn_down->type != Q4_TENSOR_Q4_K && l->ffn_down->type != Q4_TENSOR_Q6_K) {
            fprintf(stderr, "q4: ffn_down type mismatch\n");
            exit(1);
        }
        if (l->ffn_down->dim[0] != Q4_N_FFN || l->ffn_down->dim[1] != Q4_N_EMBD) {
            fprintf(stderr, "q4: ffn_down dim mismatch\n");
            exit(1);
        }

        if (layer_is_deltanet(il)) {
            /* attn_qkv: [n_embd, qkv_dim] Q4_K or Q6_K */
            if (l->attn_qkv->type != Q4_TENSOR_Q4_K && l->attn_qkv->type != Q4_TENSOR_Q6_K) { fprintf(stderr, "q4: attn_qkv type\n"); exit(1); }
            if (l->attn_qkv->dim[0] != Q4_N_EMBD || l->attn_qkv->dim[1] != 10240) { fprintf(stderr, "q4: attn_qkv dim\n"); exit(1); }
            /* ssm_conv1d: [conv_kernel, qkv_dim] F32 */
            tensor_expect_f32_layout(l->ssm_conv1d, 2, 4, 10240, 0);
            /* ssm_alpha: [n_embd, n_v_heads] F32 */
            tensor_expect_f32_layout(l->ssm_alpha, 2, Q4_N_EMBD, 48, 0);
            /* ssm_beta: [n_embd, n_v_heads] F32 */
            tensor_expect_f32_layout(l->ssm_beta, 2, Q4_N_EMBD, 48, 0);
            /* ssm_dt_bias: [n_v_heads] F32 */
            tensor_expect_f32_layout(l->ssm_dt_bias, 1, 48, 0, 0);
            /* ssm_a: [n_v_heads] F32 */
            tensor_expect_f32_layout(l->ssm_a, 1, 48, 0, 0);
            /* ssm_norm: [head_v_dim] F32 */
            tensor_expect_f32_layout(l->ssm_norm, 1, 128, 0, 0);
            /* ssm_out: [n_v_heads*head_v_dim, n_embd] Q5_K */
            tensor_expect_q5_k_layout(l->ssm_out, 2, 6144, Q4_N_EMBD, 0);
            /* attn_gate: [n_embd, n_v_heads*head_v_dim] Q4_K */
            tensor_expect_q4_k_layout(l->attn_gate, 2, Q4_N_EMBD, 6144, 0);
            /* post_attn_norm: [n_embd] F32 */
            tensor_expect_f32_layout(l->post_attn_norm, 1, Q4_N_EMBD, 0, 0);
        } else {
            /* attn_q: [n_embd, n_head*q_head_dim] Q4_K */
            tensor_expect_q4_k_layout(l->attn_q, 2, Q4_N_EMBD, 12288, 0);
            /* attn_k: [n_embd, n_head_kv*head_dim] Q4_K */
            tensor_expect_q4_k_layout(l->attn_k, 2, Q4_N_EMBD, 1024, 0);
            /* attn_v: [n_embd, n_head_kv*head_dim] Q4_K or Q6_K */
            if (l->attn_v->type != Q4_TENSOR_Q4_K && l->attn_v->type != Q4_TENSOR_Q6_K) { fprintf(stderr, "q4: attn_v type\n"); exit(1); }
            if (l->attn_v->dim[0] != Q4_N_EMBD || l->attn_v->dim[1] != 1024) { fprintf(stderr, "q4: attn_v dim\n"); exit(1); }
            /* attn_output: [n_head*q_head_dim, n_embd] Q4_K */
            tensor_expect_q4_k_layout(l->attn_output, 2, 6144, Q4_N_EMBD, 0);
            /* attn_q_norm: [256] F32 - normalized over first 256 dims of Q head */
            tensor_expect_f32_layout(l->attn_q_norm, 1, 256, 0, 0);
            /* attn_k_norm: [head_dim=256] F32 */
            tensor_expect_f32_layout(l->attn_k_norm, 1, 256, 0, 0);
            /* post_attn_norm: [n_embd] F32 */
            tensor_expect_f32_layout(l->post_attn_norm, 1, Q4_N_EMBD, 0, 0);
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

    /* tokenizer.ggml.tokens is a string array, not a single string.
     * Check that the key exists and is an array; the loop below reads it. */
    q4_kv *kv_check = model_find_kv(m, "tokenizer.ggml.tokens");
    if (!kv_check || kv_check->type != GGUF_VALUE_ARRAY) {
        q4_die("required tokenizer.ggml.tokens metadata key is missing or is not an array");
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
    case Q4_TENSOR_Q4_K: {
        /* Q4_K dequantize + F32 matvec */
        const uint64_t in_dim = w->dim[0];
        const uint64_t out_dim = w->dim[1];
        const uint64_t blocks = in_dim / QK_K;
        const block_q4_K *data = tensor_data(m, w);
        float *w_f32 = xmalloc(in_dim * sizeof(float));
        for (uint64_t row = 0; row < out_dim; row++) {
            const block_q4_K *row_b = data + row * blocks;
            for (uint64_t b = 0; b < blocks; b++) {
                const block_q4_K *xb = &row_b[b];
                const float d = f16_to_f32(xb->d);
                const float dm = f16_to_f32(xb->dmin);
                for (short e = 0; e < QK_K; e++) {
                    const short grp = e / 32;
                    const short pos = e % 32;
                    const short sub = pos / 16;
                    const short idx = pos % 16;
                    const bool is_high = grp >= 6;
                    const float dmul = is_high ? (d / 16.0f) : d;
                    float sc, mn;
                    if (grp < 6) {
                        sc = (float)(xb->scales[grp] & 0x3F);
                        mn = (float)(xb->scales[6 + grp] & 0x3F);
                    } else {
                        const short gi = grp - 6;
                        sc = (float)((xb->scales[4 + gi] >> 4) | ((xb->scales[gi] & 0xC0) >> 2));
                        mn = (float)((xb->scales[10 + gi] >> 4) | ((xb->scales[6 + gi] & 0xC0) >> 2));
                    }
                    const short qs_off = (grp / 2) * 32 + sub * 16;
                    const uint8_t qbyte = xb->qs[qs_off + idx];
                    const uint8_t mask = sub == 0 ? 0x0F : 0xF0;
                    const float q = (float)((qbyte & mask) >> (sub * 4));
                    w_f32[b * QK_K + e] = dmul * sc * q - dm * mn;
                }
            }
            double acc = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) acc += (double)w_f32[i] * x[i];
            out[row] = (float)acc;
            if (row == 0 && in_dim == 5120 && out_dim == 17408) {
                double w_abs_sum = 0, w_max = 0;
                for (uint64_t i = 0; i < in_dim; i++) {
                    w_abs_sum += fabs(w_f32[i]);
                    if (fabs(w_f32[i]) > w_max) w_max = fabs(w_f32[i]);
                }
                fprintf(stderr, "Q4K_WDBG: row=0 w_abs_avg=%.6f w_max=%.6f out=%.4f\n",
                        w_abs_sum / in_dim, w_max, out[0]);
            }
        }
        free(w_f32);
        break;
    }
    case Q4_TENSOR_Q5_K: {
        /* Q5_K dequantize + F32 matvec */
        const uint64_t in_dim = w->dim[0];
        const uint64_t out_dim = w->dim[1];
        const uint64_t blocks = in_dim / QK_K;
        const block_q5_K *data = tensor_data(m, w);
        float *w_f32 = xmalloc(in_dim * sizeof(float));
        for (uint64_t row = 0; row < out_dim; row++) {
            const block_q5_K *row_b = data + row * blocks;
            for (uint64_t b = 0; b < blocks; b++) {
                const block_q5_K *xb = &row_b[b];
                const float d = f16_to_f32(xb->d);
                const float dm = f16_to_f32(xb->dmin);
                for (short e = 0; e < QK_K; e++) {
                    const short grp = e / 32;
                    const short pos = e % 32;
                    const short sub = pos / 16;
                    const short idx = pos % 16;
                    const bool is_high = grp >= 6;
                    const float dmul = is_high ? (d / 16.0f) : d;
                    float sc, mn;
                    if (grp < 6) {
                        sc = (float)(xb->scales[grp] & 0x3F);
                        mn = (float)(xb->scales[6 + grp] & 0x3F);
                    } else {
                        const short gi = grp - 6;
                        sc = (float)((xb->scales[4 + gi] >> 4) | ((xb->scales[gi] & 0xC0) >> 2));
                        mn = (float)((xb->scales[10 + gi] >> 4) | ((xb->scales[6 + gi] & 0xC0) >> 2));
                    }
                    const short qs_off = (grp / 2) * 32 + sub * 16;
                    uint8_t q = xb->qs[qs_off + idx];
                    const uint8_t mask = sub == 0 ? 0x0F : 0xF0;
                    q = (q & mask) >> (sub * 4);
                    /* High bit from qh: 1 bit per element, packed 8 per byte */
                    const uint8_t h = (xb->qh[e / 8] >> (e % 8)) & 0x01;
                    q |= (h << 4);
                    w_f32[b * QK_K + e] = dmul * (float)q - dm * mn;
                }
            }
            double acc = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) acc += (double)w_f32[i] * x[i];
            out[row] = (float)acc;
        }
        free(w_f32);
        break;
    }
    case Q4_TENSOR_Q6_K: {
        /* Q6_K dequantize + F32 matvec */
        const uint64_t in_dim = w->dim[0];
        const uint64_t out_dim = w->dim[1];
        const uint64_t blocks = in_dim / QK_K;
        const block_q6_K *data = tensor_data(m, w);
        float *w_f32 = xmalloc(in_dim * sizeof(float));
        for (uint64_t row = 0; row < out_dim; row++) {
            const block_q6_K *row_b = data + row * blocks;
            for (uint64_t b = 0; b < blocks; b++) {
                const block_q6_K *xb = &row_b[b];
                const float d = f16_to_f32(xb->d);
                for (short e = 0; e < QK_K; e++) {
                    const short q = e / 64;       /* quarter 0..3 */
                    const short p = e % 64;       /* position within quarter 0..63 */
                    /* ql: 32 bytes per quarter, 2 elements per byte */
                    const int ql_idx = q * 32 + p / 2;
                    const uint8_t qlo = (xb->ql[ql_idx] >> ((p % 2) * 4)) & 0x0F;
                    /* qh: 16 bytes per quarter, 4 elements per byte */
                    const int qh_idx = q * 16 + p / 4;
                    const uint8_t qhi = (xb->qh[qh_idx] >> ((p % 4) * 2)) & 0x03;
                    /* scales: 4 per quarter */
                    const int s_idx = q * 4 + p / 16;
                    const int8_t q_val = (int8_t)((qlo | (qhi << 4)) - 32);
                    w_f32[b * QK_K + e] = d * xb->scales[s_idx] * q_val;
                }
            }
            double acc = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) acc += (double)w_f32[i] * x[i];
            out[row] = (float)acc;
        }
        free(w_f32);
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
 * Gated DeltaNet (CPU reference) - Gated DeltaNet (SSM) step for decode.
 * Implements: qkv = conv1d(matvec(x, W_qkv)), L2-norm q/k,
 * alpha/beta projections, delta rule state update, SiLU gate, RMS norm.
 * ========================================================================= */

static float softplus_f(float x) {
    /* numerically stable softplus: log(1 + exp(x)) */
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

static float sigmoid_f(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float l2_norm(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum);
}

/* Causal 1D conv: out[d] = sum_{k=0}^{kernel-1} w[k,d] * buf[pos-k, d] */
static void conv1d_step(const float *w, const float *buf, uint32_t pos,
                        int kernel, int dim, float *out) {
    for (int d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; k++) {
            int idx = (int)pos - k;
            if (idx < 0) idx += kernel;
            acc += w[k * dim + d] * buf[idx * dim + d];
        }
        out[d] = acc;
    }
}

static void deltanet_step_cpu(float *hidden_out, float *out_state,
                              const q4_model *m, const q4_layer_weights *l,
                              const float *normed, float *state,
                              float *conv_buf, uint32_t *conv_pos) {
    const uint32_t n_embd = Q4_N_EMBD;
    const uint32_t n_v_heads = Q4_N_V_HEADS;       /* 48 */
    const uint32_t n_k_groups = Q4_N_K_GROUPS;      /* 16 */
    const uint32_t head_k_dim = Q4_HEAD_K_DIM;      /* 128 */
    const uint32_t head_v_dim = Q4_HEAD_V_DIM;      /* 128 */
    const uint32_t qkv_dim = Q4_QKV_DIM;            /* 10240 */
    const uint32_t conv_kernel = Q4_CONV_KERNEL;    /* 4 */
    const uint32_t repeat = n_v_heads / n_k_groups; /* 3 */

    /* Temporary buffers */
    float *qkv_raw = xmalloc(qkv_dim * sizeof(float));
    float *conv_qkv = xmalloc(qkv_dim * sizeof(float));
    float *q_raw = xmalloc(n_k_groups * head_k_dim * sizeof(float));   /* 2048 */
    float *k_raw = xmalloc(n_k_groups * head_k_dim * sizeof(float));   /* 2048 */
    float *v_raw = xmalloc(n_v_heads * head_v_dim * sizeof(float));    /* 6144 */
    float *q_exp = xmalloc(n_v_heads * head_k_dim * sizeof(float));
    float *k_exp = xmalloc(n_v_heads * head_k_dim * sizeof(float));
    float *delta = xmalloc(head_v_dim * sizeof(float));
    float *z_raw = xmalloc(n_v_heads * head_v_dim * sizeof(float));
    float *output = xmalloc(n_v_heads * head_v_dim * sizeof(float));

    /* Step 1: qkv_raw = matvec(hidden, attn_qkv) */
    matvec_any(qkv_raw, m, l->attn_qkv, normed);

    /* Step 2: conv1d on qkv_raw */
    /* First, write qkv_raw into conv_buf at current position */
    {
        float *conv_buf_w = conv_buf + (*conv_pos) * qkv_dim;
        memcpy(conv_buf_w, qkv_raw, qkv_dim * sizeof(float));
    }

    const float *conv_w = tensor_data(m, l->ssm_conv1d);
    conv1d_step(conv_w, conv_buf, *conv_pos, conv_kernel, qkv_dim, conv_qkv);
    *conv_pos = (*conv_pos + 1) % conv_kernel;

    /* Step 3: Split conv_qkv into q, k, v */
    const uint32_t k_dim = n_k_groups * head_k_dim;  /* 2048 */
    memcpy(q_raw, conv_qkv, k_dim * sizeof(float));
    memcpy(k_raw, conv_qkv + k_dim, k_dim * sizeof(float));
    memcpy(v_raw, conv_qkv + 2 * k_dim, (qkv_dim - 2 * k_dim) * sizeof(float));

    /* Step 4: L2 norm q and k per group */
    {
        for (uint32_t g = 0; g < n_k_groups; g++) {
            float nq = l2_norm(q_raw + g * head_k_dim, head_k_dim);
            float nk = l2_norm(k_raw + g * head_k_dim, head_k_dim);
            if (nq > 1.0e-6f) {
                float inv = 1.0f / nq;
                for (uint32_t d = 0; d < head_k_dim; d++) q_raw[g * head_k_dim + d] *= inv;
            }
            if (nk > 1.0e-6f) {
                float inv = 1.0f / nk;
                for (uint32_t d = 0; d < head_k_dim; d++) k_raw[g * head_k_dim + d] *= inv;
            }
        }
    }

    /* Step 5: alpha_raw = matvec(normed, ssm_alpha), beta_raw = matvec(normed, ssm_beta) */
    float *alpha_raw = xmalloc(n_v_heads * sizeof(float));
    float *beta_raw = xmalloc(n_v_heads * sizeof(float));
    matvec_any(alpha_raw, m, l->ssm_alpha, normed);
    matvec_any(beta_raw, m, l->ssm_beta, normed);

    /* Step 6: alpha_biased = alpha_raw + ssm_dt_bias, gate = softplus(alpha_biased) * ssm_a */
    const float *dt_bias = tensor_data(m, l->ssm_dt_bias);
    const float *ssm_a_data = tensor_data(m, l->ssm_a);
    float *gate = xmalloc(n_v_heads * sizeof(float));
    for (uint32_t i = 0; i < n_v_heads; i++) {
        float biased = alpha_raw[i] + dt_bias[i];
        float sp = softplus_f(biased);
        gate[i] = sp * ssm_a_data[i];
    }

    /* Step 7: beta = sigmoid(beta_raw) */
    for (uint32_t i = 0; i < n_v_heads; i++) {
        beta_raw[i] = sigmoid_f(beta_raw[i]);
    }

    /* Step 8: Expand q, k from key_heads(16) to value_heads(48) */
    for (uint32_t g = 0; g < n_k_groups; g++) {
        for (uint32_t r = 0; r < repeat; r++) {
            memcpy(q_exp + (g * repeat + r) * head_k_dim,
                   q_raw + g * head_k_dim,
                   head_k_dim * sizeof(float));
            memcpy(k_exp + (g * repeat + r) * head_k_dim,
                   k_raw + g * head_k_dim,
                   head_k_dim * sizeof(float));
        }
    }

    /* Step 9: Delta rule per v_head */
    for (uint32_t i = 0; i < n_v_heads; i++) {
        float *state_i = state + i * head_v_dim * head_k_dim;
        const float *k_i = k_exp + i * head_k_dim;
        const float *v_i = v_raw + i * head_v_dim;
        float gi = gate[i];
        float bi = beta_raw[i];
        float eg = expf(gi);

        /* sk = state @ k */
        for (uint32_t d = 0; d < head_v_dim; d++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < head_k_dim; j++) {
                acc += state_i[d * head_k_dim + j] * k_i[j];
            }
            delta[d] = v_i[d] - acc;
        }

        /* state = state * eg + bi * outer(delta, k) */
        for (uint32_t d = 0; d < head_v_dim; d++) {
            for (uint32_t j = 0; j < head_k_dim; j++) {
                state_i[d * head_k_dim + j] = state_i[d * head_k_dim + j] * eg + bi * delta[d] * k_i[j];
            }
        }

        /* output = state @ k */
        for (uint32_t d = 0; d < head_v_dim; d++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < head_k_dim; j++) {
                acc += state_i[d * head_k_dim + j] * k_i[j];
            }
            output[d + i * head_v_dim] = acc;
        }
    }

    free(alpha_raw);
    free(beta_raw);
    free(gate);

    /* Step 10: z = matvec(normed, attn_gate), output *= silu(z) */
    matvec_any(z_raw, m, l->attn_gate, normed);

    const float *ssm_norm_w = tensor_data(m, l->ssm_norm);

    for (uint32_t i = 0; i < n_v_heads; i++) {
        float *out_i = output + i * head_v_dim;
        const float *z_i = z_raw + i * head_v_dim;

        /* silu gate */
        for (uint32_t d = 0; d < head_v_dim; d++) {
            out_i[d] *= z_i[d] * sigmoid_f(z_i[d]);
        }

        /* RMS norm over head_v_dim */
        float ss = 0.0f;
        for (uint32_t d = 0; d < head_v_dim; d++) ss += out_i[d] * out_i[d];
        float rms = sqrtf(ss / head_v_dim + Q4_RMS_EPS);
        for (uint32_t d = 0; d < head_v_dim; d++) {
            out_i[d] = out_i[d] / rms * ssm_norm_w[d];
        }
    }

    /* Step 11: Copy output to hidden_out [n_v_heads * head_v_dim = 6144] */
    memcpy(hidden_out, output, n_v_heads * head_v_dim * sizeof(float));

    /* Step 12: Project to n_embd via ssm_out */
    float *proj_tmp = xmalloc(n_embd * sizeof(float));

    /* Skip ssm_out matvec if hidden_out is effectively zero (avoids Q5_K dequant NaN) */
    {
        float h_max = 0;
        for (uint32_t i = 0; i < n_v_heads * head_v_dim; i++) {
            float v = fabsf(hidden_out[i]);
            if (v > h_max) h_max = v;
        }
        if (h_max < 1e-10f) {
            memset(proj_tmp, 0, n_embd * sizeof(float));
        } else {
            matvec_any(proj_tmp, m, l->ssm_out, hidden_out);
        }
    }
    memcpy(hidden_out, proj_tmp, n_embd * sizeof(float));
    free(proj_tmp);

    /* Step 13: Copy state back */
    memcpy(out_state, state, n_v_heads * head_v_dim * head_k_dim * sizeof(float));

    free(qkv_raw); free(conv_qkv); free(q_raw); free(k_raw); free(v_raw);
    free(q_exp); free(k_exp); free(delta); free(z_raw); free(output);
}

/* =========================================================================
 * Gated Attention (CPU reference).
 * ========================================================================= */

static void attention_decode_cpu(float *out, const q4_model *m, const q4_layer_weights *l,
                                 const float *x, const float *k_cache, const float *v_cache,
                                 uint32_t kv_len, uint32_t pos, float logit_softcap,
                                 float rope_freq_base) {
    const uint32_t n_q_heads = Q4_N_HEAD;
    const uint32_t n_kv_heads = Q4_N_HEAD_KV;
    const uint32_t q_head_dim = Q4_Q_HEAD_DIM;    /* 512 */
    const uint32_t kv_head_dim = Q4_HEAD_DIM;     /* 256 */
    const uint32_t n_embd = Q4_N_EMBD;
    const uint32_t q_per_kv = Q4_Q_PER_KV;
    const float inv_sqrt_d = 1.0f / sqrtf((float)kv_head_dim);

    // Q = x @ W_q  -> [n_q_heads * q_head_dim = 24 * 512 = 12288]
    float *q = xmalloc(n_q_heads * q_head_dim * sizeof(float));
    matvec_any(q, m, l->attn_q, x);

    // RoPE applied to Q (only first rope.dimension_count=64 dims are rotated)
    rope_full_cpu(q, 1, n_q_heads, q_head_dim, pos, rope_freq_base);

    for (uint32_t qh = 0; qh < n_q_heads; qh++) {
        const uint32_t kv_h = qh / q_per_kv;
        const float *q_h = q + qh * q_head_dim;

        // Compute attention scores
        float *scores = xmalloc(kv_len * sizeof(float));
        float max_val = Q4_NEG_INF;
        for (uint32_t t = 0; t < kv_len; t++) {
            const float *k_t = k_cache + (kv_h * kv_head_dim) + (t * n_kv_heads * kv_head_dim);
            float score = 0.0f;
            for (uint32_t d = 0; d < kv_head_dim; d++) {
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

        // Weighted sum of V -> [kv_head_dim] per Q head
        float *out_h = out + qh * kv_head_dim;
        for (uint32_t d = 0; d < kv_head_dim; d++) out_h[d] = 0.0f;
        for (uint32_t t = 0; t < kv_len; t++) {
            const float *v_t = v_cache + (kv_h * kv_head_dim) + (t * n_kv_heads * kv_head_dim);
            const float w = scores[t] * inv_sum;
            for (uint32_t d = 0; d < kv_head_dim; d++) {
                out_h[d] += w * v_t[d];
            }
        }

        free(scores);
    }

    // Output projection: [n_q_heads * kv_head_dim = 24*256 = 6144] -> [n_embd]
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
    float *k_cache;  // [ctx_size, n_kv_heads=4, head_dim=256]
    float *v_cache;  // [ctx_size, n_kv_heads=4, head_dim=256]
    uint32_t kv_len;

    // SSM state per layer: [n_layer, n_v_heads=48, head_v_dim=128, head_k_dim=128]
    float *ssm_state;

    // Conv buffer per layer: [n_layer, conv_kernel=4, qkv_dim=10240]
    // Stored as circular buffer: conv_pos wraps around
    float *conv_buf;
    uint32_t *conv_pos;  // [n_layer] current write position in conv buffer

    // Token buffer
    int *tokens;
    int n_tokens;

    // Logits
    float *logits;

#ifndef Q4_NO_GPU
    // GPU-side state (allocated when backend uses GPU)
    q4_gpu_tensor *gpu_hidden;      /* [n_embd] F32 */
    q4_gpu_tensor *gpu_residual;    /* [n_embd] F32 */
    q4_gpu_tensor *gpu_normed;      /* [n_embd] F32 */
    q4_gpu_tensor *gpu_logits;      /* [n_vocab] F32 */
    q4_gpu_tensor *gpu_kv_k;        /* [ctx_size, n_kv_heads, head_dim] F32 */
    q4_gpu_tensor *gpu_kv_v;        /* [ctx_size, n_kv_heads, head_dim] F32 */
    q4_gpu_tensor *gpu_ssm_state;   /* [n_layer, n_v_heads, head_v_dim, head_k_dim] F32 */
    q4_gpu_tensor *gpu_conv_buf;    /* [n_layer, Q4_CONV_KERNEL, Q4_QKV_DIM] F32 ring buffer */
    /* Scratch buffers reused across layers */
    q4_gpu_tensor *gpu_scratch;     /* large scratch buffer */
    uint64_t gpu_scratch_offset;    /* current offset in scratch */
#endif
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
    e->config.n_layer = required_u32(&e->model, "qwen35.block_count");
    e->config.n_embd = required_u32(&e->model, "qwen35.embedding_length");
    e->config.n_head = required_u32(&e->model, "qwen35.attention.head_count");
    e->config.n_head_kv = required_u32(&e->model, "qwen35.attention.head_count_kv");
    e->config.head_dim = required_u32(&e->model, "qwen35.attention.key_length");
    e->config.n_ffn = required_u32(&e->model, "qwen35.feed_forward_length");
    // vocab_size from embedding tensor (second dimension)
    q4_tensor *token_embd = required_tensor(&e->model, "token_embd.weight");
    e->config.n_vocab = (uint32_t)token_embd->dim[1];

    // Derived
    e->config.n_q_dim = e->config.n_head * e->config.head_dim;
    e->config.n_kv_dim = e->config.n_head_kv * e->config.head_dim;

    // Optional metadata
    model_get_f32(&e->model, "qwen35.rope.freq_base", &e->rope_freq_base);
    if (e->rope_freq_base < 1.0e-6f) e->rope_freq_base = 10000000.0f;
    model_get_f32(&e->model, "qwen35.attention.logit_softcap", &e->logit_softcap);

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
    int n = vocab_tokenize(&e->vocab, text, text_len, out_tokens);
    return n > 0 ? 0 : -1;
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

    // SSM state: [n_layer, n_v_heads, head_v_dim, head_k_dim]
    const uint64_t ssm_state_bytes = (uint64_t)Q4_N_LAYER * Q4_N_V_HEADS * Q4_HEAD_V_DIM * Q4_HEAD_K_DIM * sizeof(float);
    s->ssm_state = xmalloc_zeroed(1, ssm_state_bytes);

    // Conv buffer: [n_layer, conv_kernel, qkv_dim]
    const uint64_t conv_buf_bytes = (uint64_t)Q4_N_LAYER * Q4_CONV_KERNEL * Q4_QKV_DIM * sizeof(float);
    s->conv_buf = xmalloc_zeroed(1, conv_buf_bytes);
    s->conv_pos = xmalloc_zeroed(Q4_N_LAYER, sizeof(uint32_t));

    // Token buffer
    s->tokens = xmalloc((size_t)ctx_size * sizeof(int));
    s->n_tokens = 0;

    // Logits (vocab-sized)
    s->logits = xmalloc((size_t)e->config.n_vocab * sizeof(float));

#ifndef Q4_NO_GPU
    // Allocate GPU-side state
    if (q4_backend_uses_graph(e->backend)) {
        const uint64_t embd_bytes = (uint64_t)Q4_N_EMBD * sizeof(float);
        s->gpu_hidden   = q4_gpu_tensor_alloc(embd_bytes);
        s->gpu_residual = q4_gpu_tensor_alloc(embd_bytes);
        s->gpu_normed   = q4_gpu_tensor_alloc(embd_bytes);
        s->gpu_logits   = q4_gpu_tensor_alloc((uint64_t)e->config.n_vocab * sizeof(float));
        s->gpu_kv_k     = q4_gpu_tensor_alloc((uint64_t)ctx_size * kv_head_bytes);
        s->gpu_kv_v     = q4_gpu_tensor_alloc((uint64_t)ctx_size * kv_head_bytes);
        s->gpu_ssm_state = q4_gpu_tensor_alloc(ssm_state_bytes);
        /* GPU conv buffer: [n_layer, Q4_CONV_KERNEL, Q4_QKV_DIM] F32 ring buffer */
        const uint64_t conv_buf_bytes = (uint64_t)Q4_N_LAYER * Q4_CONV_KERNEL * Q4_QKV_DIM * sizeof(float);
        s->gpu_conv_buf = q4_gpu_tensor_alloc(conv_buf_bytes);
        /* Zero-initialize conv buffer */
        q4_gpu_tensor_fill_f32(s->gpu_conv_buf, 0.0f, Q4_N_LAYER * Q4_CONV_KERNEL * Q4_QKV_DIM);
        /* Scratch buffer: enough for Q [24*512], K/V [4*256], attn_out [5120], ffn_gate [17408], ffn_up [17408], ffn_out [5120] */
        const uint64_t scratch_bytes = 512 * 1024;  /* 512 KB: attn Q/K/V/out (~78K) + FFN gate/up/out (~160K) + DeltaNet intermediates (~350K) */
        s->gpu_scratch = q4_gpu_tensor_alloc(scratch_bytes);
        s->gpu_scratch_offset = 0;
    }
#endif

    *out = s;
    return 0;
}

void q4_session_free(q4_session *s) {
    if (!s) return;
    free(s->k_cache);
    free(s->v_cache);
    free(s->ssm_state);
    free(s->conv_buf);
    free(s->conv_pos);
    free(s->tokens);
    free(s->logits);
#ifndef Q4_NO_GPU
    q4_gpu_tensor_free(s->gpu_hidden);
    q4_gpu_tensor_free(s->gpu_residual);
    q4_gpu_tensor_free(s->gpu_normed);
    q4_gpu_tensor_free(s->gpu_logits);
    q4_gpu_tensor_free(s->gpu_kv_k);
    q4_gpu_tensor_free(s->gpu_kv_v);
    q4_gpu_tensor_free(s->gpu_ssm_state);
    q4_gpu_tensor_free(s->gpu_conv_buf);
    q4_gpu_tensor_free(s->gpu_scratch);
#endif
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
    const uint64_t stride = te->dim[0];  /* n_embd */
    const uint64_t token_off = (uint64_t)token * stride;

    if (te->type == Q4_TENSOR_F16) {
        const uint16_t *base = tensor_data(m, te);
        const uint16_t *row = base + token_off;
        for (uint64_t i = 0; i < stride; i++) {
            out[i] = f16_to_f32(row[i]);
        }
    } else if (te->type == Q4_TENSOR_Q4_K) {
        const block_q4_K *data = tensor_data(m, te);
        const uint64_t blocks_per_row = stride / QK_K;
        for (uint64_t b = 0; b < blocks_per_row; b++) {
            const block_q4_K *xb = &data[token_off / QK_K * blocks_per_row + b];
            const float d = f16_to_f32(xb->d);
            const float dm = f16_to_f32(xb->dmin);
            for (short e = 0; e < QK_K; e++) {
                const short grp = e / 32;
                const short pos = e % 32;
                const short sub = pos / 16;
                const short idx = pos % 16;
                const float dmul = (grp >= 6) ? (d / 16.0f) : d;
                float sc, mn;
                if (grp < 6) {
                    sc = (float)(xb->scales[grp] & 0x3F);
                    mn = (float)(xb->scales[6 + grp] & 0x3F);
                } else {
                    const short gi = grp - 6;
                    sc = (float)((xb->scales[4 + gi] >> 4) | ((xb->scales[gi] & 0xC0) >> 2));
                    mn = (float)((xb->scales[10 + gi] >> 4) | ((xb->scales[6 + gi] & 0xC0) >> 2));
                }
                const short qs_off = (grp / 2) * 32 + sub * 16;
                const uint8_t qbyte = xb->qs[qs_off + idx];
                const uint8_t mask = sub == 0 ? 0x0F : 0xF0;
                const float q = (float)((qbyte & mask) >> (sub * 4));
                out[b * QK_K + e] = dmul * sc * q - dm * mn;
            }
        }
    } else {
        q4_die("embed_token_cpu: unsupported tensor type");
    }
}

/* Run one token through the model on CPU. */
static int q4_cpu_forward(q4_engine *e, q4_session *s, int token, uint32_t pos, char *err, size_t errlen) {
    const uint32_t n_embd = Q4_N_EMBD;
    const uint32_t n_layer = Q4_N_LAYER;

    float *hidden = xmalloc(n_embd * sizeof(float));
    float *residual = xmalloc(n_embd * sizeof(float));

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
            // DeltaNet: compute output from SSM state, project to hidden
            float *layer_ssm_state = s->ssm_state + il * Q4_N_V_HEADS * Q4_HEAD_V_DIM * Q4_HEAD_K_DIM;
            float *state_tmp = xmalloc(Q4_N_V_HEADS * Q4_HEAD_V_DIM * Q4_HEAD_K_DIM * sizeof(float));
            float *conv_buf = s->conv_buf + il * Q4_CONV_KERNEL * Q4_QKV_DIM;
            uint32_t *conv_pos = &s->conv_pos[il];

            deltanet_step_cpu(hidden, state_tmp, &e->model, l, normed, layer_ssm_state, conv_buf, conv_pos);

            // Copy updated state back to persistent storage
            memcpy(layer_ssm_state, state_tmp, Q4_N_V_HEADS * Q4_HEAD_V_DIM * Q4_HEAD_K_DIM * sizeof(float));
            free(state_tmp);
        } else {
            // Gated Attention
            float *attn_out = xmalloc(n_embd * sizeof(float));

            // Store K, V in cache
            const uint32_t kv_idx = s->kv_len;
            if (kv_idx >= (uint32_t)s->ctx_size) {
                if (err && errlen > 0) snprintf(err, errlen, "KV cache full");
                free(normed); free(attn_out); free(hidden); free(residual);
                return -1;
            }

            // Project Q, K, V
            float *q = xmalloc(Q4_N_HEAD * Q4_Q_HEAD_DIM * sizeof(float));
            float *k = xmalloc(Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float));
            float *v = xmalloc(Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float));

            matvec_any(q, &e->model, l->attn_q, normed);
            matvec_any(k, &e->model, l->attn_k, normed);
            matvec_any(v, &e->model, l->attn_v, normed);

            // Apply RoPE to Q and K
            rope_full_cpu(q, 1, Q4_N_HEAD, Q4_Q_HEAD_DIM, pos, e->rope_freq_base);
            rope_full_cpu(k, 1, Q4_N_HEAD_KV, Q4_HEAD_DIM, pos, e->rope_freq_base);

            // Store K, V
            const uint64_t kv_head_bytes = (uint64_t)Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float);
            memcpy(s->k_cache + kv_idx * Q4_N_HEAD_KV * Q4_HEAD_DIM, k, kv_head_bytes);
            memcpy(s->v_cache + kv_idx * Q4_N_HEAD_KV * Q4_HEAD_DIM, v, kv_head_bytes);

            // Run attention (attention_decode_cpu will re-project Q internally, which is wasteful
            // but the KV cache lookup and scoring are correct)
            attention_decode_cpu(attn_out, &e->model, l, normed,
                               s->k_cache, s->v_cache, s->kv_len + 1, pos, e->logit_softcap,
                               e->rope_freq_base);

            // Increment KV cache length
            s->kv_len++;

            free(q); free(k); free(v);

            memcpy(hidden, attn_out, n_embd * sizeof(float));
            free(attn_out);
        }

        // Post-attention/post-SSM norm + residual
        float *post_normed = xmalloc(n_embd * sizeof(float));
        const float *post_norm_weight = tensor_data(&e->model, l->post_attn_norm);
        rms_norm_weight(post_normed, hidden, post_norm_weight, n_embd, Q4_RMS_EPS);
        for (uint32_t i = 0; i < n_embd; i++) {
            hidden[i] = post_normed[i] + residual[i];
        }
        memcpy(residual, hidden, n_embd * sizeof(float));

        free(normed);
        free(post_normed);

        // FFN
        normed = xmalloc(n_embd * sizeof(float));
        const float *ffn_norm_weight = tensor_data(&e->model, l->post_attn_norm);
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
    matvec_any(s->logits, &e->model, e->weights.output, hidden);

    free(hidden);
    free(residual);

    return 0;
}

/* =========================================================================
 * GPU Inference Path.
 * ========================================================================= */

#ifndef Q4_NO_GPU
/* Scratch allocator within the session's gpu_scratch tensor. */
static uint64_t gpu_scratch_alloc(q4_session *s, uint64_t bytes) {
    uint64_t offset = s->gpu_scratch_offset;
    s->gpu_scratch_offset += bytes;
    return offset;
}

static q4_gpu_tensor *gpu_scratch_view(q4_session *s, uint64_t offset, uint64_t bytes) {
    q4_gpu_tensor *view = q4_gpu_tensor_view(s->gpu_scratch, offset, bytes);
    return view;
}

static void gpu_scratch_reset(q4_session *s) {
    s->gpu_scratch_offset = 0;
}

/* Run one token through the model on GPU. */
static int q4_gpu_forward(q4_engine *e, q4_session *s, int token, uint32_t pos, char *err, size_t errlen) {
    const uint32_t n_embd = Q4_N_EMBD;
    const uint32_t n_layer = Q4_N_LAYER;
    const uint64_t embd_bytes = n_embd * sizeof(float);
    const uint64_t ffn_bytes = Q4_N_FFN * sizeof(float);
    const uint64_t q_bytes = Q4_N_HEAD * Q4_Q_HEAD_DIM * sizeof(float);   /* 24*512*4 = 49152 */
    const uint64_t kv_bytes = Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float);  /* 4*256*4 = 4096 */
    const uint64_t kv_head_bytes = (uint64_t)Q4_N_HEAD_KV * Q4_HEAD_DIM * sizeof(float);

    /* Reset scratch allocator */
    gpu_scratch_reset(s);

    /* 1. Embed token on CPU (fast: 5120 floats), upload to GPU */
    float *host_hidden = xmalloc(embd_bytes);
    embed_token_cpu(&e->model, e->weights.token_embd, token, host_hidden);
    if (q4_gpu_tensor_write(s->gpu_hidden, 0, host_hidden, embd_bytes) != 0) {
        free(host_hidden);
        if (err && errlen > 0) snprintf(err, errlen, "failed to write hidden state");
        return -1;
    }
    /* Copy hidden to residual for residual connection */
    if (q4_gpu_tensor_copy(s->gpu_residual, 0, s->gpu_hidden, 0, embd_bytes) != 0) {
        free(host_hidden);
        return -1;
    }
    free(host_hidden);

    /* Begin GPU command sequence */
    if (q4_gpu_begin_commands() != 0) {
        if (err && errlen > 0) snprintf(err, errlen, "failed to begin GPU commands");
        return -1;
    }

    for (uint32_t il = 0; il < n_layer; il++) {
        gpu_scratch_reset(s);

        const q4_layer_weights *l = &e->weights.layer[il];
        const void *model_map = e->model.map;
        uint64_t model_size = e->model.size;

        /* --- RMSNorm: gpu_hidden -> gpu_normed --- */
        uint64_t norm_w_offset = l->attn_norm->abs_offset;
        if (q4_gpu_rms_norm_weight_rows_tensor(s->gpu_normed, s->gpu_hidden,
                model_map, model_size, norm_w_offset, n_embd, 1, Q4_RMS_EPS) != 0) {
            if (err && errlen > 0) snprintf(err, errlen, "layer %u RMSNorm failed", il);
            return -1;
        }

        if (layer_is_deltanet(il)) {
            /* DeltaNet on GPU: chain matmuls + specialized kernels. */
            const uint64_t qkv_raw_bytes = (uint64_t)Q4_QKV_DIM * sizeof(float);          /* 163840 */
            const uint64_t small_bytes = Q4_N_V_HEADS * sizeof(float);                      /* 192 */
            const uint64_t z_raw_bytes = (uint64_t)Q4_N_V_HEADS * Q4_HEAD_V_DIM * sizeof(float); /* 24576 */
            const uint64_t expand_bytes = z_raw_bytes;                                       /* 24576 each */

            uint64_t gate_off = gpu_scratch_alloc(s, small_bytes);
            uint64_t beta_sig_off = gpu_scratch_alloc(s, small_bytes);
            uint64_t q_exp_off = gpu_scratch_alloc(s, expand_bytes);
            uint64_t k_exp_off = gpu_scratch_alloc(s, expand_bytes);
            uint64_t v_off = gpu_scratch_alloc(s, expand_bytes);
            uint64_t delta_out_off = gpu_scratch_alloc(s, expand_bytes);
            uint64_t proj_off = gpu_scratch_alloc(s, embd_bytes);

            q4_gpu_tensor *gate_t = gpu_scratch_view(s, gate_off, small_bytes);
            q4_gpu_tensor *beta_sig_t = gpu_scratch_view(s, beta_sig_off, small_bytes);
            q4_gpu_tensor *q_exp_t = gpu_scratch_view(s, q_exp_off, expand_bytes);
            q4_gpu_tensor *k_exp_t = gpu_scratch_view(s, k_exp_off, expand_bytes);
            q4_gpu_tensor *v_raw_t = gpu_scratch_view(s, v_off, expand_bytes);
            q4_gpu_tensor *delta_out_t = gpu_scratch_view(s, delta_out_off, expand_bytes);
            q4_gpu_tensor *proj_t = gpu_scratch_view(s, proj_off, embd_bytes);

            /* 1-4. Fused: qkv_raw, alpha_raw, beta_raw, z_raw = normed @ [attn_qkv, ssm_alpha, ssm_beta, attn_gate] (Q4_K) */
            /* Output buffer layout: scratch space must hold qkv_raw(10240) + alpha_raw(48) + beta_raw(48) + z_raw(6144) = 16280 floats */
            /* But our scratch tensors are separate, so we need a combined output buffer.
             * Instead, we compute all 4 into a single large scratch region, then create views. */
            const uint64_t fused_out_bytes = (Q4_QKV_DIM + Q4_N_V_HEADS + Q4_N_V_HEADS + Q4_N_V_HEADS * Q4_HEAD_V_DIM) * sizeof(float);
            uint64_t fused_off = gpu_scratch_alloc(s, fused_out_bytes);
            q4_gpu_tensor *fused_out_t = gpu_scratch_view(s, fused_off, fused_out_bytes);

            const uint64_t fused_weights[4] = {
                l->attn_qkv->abs_offset,
                l->ssm_alpha->abs_offset,
                l->ssm_beta->abs_offset,
                l->attn_gate->abs_offset,
            };
            const uint32_t fused_out_dims[4] = {
                Q4_QKV_DIM, Q4_N_V_HEADS, Q4_N_V_HEADS, Q4_N_V_HEADS * Q4_HEAD_V_DIM,
            };
            if (q4_gpu_matmul_q4_k_fused4_tensor(fused_out_t, model_map, model_size,
                    fused_weights, fused_out_dims, n_embd, s->gpu_normed) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u DeltaNet fused matmuls failed", il);
                return -1;
            }

            /* Create views into the fused output for downstream kernels */
            const uint64_t qkv_float_off = 0;
            const uint64_t alpha_float_off = Q4_QKV_DIM;
            const uint64_t beta_float_off = Q4_QKV_DIM + Q4_N_V_HEADS;
            const uint64_t z_float_off = Q4_QKV_DIM + Q4_N_V_HEADS + Q4_N_V_HEADS;
            q4_gpu_tensor *qkv_raw_t = q4_gpu_tensor_view(fused_out_t, qkv_float_off * sizeof(float), qkv_raw_bytes);
            q4_gpu_tensor *alpha_raw_t = q4_gpu_tensor_view(fused_out_t, alpha_float_off * sizeof(float), small_bytes);
            q4_gpu_tensor *beta_raw_t = q4_gpu_tensor_view(fused_out_t, beta_float_off * sizeof(float), small_bytes);
            q4_gpu_tensor *z_raw_t = q4_gpu_tensor_view(fused_out_t, z_float_off * sizeof(float), z_raw_bytes);

            /* 5. conv1D + split + L2 norm + expand */
            const uint64_t layer_conv_off = il * Q4_CONV_KERNEL * Q4_QKV_DIM * sizeof(float);
            q4_gpu_tensor *conv_buf_view = q4_gpu_tensor_view(s->gpu_conv_buf, layer_conv_off,
                    Q4_CONV_KERNEL * Q4_QKV_DIM * sizeof(float));
            q4_gpu_tensor *conv_buf_rw_view = q4_gpu_tensor_view(s->gpu_conv_buf, layer_conv_off,
                    Q4_CONV_KERNEL * Q4_QKV_DIM * sizeof(float));

            if (q4_gpu_deltanet_conv_split_tensor(qkv_raw_t, conv_buf_view, conv_buf_rw_view,
                    model_map, model_size, l->ssm_conv1d->abs_offset,
                    q_exp_t, k_exp_t, v_raw_t,
                    Q4_QKV_DIM, Q4_N_HEAD_KV, Q4_N_V_HEADS,
                    Q4_HEAD_K_DIM, Q4_HEAD_V_DIM,
                    Q4_N_V_HEADS / Q4_N_HEAD_KV,
                    s->conv_pos[il]) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u DeltaNet conv/split failed", il);
                return -1;
            }

            /* 6. Gate transform: gate = softplus(alpha+bias)*a, beta = sigmoid(beta) */
            if (q4_gpu_deltanet_gate_transform_tensor(alpha_raw_t, beta_raw_t,
                    gate_t, beta_sig_t,
                    model_map, model_size,
                    l->ssm_dt_bias->abs_offset, l->ssm_a->abs_offset,
                    Q4_N_V_HEADS) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u DeltaNet gate transform failed", il);
                return -1;
            }

            /* 7. Delta rule */
            if (q4_gpu_delta_rule_tensor(s->gpu_ssm_state, k_exp_t, v_raw_t,
                    gate_t, beta_sig_t, delta_out_t,
                    Q4_N_V_HEADS, Q4_HEAD_V_DIM, Q4_HEAD_K_DIM) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u DeltaNet delta rule failed", il);
                return -1;
            }

            /* 8. SiLU + RMS norm */
            if (q4_gpu_deltanet_silu_rms_tensor(delta_out_t, z_raw_t,
                    model_map, model_size, l->ssm_norm->abs_offset,
                    Q4_N_V_HEADS, Q4_HEAD_V_DIM) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u DeltaNet SiLU/RMS failed", il);
                return -1;
            }

            /* 9. ssm_out projection: delta_out [6144] -> proj [5120] via Q5_K */
            if (q4_gpu_vec_matmul_q5k_tensor(proj_t, model_map, model_size,
                    l->ssm_out->abs_offset,
                    Q4_N_V_HEADS * Q4_HEAD_V_DIM, n_embd, delta_out_t) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u DeltaNet ssm_out failed", il);
                return -1;
            }

            /* 10. Copy proj to gpu_hidden */
            if (q4_gpu_tensor_copy(s->gpu_hidden, 0, proj_t, 0, embd_bytes) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u DeltaNet copy failed", il);
                return -1;
            }

            /* 11. Update conv_pos */
            s->conv_pos[il] = (s->conv_pos[il] + 1) % Q4_CONV_KERNEL;

            /* SSM state stays on GPU between tokens — only read back at session save time. */
        } else {
            /* --- Gated Attention --- */

            /* Q, K, V projections from normed input.
             * Fuse into single dispatch when all weights are Q4_K. */
            q4_gpu_tensor *q_t, *k_t, *v_t;

            if (l->attn_q->type == Q4_TENSOR_Q4_K &&
                l->attn_k->type == Q4_TENSOR_Q4_K &&
                l->attn_v->type == Q4_TENSOR_Q4_K) {
                /* Fused QKV: 3 matmuls from same input in one dispatch. */
                const uint32_t q_out_dim = Q4_N_HEAD * Q4_Q_HEAD_DIM;    /* 12288 */
                const uint32_t kv_out_dim = Q4_N_HEAD_KV * Q4_HEAD_DIM;  /* 1024 */
                const uint64_t fused_qkv_bytes = (q_out_dim + kv_out_dim + kv_out_dim) * sizeof(float);
                uint64_t fused_qkv_off = gpu_scratch_alloc(s, fused_qkv_bytes);
                q4_gpu_tensor *fused_qkv_t = gpu_scratch_view(s, fused_qkv_off, fused_qkv_bytes);

                const uint64_t qkv_weights[4] = {
                    l->attn_q->abs_offset, l->attn_k->abs_offset, l->attn_v->abs_offset, 0,
                };
                const uint32_t qkv_dims[4] = { q_out_dim, kv_out_dim, kv_out_dim, 0 };
                if (q4_gpu_matmul_q4_k_fused4_tensor(fused_qkv_t, model_map, model_size,
                        qkv_weights, qkv_dims, n_embd, s->gpu_normed) != 0) {
                    if (err && errlen > 0) snprintf(err, errlen, "layer %u fused QKV projection failed", il);
                    return -1;
                }

                q_t = q4_gpu_tensor_view(fused_qkv_t, 0, q_bytes);
                k_t = q4_gpu_tensor_view(fused_qkv_t, (uint64_t)q_out_dim * sizeof(float), kv_bytes);
                v_t = q4_gpu_tensor_view(fused_qkv_t, (uint64_t)(q_out_dim + kv_out_dim) * sizeof(float), kv_bytes);
            } else {
                /* Fallback: separate matmuls. */
                uint64_t q_off = gpu_scratch_alloc(s, q_bytes);
                uint64_t k_off = gpu_scratch_alloc(s, kv_bytes);
                uint64_t v_off = gpu_scratch_alloc(s, kv_bytes);
                q_t = gpu_scratch_view(s, q_off, q_bytes);
                k_t = gpu_scratch_view(s, k_off, kv_bytes);
                v_t = gpu_scratch_view(s, v_off, kv_bytes);

                if (q4_gpu_matmul_any_tensor(q_t, model_map, model_size, l->attn_q->abs_offset,
                        n_embd, Q4_N_HEAD * Q4_Q_HEAD_DIM, l->attn_q->type, s->gpu_normed, 1) != 0 ||
                    q4_gpu_matmul_any_tensor(k_t, model_map, model_size, l->attn_k->abs_offset,
                        n_embd, Q4_N_HEAD_KV * Q4_HEAD_DIM, l->attn_k->type, s->gpu_normed, 1) != 0 ||
                    q4_gpu_matmul_any_tensor(v_t, model_map, model_size, l->attn_v->abs_offset,
                        n_embd, Q4_N_HEAD_KV * Q4_HEAD_DIM, l->attn_v->type, s->gpu_normed, 1) != 0) {
                    if (err && errlen > 0) snprintf(err, errlen, "layer %u QKV projection failed", il);
                    return -1;
                }
            }

            /* Apply RoPE to Q and K */
            if (q4_gpu_rope_full_tensor(q_t, 1, Q4_N_HEAD, Q4_Q_HEAD_DIM, pos, e->rope_freq_base, 1.0f) != 0 ||
                q4_gpu_rope_full_tensor(k_t, 1, Q4_N_HEAD_KV, Q4_HEAD_DIM, pos, e->rope_freq_base, 1.0f) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u RoPE failed", il);
                return -1;
            }

            /* Store K, V to KV cache at position pos */
            uint64_t cache_k_off = (uint64_t)pos * kv_head_bytes;
            uint64_t cache_v_off = (uint64_t)pos * kv_head_bytes;
            q4_gpu_tensor *cache_k_view = q4_gpu_tensor_view(s->gpu_kv_k, cache_k_off, kv_bytes);
            q4_gpu_tensor *cache_v_view = q4_gpu_tensor_view(s->gpu_kv_v, cache_v_off, kv_bytes);

            if (q4_gpu_kv_cache_store_tensor(k_t, v_t, cache_k_view, cache_v_view,
                    0, Q4_N_HEAD_KV, Q4_HEAD_DIM) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u KV cache store failed", il);
                return -1;
            }

            /* Flash Attention: q_t -> gpu_hidden (output) */
            /* Need scratch for attention output */
            uint64_t attn_out_off = gpu_scratch_alloc(s, embd_bytes);
            q4_gpu_tensor *attn_out = gpu_scratch_view(s, attn_out_off, embd_bytes);

            if (q4_gpu_flash_attn_tensor(attn_out, model_map, model_size,
                    q4_gpu_tensor_offset(q_t), q4_gpu_tensor_offset(k_t), q4_gpu_tensor_offset(v_t),
                    s->gpu_kv_k, s->gpu_kv_v,
                    Q4_N_HEAD, Q4_N_HEAD_KV, Q4_HEAD_DIM,
                    1, pos, s->kv_len + 1, e->logit_softcap) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u flash attention failed", il);
                return -1;
            }

            /* Increment KV cache length */
            s->kv_len++;

            /* Post-norm on attention output: normed = rms_norm(attn_out) */
            if (q4_gpu_rms_norm_weight_rows_tensor(s->gpu_normed, attn_out,
                    model_map, model_size, l->post_attn_norm->abs_offset, n_embd, 1, Q4_RMS_EPS) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u post-norm failed", il);
                return -1;
            }

            /* Residual: hidden = attn_out + residual.
             * Optimize: residual += attn_out (in-place), then copy to hidden.
             * Saves 1 dispatch vs copy(attn_out→hidden) + add + copy(hidden→residual). */
            if (q4_gpu_residual_add_tensor(s->gpu_residual, attn_out, n_embd) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u attn residual add failed", il);
                return -1;
            }
            if (q4_gpu_tensor_copy(s->gpu_hidden, 0, s->gpu_residual, 0, embd_bytes) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u attn residual copy failed", il);
                return -1;
            }

            /* Skip the shared post-norm/residual section below — already done above */
            goto skip_post_attn_norm;
        }

        /* --- Post-attention/post-SSM norm + residual --- */
        {
            uint64_t post_norm_w_offset = l->post_attn_norm->abs_offset;
            if (q4_gpu_rms_norm_weight_rows_tensor(s->gpu_normed, s->gpu_hidden,
                    model_map, model_size, post_norm_w_offset, n_embd, 1, Q4_RMS_EPS) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u post-norm failed", il);
                return -1;
            }

            /* Residual add: hidden = normed + residual */
            /* Optimize: residual += normed (in-place), then copy to hidden.
             * Saves 1 dispatch vs copy + add + copy. */
            if (q4_gpu_residual_add_tensor(s->gpu_residual, s->gpu_normed, n_embd) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u residual add failed", il);
                return -1;
            }
            if (q4_gpu_tensor_copy(s->gpu_hidden, 0, s->gpu_residual, 0, embd_bytes) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u residual copy failed", il);
                return -1;
            }
        }

skip_post_attn_norm:
        /* --- FFN --- */
        /* Norm again for FFN input */
        if (q4_gpu_rms_norm_weight_rows_tensor(s->gpu_normed, s->gpu_hidden,
                model_map, model_size, l->post_attn_norm->abs_offset, n_embd, 1, Q4_RMS_EPS) != 0) {
            if (err && errlen > 0) snprintf(err, errlen, "layer %u FFN pre-norm failed", il);
            return -1;
        }

        /* FFN: gate = normed @ ffn_gate (Q4_K), up = normed @ ffn_up (Q4_K)
         * mid = SiLU(clamp(gate)) * up, then ffn_out = mid @ ffn_down
         * Since shared kernel only supports Q8_0, do separate matmuls + elementwise. */

        /* Check if both FFN gate and up are Q4_K for fused matmul */
        q4_gpu_tensor *gate_t, *up_t;
        if (l->ffn_gate->type == Q4_TENSOR_Q4_K && l->ffn_up->type == Q4_TENSOR_Q4_K) {
            /* Fused gate+up: 2 matmuls from same input in one dispatch */
            const uint64_t fused_ffn_bytes = (uint64_t)Q4_N_FFN * 2 * sizeof(float);
            uint64_t fused_ffn_off = gpu_scratch_alloc(s, fused_ffn_bytes);
            q4_gpu_tensor *fused_ffn_t = gpu_scratch_view(s, fused_ffn_off, fused_ffn_bytes);

            const uint64_t ffn_weights[4] = {
                l->ffn_gate->abs_offset, l->ffn_up->abs_offset, 0, 0,
            };
            const uint32_t ffn_dims[4] = { Q4_N_FFN, Q4_N_FFN, 0, 0 };
            if (q4_gpu_matmul_q4_k_fused4_tensor(fused_ffn_t, model_map, model_size,
                    ffn_weights, ffn_dims, n_embd, s->gpu_normed) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u FFN fused gate/up failed", il);
                return -1;
            }

            gate_t = q4_gpu_tensor_view(fused_ffn_t, 0, ffn_bytes);
            up_t = q4_gpu_tensor_view(fused_ffn_t, (uint64_t)Q4_N_FFN * sizeof(float), ffn_bytes);
        } else {
            /* Fallback: separate matmuls for non-Q4_K weights */
            uint64_t gate_off = gpu_scratch_alloc(s, ffn_bytes);
            uint64_t up_off = gpu_scratch_alloc(s, ffn_bytes);
            gate_t = gpu_scratch_view(s, gate_off, ffn_bytes);
            up_t = gpu_scratch_view(s, up_off, ffn_bytes);

            if (q4_gpu_matmul_any_tensor(gate_t, model_map, model_size, l->ffn_gate->abs_offset,
                    n_embd, Q4_N_FFN, l->ffn_gate->type, s->gpu_normed, 1) != 0 ||
                q4_gpu_matmul_any_tensor(up_t, model_map, model_size, l->ffn_up->abs_offset,
                    n_embd, Q4_N_FFN, l->ffn_up->type, s->gpu_normed, 1) != 0) {
                if (err && errlen > 0) snprintf(err, errlen, "layer %u FFN gate/up failed", il);
                return -1;
            }
        }

        /* Fused SiLU+mul on GPU: gate_t = SiLU(clamp(gate_t)) * up_t */
        if (q4_gpu_silu_clamped_mul_tensor(gate_t, gate_t, up_t, Q4_N_FFN, 10.0f) != 0) {
            if (err && errlen > 0) snprintf(err, errlen, "layer %u FFN SiLU+mul failed", il);
            return -1;
        }

        /* ffn_down: mid -> ffn_out */
        uint64_t ffn_out_off = gpu_scratch_alloc(s, embd_bytes);
        q4_gpu_tensor *ffn_out_t = gpu_scratch_view(s, ffn_out_off, embd_bytes);

        if (q4_gpu_matmul_any_tensor(ffn_out_t, model_map, model_size, l->ffn_down->abs_offset,
                Q4_N_FFN, n_embd, l->ffn_down->type, gate_t, 1) != 0) {
            if (err && errlen > 0) snprintf(err, errlen, "layer %u FFN down failed", il);
            return -1;
        }

        /* Residual add: hidden = ffn_out + hidden */

        if (q4_gpu_residual_add_tensor(s->gpu_hidden, ffn_out_t, n_embd) != 0) {
            if (err && errlen > 0) snprintf(err, errlen, "layer %u FFN residual failed", il);
            return -1;
        }

        /* Update residual = hidden */
        if (q4_gpu_tensor_copy(s->gpu_residual, 0, s->gpu_hidden, 0, embd_bytes) != 0) {
            if (err && errlen > 0) snprintf(err, errlen, "layer %u FFN residual copy failed", il);
            return -1;
        }

    }

    /* --- Final RMSNorm --- */
    if (q4_gpu_rms_norm_weight_rows_tensor(s->gpu_normed, s->gpu_hidden,
            e->model.map, e->model.size, e->weights.output_norm->abs_offset, n_embd, 1, Q4_RMS_EPS) != 0) {
        if (err && errlen > 0) snprintf(err, errlen, "final RMSNorm failed");
        return -1;
    }

    /* --- Output projection: normed -> logits --- */
    if (q4_gpu_matmul_any_tensor(s->gpu_logits, e->model.map, e->model.size,
            e->weights.output->abs_offset, n_embd, e->config.n_vocab,
            e->weights.output->type, s->gpu_normed, 1) != 0) {
        if (err && errlen > 0) snprintf(err, errlen, "output projection failed");
        return -1;
    }

    /* Flush commands */
    if (q4_gpu_flush_commands() != 0) {
        if (err && errlen > 0) snprintf(err, errlen, "failed to flush GPU commands");
        return -1;
    }

    /* Read logits back to CPU */
    uint64_t logits_bytes = (uint64_t)e->config.n_vocab * sizeof(float);
    if (q4_gpu_tensor_read(s->gpu_logits, 0, s->logits, logits_bytes) != 0) {
        if (err && errlen > 0) snprintf(err, errlen, "failed to read logits");
        return -1;
    }

    /* Debug: print some logits stats */
    float max_logit = -1e30f;
    int max_idx = 0;
    for (uint64_t i = 0; i < e->config.n_vocab; i++) {
        if (s->logits[i] > max_logit) {
            max_logit = s->logits[i];
            max_idx = (int)i;
        }
    }

    return 0;
}
#endif /* Q4_NO_GPU */

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
        /* GPU path: process tokens one by one on GPU */
        s->n_tokens = 0;
        s->kv_len = 0;

        for (int i = 0; i < prompt->len; i++) {
            s->tokens[s->n_tokens++] = prompt->v[i];
            if (q4_gpu_forward(e, s, prompt->v[i], (uint32_t)i, err, errlen) != 0) {
                return -1;
            }
        }
        return 0;
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
        /* GPU path: forward pass on GPU */
        uint32_t pos = s->kv_len;
        if (q4_gpu_forward(e, s, token, pos, err, errlen) != 0) {
            return -1;
        }
        if (s->n_tokens < s->ctx_size) {
            s->tokens[s->n_tokens++] = token;
        }
        return 0;
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
        fprintf(stderr, "GEN_DBG: argmax returned token=%d\n", token);
        if (token < 0) {
            fprintf(stderr, "q4: argmax failed: %s\n", err);
            break;
        }

        if (emit) {
            fprintf(stderr, "GEN_DBG: calling emit with token=%d\n", token);
            emit(emit_ud, token);
        }

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
