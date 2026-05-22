/* q4 benchmark - measures prefill/decode throughput and memory usage.
 *
 * Usage:
 *   q4-bench -m model.gguf                    Run all benchmarks
 *   q4-bench -m model.gguf --prefill N        Prefill benchmark with N tokens
 *   q4-bench -m model.gguf --decode N         Decode benchmark with N steps
 *   q4-bench -m model.gguf --ctx N            Context size
 *
 * Output is printed as TSV for easy parsing and comparison with llama.cpp. */

#include "q4.h"

#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* =========================================================================
 * Utilities.
 * ========================================================================= */

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static uint64_t bench_rss_bytes(void) {
#if defined(__APPLE__)
    /* macOS: read from proc */
    FILE *fp = popen("ps -o rss= -p $$ 2>/dev/null", "r");
    if (!fp) return 0;
    uint64_t rss_kb = 0;
    if (fscanf(fp, "%llu", (unsigned long long *)&rss_kb) == 1) {
        pclose(fp);
        return rss_kb * 1024;
    }
    pclose(fp);
    return 0;
#elif defined(__linux__)
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return 0;
    char line[256];
    uint64_t rss_kb = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%llu", (unsigned long long *)&rss_kb);
            break;
        }
    }
    fclose(fp);
    return rss_kb * 1024;
#else
    return 0;
#endif
}

/* =========================================================================
 * Benchmark options.
 * ========================================================================= */

typedef struct {
    const char *model_path;
    q4_backend backend;
    int ctx_size;
    int n_threads;
    bool warm_weights;

    /* Prefill benchmark */
    int prefill_tokens;

    /* Decode benchmark */
    int decode_steps;
} bench_config;

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: q4-bench [options]\n"
        "\n"
        "  -m, --model FILE       GGUF model path. Default: model.gguf\n"
        "  -c, --ctx N            Context size. Default: 32768\n"
        "  -t, --threads N        CPU helper threads\n"
        "  --cpu                  CPU backend\n"
        "  --metal                Metal backend\n"
        "  --warm-weights         Touch mapped weights before benchmark\n"
        "  --prefill N            Number of tokens for prefill benchmark. Default: 512\n"
        "  --decode N             Number of decode steps. Default: 128\n"
        "  -h, --help             Show this help\n");
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "model.gguf",
        .backend = Q4_BACKEND_CPU,
        .ctx_size = 32768,
        .prefill_tokens = 512,
        .decode_steps = 128,
    };

#ifdef __APPLE__
    c.backend = Q4_BACKEND_METAL;
#endif

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            if (i + 1 >= argc) { fprintf(stderr, "q4-bench: missing value for %s\n", arg); exit(2); }
            c.model_path = argv[++i];
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            if (i + 1 >= argc) { fprintf(stderr, "q4-bench: missing value for %s\n", arg); exit(2); }
            c.ctx_size = atoi(argv[++i]);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            if (i + 1 >= argc) { fprintf(stderr, "q4-bench: missing value for %s\n", arg); exit(2); }
            c.n_threads = atoi(argv[++i]);
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = Q4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.backend = Q4_BACKEND_METAL;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--prefill")) {
            if (i + 1 >= argc) { fprintf(stderr, "q4-bench: missing value for %s\n", arg); exit(2); }
            c.prefill_tokens = atoi(argv[++i]);
        } else if (!strcmp(arg, "--decode")) {
            if (i + 1 >= argc) { fprintf(stderr, "q4-bench: missing value for %s\n", arg); exit(2); }
            c.decode_steps = atoi(argv[++i]);
        } else {
            fprintf(stderr, "q4-bench: unknown option: %s\n", arg);
            exit(2);
        }
    }

    return c;
}

/* =========================================================================
 * Generate dummy prompt tokens (just sequential token IDs, no tokenization).
 * ========================================================================= */

static q4_tokens *make_dummy_tokens(int n) {
    q4_tokens *t = calloc(1, sizeof(q4_tokens));
    if (!t) return NULL;
    t->cap = n;
    t->v = malloc((size_t)n * sizeof(int));
    if (!t->v) { free(t); return NULL; }
    /* Use simple repeating pattern: tokens 1..n-1 (avoid 0 which is often padding) */
    for (int i = 0; i < n; i++) {
        t->v[i] = (i % 1000) + 1;
    }
    t->len = n;
    return t;
}

/* =========================================================================
 * Benchmarks.
 * ========================================================================= */

static int bench_load(const bench_config *cfg) {
    fprintf(stderr, "q4-bench: loading model: %s\n", cfg->model_path);

    const double t0 = bench_now_sec();
    uint64_t rss0 = bench_rss_bytes();

    q4_engine *engine = NULL;
    q4_engine_options opt = {
        .model_path = cfg->model_path,
        .backend = cfg->backend,
        .n_threads = cfg->n_threads,
        .warm_weights = cfg->warm_weights,
    };

    if (q4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "q4-bench: FAILED to load model\n");
        return 1;
    }

    const double t1 = bench_now_sec();
    uint64_t rss1 = bench_rss_bytes();

    q4_context_memory mem = q4_context_memory_estimate(cfg->backend, cfg->ctx_size);

    printf("# benchmark: load\n");
    printf("load_time_ms\t%.1f\n", (t1 - t0) * 1000.0);
    printf("load_rss_mb\t%.1f\n", (double)(rss1 - rss0) / (1024.0 * 1024.0));
    printf("model_total_mb\t%.1f\n", (double)mem.total_bytes / (1024.0 * 1024.0));
    printf("model_raw_mb\t%.1f\n", (double)mem.raw_bytes / (1024.0 * 1024.0));
    printf("model_scratch_mb\t%.1f\n", (double)mem.scratch_bytes / (1024.0 * 1024.0));
    printf("context_prefill_cap\t%u\n", mem.prefill_cap);
    printf("context_raw_kv_rows\t%u\n", mem.raw_cap);

    q4_engine_summary(engine);

    /* Print summary to stderr */
    fprintf(stderr, "q4-bench: load %.1f ms, RSS %.1f MiB (model)\n",
            (t1 - t0) * 1000.0, (double)mem.total_bytes / (1024.0 * 1024.0));

    q4_engine_close(engine);
    return 0;
}

static int bench_prefill(const bench_config *cfg) {
    fprintf(stderr, "q4-bench: prefill benchmark (%d tokens)\n", cfg->prefill_tokens);

    q4_engine *engine = NULL;
    q4_engine_options opt = {
        .model_path = cfg->model_path,
        .backend = cfg->backend,
        .n_threads = cfg->n_threads,
        .warm_weights = cfg->warm_weights,
    };

    if (q4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "q4-bench: FAILED to load model\n");
        return 1;
    }

    q4_session *session = NULL;
    if (q4_session_create(&session, engine, cfg->ctx_size) != 0) {
        fprintf(stderr, "q4-bench: FAILED to create session\n");
        q4_engine_close(engine);
        return 1;
    }

    /* Create dummy prompt tokens */
    q4_tokens *prompt = make_dummy_tokens(cfg->prefill_tokens);
    if (!prompt) {
        fprintf(stderr, "q4-bench: FAILED to allocate tokens\n");
        q4_session_free(session);
        q4_engine_close(engine);
        return 1;
    }

    /* Run prefill */
    char err[256];
    const double t0 = bench_now_sec();
    int rc = q4_session_sync(session, prompt, err, sizeof(err));
    const double t1 = bench_now_sec();

    const double elapsed = t1 - t0;
    const double tok_per_sec = elapsed > 0 ? cfg->prefill_tokens / elapsed : 0;
    const double ms_per_token = cfg->prefill_tokens > 0 ? (elapsed * 1000.0) / cfg->prefill_tokens : 0;

    printf("# benchmark: prefill\n");
    printf("prefill_tokens\t%d\n", cfg->prefill_tokens);
    printf("prefill_time_ms\t%.1f\n", elapsed * 1000.0);
    printf("prefill_tok_per_sec\t%.1f\n", tok_per_sec);
    printf("prefill_ms_per_token\t%.2f\n", ms_per_token);
    printf("prefill_session_pos\t%d\n", q4_session_pos(session));

    fprintf(stderr, "q4-bench: prefill %d tokens in %.1f ms (%.1f tok/s, %.2f ms/tok)\n",
            cfg->prefill_tokens, elapsed * 1000.0, tok_per_sec, ms_per_token);

    q4_tokens_free(prompt);
    q4_session_free(session);
    q4_engine_close(engine);

    return rc != 0 ? 1 : 0;
}

static int bench_decode(const bench_config *cfg) {
    fprintf(stderr, "q4-bench: decode benchmark (%d steps)\n", cfg->decode_steps);

    q4_engine *engine = NULL;
    q4_engine_options opt = {
        .model_path = cfg->model_path,
        .backend = cfg->backend,
        .n_threads = cfg->n_threads,
        .warm_weights = cfg->warm_weights,
    };

    if (q4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "q4-bench: FAILED to load model\n");
        return 1;
    }

    q4_session *session = NULL;
    if (q4_session_create(&session, engine, cfg->ctx_size) != 0) {
        fprintf(stderr, "q4-bench: FAILED to create session\n");
        q4_engine_close(engine);
        return 1;
    }

    /* Seed with a short prompt */
    q4_tokens prompt = { .cap = 4, .len = 4, .v = (int[]){1, 2, 3, 4} };
    char err[256];
    if (q4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "q4-bench: FAILED to sync prompt: %s\n", err);
        q4_session_free(session);
        q4_engine_close(engine);
        return 1;
    }

    /* Decode loop */
    const double t0 = bench_now_sec();
    int generated = 0;

    for (int i = 0; i < cfg->decode_steps; i++) {
        int token = q4_session_argmax(session, err, sizeof(err));
        if (token < 0) {
            fprintf(stderr, "q4-bench: argmax failed at step %d: %s\n", i, err);
            break;
        }
        if (q4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "q4-bench: eval failed at step %d: %s\n", i, err);
            break;
        }
        generated++;
    }

    const double t1 = bench_now_sec();
    const double elapsed = t1 - t0;
    const double tok_per_sec = elapsed > 0 ? generated / elapsed : 0;
    const double ms_per_token = generated > 0 ? (elapsed * 1000.0) / generated : 0;

    printf("# benchmark: decode\n");
    printf("decode_steps\t%d\n", cfg->decode_steps);
    printf("decode_generated\t%d\n", generated);
    printf("decode_time_ms\t%.1f\n", elapsed * 1000.0);
    printf("decode_tok_per_sec\t%.1f\n", tok_per_sec);
    printf("decode_ms_per_token\t%.2f\n", ms_per_token);

    fprintf(stderr, "q4-bench: decode %d steps in %.1f ms (%.1f tok/s, %.2f ms/tok)\n",
            generated, elapsed * 1000.0, tok_per_sec, ms_per_token);

    q4_session_free(session);
    q4_engine_close(engine);

    return 0;
}

/* =========================================================================
 * Main.
 * ========================================================================= */

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    fprintf(stderr, "q4-bench: model=%s, backend=%s, ctx=%d, threads=%d\n",
            cfg.model_path,
            q4_backend_name(cfg.backend),
            cfg.ctx_size,
            cfg.n_threads);

    int rc = 0;

    rc |= bench_load(&cfg);
    if (rc != 0) return rc;

    rc |= bench_prefill(&cfg);
    rc |= bench_decode(&cfg);

    return rc;
}
