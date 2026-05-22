#include "q4.h"
#include "linenoise.h"

/* q4 CLI - interactive and one-shot inference for Qwen3.6-27B.
 *
 * Usage:
 *   q4                          Start interactive REPL
 *   q4 -p "Hello"               One-shot generation
 *   q4 -m model.gguf -p "Hi"    Load specific model, generate
 *   q4 --inspect                Load model and print summary only */

#include <ctype.h>
#include <errno.h>
#include <limits.h>
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
 * Command-line options.
 * ========================================================================= */

typedef struct {
    const char *model_path;
    q4_backend backend;
    int n_threads;
    bool warm_weights;
    bool inspect;
    bool dump_tokens;

    /* Generation */
    const char *prompt;
    char *prompt_owned;    /* if read from file */
    const char *system;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    int top_k;
    uint64_t seed;
} cli_config;

/* =========================================================================
 * Utilities.
 * ========================================================================= */

static double cli_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "q4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "q4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "q4: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static q4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return Q4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return Q4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return Q4_BACKEND_CPU;
    fprintf(stderr, "q4: invalid backend: %s\n", s);
    fprintf(stderr, "q4: valid backends are: metal, cuda, cpu\n");
    exit(2);
}

static q4_backend default_backend(void) {
#ifdef Q4_NO_GPU
    return Q4_BACKEND_CPU;
#elif defined(__APPLE__)
    return Q4_BACKEND_METAL;
#else
    return Q4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long len = ftell(fp);
    if (len < 0) { fclose(fp); return NULL; }
    rewind(fp);
    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t n = fread(buf, 1, (size_t)len, fp);
    fclose(fp);
    if (n != (size_t)len) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

/* =========================================================================
 * Usage.
 * ========================================================================= */

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: q4 [(-p PROMPT | --prompt-file FILE)] [options]\n"
        "\n"
        "Invocation modes:\n"
        "  q4\n"
        "      Start the interactive chat prompt with a session backend: q4>\n"
        "  q4 -p TEXT\n"
        "      Run one prompt and exit.\n"
        "  q4 --prompt-file FILE\n"
        "      Run one prompt read from FILE and exit.\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: model.gguf\n"
        "  -c, --ctx N\n"
        "      Context size allocated for the session. Default: 32768\n"
        "  --metal\n"
        "      Use the Metal backend. This is the normal fast path on macOS.\n"
        "  --cuda\n"
        "      Use the CUDA backend.\n"
        "  --cpu\n"
        "      Use the CPU reference/debug backend.\n"
        "  --backend NAME\n"
        "      Select backend explicitly: metal, cuda, or cpu.\n"
        "  -t, --threads N\n"
        "      CPU helper threads. Default: hardware concurrency\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before generation. Slower startup, fewer first-use stalls.\n"
        "\n"
        "Prompt and generation:\n"
        "  -p, --prompt TEXT\n"
        "      Prompt to generate from.\n"
        "  --prompt-file FILE\n"
        "      Read the prompt text from FILE.\n"
        "  --system TEXT\n"
        "      System prompt. Empty string disables the default.\n"
        "  -n, --tokens N\n"
        "      Maximum tokens to generate. Default: 50000\n"
        "  --temp F\n"
        "      Sampling temperature. 0 is greedy/deterministic. Default: 0.6\n"
        "  --top-p F\n"
        "      Nucleus sampling probability. Default: 0.95\n"
        "  --min-p F\n"
        "      Keep tokens scoring at least F times the top token. Default: 0.01\n"
        "  --top-k N\n"
        "      Keep top-K tokens. 0 = no limit. Default: 20\n"
        "  --seed N\n"
        "      Sampling seed for reproducible non-greedy runs.\n"
        "\n"
        "Interactive commands:\n"
        "  /help\n"
        "      Show interactive commands.\n"
        "  /ctx N\n"
        "      Recreate the session with a new context size.\n"
        "  /read FILE\n"
        "      Read a prompt from FILE and run it as the next user message.\n"
        "  /quit, /exit\n"
        "      Leave the interactive prompt.\n"
        "  Ctrl+C\n"
        "      Stop the current generation and return to q4>.\n"
        "\n"
        "Diagnostics:\n"
        "  --inspect\n"
        "      Load the model and print a summary only.\n"
        "  --dump-tokens\n"
        "      Tokenize -p/--prompt-file and exit without inference.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
}

/* =========================================================================
 * Chat template for Qwen3.6.
 *
 * Qwen3.6 uses a simple chat template:
 *   <｜begin▁of▁sentence｜>{system_text}User: {user}\nAssistant:
 *
 * For subsequent turns:
 *   {generated_text}<|endoftext|>User: {user}\nAssistant:
 * ========================================================================= */

/* Check if a prompt string already starts with the BOS marker. */
static bool is_rendered_chat_prompt(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

/* Build a minimal chat prompt.  The engine's tokenizer handles the raw string. */
static void build_chat_prompt(q4_engine *engine, const cli_config *cfg,
                               const char *user_text, char *buf, size_t bufsize) {
    (void)engine;
    if (cfg->system && cfg->system[0]) {
        snprintf(buf, bufsize, "<｜begin▁of▁sentence｜>%s\n\nUser: %s\nAssistant:",
                 cfg->system, user_text);
    } else {
        snprintf(buf, bufsize, "<｜begin▁of▁sentence｜>User: %s\nAssistant:", user_text);
    }
}

static void build_chat_continuation(q4_engine *engine, const char *user_text,
                                     char *buf, size_t bufsize) {
    (void)engine;
    snprintf(buf, bufsize, "User: %s\nAssistant:", user_text);
}

/* =========================================================================
 * Token printer.
 * ========================================================================= */

static void print_generated_token(void *ud, int token) {
    q4_engine *e = ud;
    size_t len = 0;
    char *text = q4_token_text(e, token, &len);
    if (text) {
        fwrite(text, 1, len, stdout);
        fflush(stdout);
        free(text);
    }
}

/* =========================================================================
 * One-shot generation.
 * ========================================================================= */

static int run_one_shot(q4_engine *engine, const cli_config *cfg) {
    char prompt_buf[65536];

    if (is_rendered_chat_prompt(cfg->prompt)) {
        strncpy(prompt_buf, cfg->prompt, sizeof(prompt_buf) - 1);
        prompt_buf[sizeof(prompt_buf) - 1] = '\0';
    } else {
        build_chat_prompt(engine, cfg, cfg->prompt, prompt_buf, sizeof(prompt_buf));
    }

    q4_tokens *tokens = NULL;
    if (q4_engine_tokenize(engine, prompt_buf, (int)strlen(prompt_buf), &tokens) != 0) {
        fprintf(stderr, "q4: failed to tokenize prompt\n");
        return 1;
    }

    if (cfg->dump_tokens) {
        printf("Prompt tokens: %d\n", tokens->len);
        for (int i = 0; i < tokens->len; i++) {
            size_t tlen = 0;
            char *t = q4_token_text(engine, tokens->v[i], &tlen);
            if (t) {
                printf("  [%d] %d: '", i, tokens->v[i]);
                for (size_t j = 0; j < tlen; j++) {
                    if (isprint((unsigned char)t[j])) putchar(t[j]);
                    else printf("\\x%02x", (unsigned char)t[j]);
                }
                printf("'\n");
                free(t);
            }
        }
        q4_tokens_free(tokens);
        return 0;
    }

    /* Timing */
    const double t0 = cli_now_sec();

    int rc = q4_engine_generate_argmax(engine, tokens, cfg->n_predict,
                                        cfg->ctx_size,
                                        print_generated_token,
                                        NULL, engine);
    const double t1 = cli_now_sec();

    q4_tokens_free(tokens);

    /* Print stats */
    const double elapsed = t1 - t0;
    if (elapsed > 0.001) {
        fprintf(stderr, "\nq4: generated %d tokens in %.2f s (%.1f tok/s)\n",
                cfg->n_predict, elapsed, cfg->n_predict / elapsed);
    }

    return rc;
}

/* =========================================================================
 * Interactive REPL with session-based KV cache reuse.
 * ========================================================================= */

static volatile sig_atomic_t g_interrupted;

static void sigint_handler(int sig) {
    (void)sig;
    g_interrupted = 1;
}

static bool interrupt_requested(void) {
    return g_interrupted != 0;
}

typedef struct {
    q4_session *session;
    q4_tokens *transcript;   /* accumulated token history */
    int ctx_size;
    int bos_token;
} repl_state;


static int repl_init(q4_engine *engine, repl_state *rs, const cli_config *cfg) {
    memset(rs, 0, sizeof(*rs));
    rs->ctx_size = cfg->ctx_size;
    rs->bos_token = q4_token_eos(engine); /* will be corrected below */

    /* Create initial transcript with BOS + system prompt */
    rs->transcript = calloc(1, sizeof(q4_tokens));
    if (!rs->transcript) return -1;

    char prompt_buf[65536];
    if (cfg->system && cfg->system[0]) {
        snprintf(prompt_buf, sizeof(prompt_buf), "<｜begin▁of▁sentence｜>%s", cfg->system);
    } else {
        snprintf(prompt_buf, sizeof(prompt_buf), "<｜begin▁of▁sentence｜>");
    }

    if (q4_engine_tokenize(engine, prompt_buf, (int)strlen(prompt_buf), &rs->transcript) != 0) {
        q4_tokens_free(rs->transcript);
        rs->transcript = NULL;
        return -1;
    }

    /* Create session */
    if (q4_session_create(&rs->session, engine, cfg->ctx_size) != 0) {
        fprintf(stderr, "q4: failed to create session\n");
        q4_tokens_free(rs->transcript);
        rs->transcript = NULL;
        return -1;
    }

    return 0;
}

static void repl_free(repl_state *rs) {
    if (!rs) return;
    q4_session_free(rs->session);
    q4_tokens_free(rs->transcript);
    memset(rs, 0, sizeof(*rs));
}

static int repl_recreate_session(q4_engine *engine, repl_state *rs, int ctx_size) {
    q4_session_free(rs->session);
    rs->session = NULL;
    rs->ctx_size = ctx_size;
    return q4_session_create(&rs->session, engine, ctx_size);
}

/* Compute a naive common prefix between the current session tokens and the transcript.
 * This is a simplified version - production would use a smarter algorithm. */
static int compute_common_prefix(q4_session *session, const q4_tokens *transcript) {
    (void)session; (void)transcript;
    /* For simplicity, return 0 and force full re-sync.
     * A production implementation would track session tokens separately. */
    return 0;
}

static int run_chat_turn(q4_engine *engine, cli_config *cfg, repl_state *rs, const char *user_text) {
    if (!rs->session || !rs->transcript) {
        fprintf(stderr, "q4: no active session\n");
        return 1;
    }

    /* Append user message to transcript */
    char cont_buf[65536];
    build_chat_continuation(engine, user_text, cont_buf, sizeof(cont_buf));

    q4_tokens *new_tokens = NULL;
    if (q4_engine_tokenize(engine, cont_buf, (int)strlen(cont_buf), &new_tokens) != 0) {
        fprintf(stderr, "q4: failed to tokenize user input\n");
        return 1;
    }

    /* Extend transcript */
    for (int i = 0; i < new_tokens->len; i++) {
        q4_tokens_push(rs->transcript, new_tokens->v[i]);
    }
    q4_tokens_free(new_tokens);

    /* Save rollback position in case of error */
    int rollback_len = rs->transcript->len;

    /* Sync session with updated transcript */
    int common = compute_common_prefix(rs->session, rs->transcript);
    int suffix = rs->transcript->len - common;

    fprintf(stderr, "q4: processing %d new tokens (common prefix=%d)\n", suffix, common);

    char err[256];
    const double t_prefill0 = cli_now_sec();
    if (q4_session_sync(rs->session, rs->transcript, err, sizeof(err)) != 0) {
        fprintf(stderr, "q4: prompt processing failed: %s\n", err);
        rs->transcript->len = rollback_len;
        return 1;
    }
    const double t_prefill1 = cli_now_sec();
    const double t_prefill = t_prefill1 - t_prefill0;
    if (t_prefill > 0.001) {
        fprintf(stderr, "q4: prefill %.2f s (%.1f tok/s)\n", t_prefill, suffix / t_prefill);
    }

    /* Generate tokens */
    int max_tokens = cfg->n_predict;
    int room = q4_session_ctx(rs->session) - q4_session_pos(rs->session);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    uint64_t rng = cfg->seed ? cfg->seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());

    int generated = 0;
    const double t_decode0 = cli_now_sec();

    while (generated < max_tokens && !interrupt_requested()) {
        int token;
        if (cfg->temperature < 1.0e-6f) {
            token = q4_session_argmax(rs->session, err, sizeof(err));
        } else {
            token = q4_session_sample(rs->session, cfg->temperature, cfg->top_k,
                                       cfg->top_p, cfg->min_p, &rng,
                                       err, sizeof(err));
        }
        if (token < 0) {
            fprintf(stderr, "q4: sampling failed: %s\n", err);
            break;
        }
        if (token == q4_token_eos(engine)) break;

        /* Print token */
        size_t tlen = 0;
        char *ttext = q4_token_text(engine, token, &tlen);
        if (ttext) {
            fwrite(ttext, 1, tlen, stdout);
            fflush(stdout);
            free(ttext);
        }

        /* Add to transcript and evaluate */
        q4_tokens_push(rs->transcript, token);
        if (q4_session_eval(rs->session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "\nq4: eval failed: %s\n", err);
            break;
        }
        generated++;
    }

    const double t_decode1 = cli_now_sec();
    const double t_decode = t_decode1 - t_decode0;

    printf("\n");
    fflush(stdout);

    if (t_decode > 0.001 && generated > 0) {
        fprintf(stderr, "q4: generated %d tokens in %.2f s (%.1f tok/s)\n",
                generated, t_decode, generated / t_decode);
    }
    if (interrupt_requested()) {
        g_interrupted = 0;
        fprintf(stderr, "q4: generation interrupted\n");
    }

    return 0;
}

static int run_repl(q4_engine *engine, const cli_config *cfg) {
    repl_state rs;
    if (repl_init(engine, &rs, cfg) != 0) {
        fprintf(stderr, "q4: failed to initialize REPL\n");
        return 1;
    }

    struct sigaction sa_new = {0};
    struct sigaction sa_old;
    sa_new.sa_handler = sigint_handler;
    sigemptyset(&sa_new.sa_mask);
    bool sigint_installed = (sigaction(SIGINT, &sa_new, &sa_old) == 0);

    int rc = 0;

    /* Show welcome */
    fprintf(stderr, "q4: interactive chat (model: %s, backend: %s, ctx: %d)\n",
            cfg->model_path ? cfg->model_path : "(none)",
            q4_backend_name(cfg->backend), cfg->ctx_size);
    fprintf(stderr, "q4: type /help for commands\n");

    /* History file */
    char *home = getenv("HOME");
    char hist_path[512];
    if (home) {
        snprintf(hist_path, sizeof(hist_path), "%s/.q4_history", home);
        linenoiseHistoryLoad(hist_path);
    }

    while (1) {
        char *line = linenoise("q4> ");
        if (!line) break;

        if (line[0] == '\0') {
            linenoiseFree(line);
            continue;
        }

        linenoiseHistoryAdd(line);

        /* Strip leading whitespace */
        const char *cmd = line;
        while (*cmd && isspace((unsigned char)*cmd)) cmd++;

        if (!*cmd) {
            linenoiseFree(line);
            continue;
        }

        if (strcmp(cmd, "/quit") == 0 || strcmp(cmd, "/exit") == 0) {
            linenoiseFree(line);
            break;
        } else if (strcmp(cmd, "/help") == 0) {
            fprintf(stderr,
                "q4 commands:\n"
                "  /help       Show this help\n"
                "  /quit       Exit the REPL\n"
                "  /exit       Exit the REPL\n"
                "  /ctx N      Change context size\n"
                "  /read FILE  Read prompt from file\n"
                "\n"
                "Just type your message to chat.\n");
        } else if (strncmp(cmd, "/ctx ", 5) == 0) {
            int new_ctx = parse_int(cmd + 5, "/ctx");
            if (repl_recreate_session(engine, &rs, new_ctx) != 0) {
                fprintf(stderr, "q4: failed to recreate session with ctx=%d\n", new_ctx);
            } else {
                rs.ctx_size = new_ctx;
                fprintf(stderr, "q4: context size changed to %d\n", new_ctx);
            }
        } else if (strncmp(cmd, "/read ", 6) == 0) {
            const char *fpath = cmd + 6;
            char *text = read_file(fpath);
            if (!text) {
                fprintf(stderr, "q4: failed to read file: %s\n", fpath);
            } else {
                rc = run_chat_turn(engine, (cli_config *)cfg, &rs, text);
                free(text);
            }
        } else if (cmd[0] == '/') {
            fprintf(stderr, "q4: unknown command: %s\n", cmd);
            fprintf(stderr, "q4: type /help for commands\n");
        } else {
            rc = run_chat_turn(engine, (cli_config *)cfg, &rs, cmd);
        }

        linenoiseFree(line);
    }

    if (sigint_installed) sigaction(SIGINT, &sa_old, NULL);
    repl_free(&rs);

    if (home) {
        snprintf(hist_path, sizeof(hist_path), "%s/.q4_history", home);
        linenoiseHistorySave(hist_path);
    }

    return rc;
}

/* =========================================================================
 * Option parsing.
 * ========================================================================= */

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "q4: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static cli_config parse_options(int argc, char **argv) {
    cli_config c = {
        .model_path = "model.gguf",
        .backend = default_backend(),
        .system = "You are a helpful assistant",
        .n_predict = 50000,
        .ctx_size = 32768,
        .temperature = Q4_DEFAULT_TEMPERATURE,
        .top_p = Q4_DEFAULT_TOP_P,
        .min_p = Q4_DEFAULT_MIN_P,
        .top_k = Q4_DEFAULT_TOP_K,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.prompt) {
                fprintf(stderr, "q4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.prompt) {
                fprintf(stderr, "q4: specify only one prompt source\n");
                exit(2);
            }
            c.prompt_owned = read_file(need_arg(&i, argc, argv, arg));
            if (!c.prompt_owned) {
                fprintf(stderr, "q4: failed to read prompt file\n");
                exit(2);
            }
            c.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--top-k")) {
            c.top_k = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--seed")) {
            c.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = Q4_BACKEND_CPU;
        } else if (!strcmp(arg, "--metal")) {
            c.backend = Q4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = Q4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--inspect")) {
            c.inspect = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--dump-tokens")) {
            c.dump_tokens = true;
        } else {
            fprintf(stderr, "q4: unknown option: %s\n", arg);
            fprintf(stderr, "q4: use --help for usage\n");
            exit(2);
        }
    }

    return c;
}

/* =========================================================================
 * Log context memory.
 * ========================================================================= */

static void log_context_memory(q4_backend backend, int ctx_size) {
    q4_context_memory m = q4_context_memory_estimate(backend, ctx_size);
    fprintf(stderr,
            "q4: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_cap=%u, raw_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            q4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap);
}

/* =========================================================================
 * Main.
 * ========================================================================= */

int main(int argc, char **argv) {
    cli_config cfg = parse_options(argc, argv);

    /* Open engine */
    q4_engine *engine = NULL;
    q4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.n_threads,
        .warm_weights = cfg.warm_weights,
    };

    if (q4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "q4: failed to open engine (model: %s)\n", cfg.model_path);
        free(cfg.prompt_owned);
        return 1;
    }

    if (cfg.inspect) {
        q4_engine_summary(engine);
        q4_engine_close(engine);
        free(cfg.prompt_owned);
        return 0;
    }

    /* Dump tokens mode */
    if (cfg.dump_tokens) {
        if (!cfg.prompt) {
            fprintf(stderr, "q4: --dump-tokens requires -p or --prompt-file\n");
            q4_engine_close(engine);
            free(cfg.prompt_owned);
            return 1;
        }
        log_context_memory(cfg.backend, cfg.ctx_size);
        int rc = run_one_shot(engine, &cfg);
        q4_engine_close(engine);
        free(cfg.prompt_owned);
        return rc;
    }

    /* No prompt -> REPL */
    if (!cfg.prompt) {
        log_context_memory(cfg.backend, cfg.ctx_size);
        int rc = run_repl(engine, &cfg);
        q4_engine_close(engine);
        free(cfg.prompt_owned);
        return rc;
    }

    /* One-shot generation */
    log_context_memory(cfg.backend, cfg.ctx_size);
    int rc = run_one_shot(engine, &cfg);
    q4_engine_close(engine);
    free(cfg.prompt_owned);
    return rc;
}
