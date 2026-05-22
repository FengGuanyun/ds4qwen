/* =========================================================================
 * q4_metal.m - Metal GPU backend for Qwen3.6-27B.
 * =========================================================================
 *
 * Implements q4_gpu.h interface using Apple Metal.
 * Provides tensor lifecycle management, command batching, and kernel dispatch
 * for Q8_0 matmul, RMSNorm, RoPE, Flash Attention, and DeltaNet.
 */

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "q4_gpu.h"

/* =========================================================================
 * Metal Library and Device.
 * ========================================================================= */

static id<MTLDevice> g_device;
static id<MTLLibrary> g_library;
static id<MTLCommandQueue> g_queue;
static id<MTLBuffer> g_model_buffer;
static const void *g_model_map;
static uint64_t g_model_size;

typedef struct q4_gpu_tensor {
    id<MTLBuffer> buffer;
    uint64_t bytes;
    bool is_view;
} q4_gpu_tensor;

static NSString *metal_source(void) {
    /* Load pre-compiled metallib or compile from source at runtime. */
    /* Try bundled .metallib first */
    NSBundle *bundle = [NSBundle mainBundle];
    NSString *lib_path = [bundle pathForResource:@"q4_kernels" ofType:@"metallib"];
    if (lib_path) {
        NSError *err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithURL:[NSURL fileURLWithPath:lib_path] error:&err];
        if (lib) return lib_path;
    }

    /* Fall back to compiling from source */
    NSString *res_path = [bundle resourcePath];
    if (!res_path) res_path = @".";

    NSString *metal_dir = [res_path stringByAppendingPathComponent:@"metal"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:metal_dir]) {
        /* Try sibling directory for dev builds */
        metal_dir = [res_path stringByAppendingPathComponent:@"../metal"];
    }

    if (![[NSFileManager defaultManager] fileExistsAtPath:metal_dir]) {
        fprintf(stderr, "q4: cannot find metal kernel directory\n");
        return nil;
    }

    /* Collect all .metal files */
    NSMutableString *source = [NSMutableString string];
    NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:metal_dir error:nil];
    for (NSString *file in files) {
        if ([file hasSuffix:@".metal"]) {
            NSString *path = [metal_dir stringByAppendingPathComponent:file];
            NSString *content = [NSString stringWithContentsOfFile:path
                                                          encoding:NSUTF8StringEncoding error:nil];
            if (content) {
                [source appendFormat:@"\n// --- %@ ---\n", file];
                [source appendString:content];
            }
        }
    }

    return source;
}

int q4_gpu_init(void) {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "q4: Metal is not available\n");
        return -1;
    }

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "q4: failed to create Metal command queue\n");
        return -1;
    }

    NSString *source = metal_source();
    if (!source) {
        fprintf(stderr, "q4: failed to load Metal kernels\n");
        return -1;
    }

    /* Compile from source string */
    NSError *err = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:source options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "q4: failed to compile Metal kernels: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown error");
        return -1;
    }
    g_library = lib;

    return 0;
}

void q4_gpu_cleanup(void) {
    g_model_buffer = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
}

/* =========================================================================
 * Tensor Lifecycle.
 * ========================================================================= */

q4_gpu_tensor *q4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!buf) return NULL;

    q4_gpu_tensor *t = malloc(sizeof(*t));
    t->buffer = buf;
    t->bytes = bytes;
    t->is_view = false;
    return t;
}

q4_gpu_tensor *q4_gpu_tensor_view(const q4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset + bytes > base->bytes) return NULL;

    id<MTLBuffer> view = [g_device newBufferWithBuffer:base->buffer
                                                offset:offset
                                                length:bytes
                                               options:MTLResourceStorageModeShared];
    if (!view) return NULL;

    q4_gpu_tensor *t = malloc(sizeof(*t));
    t->buffer = view;
    t->bytes = bytes;
    t->is_view = true;
    return t;
}

void q4_gpu_tensor_free(q4_gpu_tensor *tensor) {
    if (!tensor) return;
    tensor->buffer = nil;
    free(tensor);
}

uint64_t q4_gpu_tensor_bytes(const q4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *q4_gpu_tensor_contents(q4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    return tensor->buffer.contents;
}

int q4_gpu_tensor_fill_f32(q4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count * sizeof(float) > tensor->bytes) return -1;
    float *p = tensor->buffer.contents;
    for (uint64_t i = 0; i < count; i++) p[i] = value;
    return 0;
}

int q4_gpu_tensor_write(q4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || offset + bytes > tensor->bytes) return -1;
    memcpy((uint8_t *)tensor->buffer.contents + offset, data, (size_t)bytes);
    return 0;
}

int q4_gpu_tensor_read(const q4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || offset + bytes > tensor->bytes) return -1;
    memcpy(data, (const uint8_t *)tensor->buffer.contents + offset, (size_t)bytes);
    return 0;
}

int q4_gpu_tensor_copy(q4_gpu_tensor *dst, uint64_t dst_offset,
                          const q4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src || dst_offset + bytes > dst->bytes ||
        src_offset + bytes > src->bytes) return -1;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src->buffer
            sourceOffset:src_offset
                toBuffer:dst->buffer
           destinationOffset:dst_offset
                       size:bytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return 0;
}

/* =========================================================================
 * Command Batching.
 * ========================================================================= */

static id<MTLCommandBuffer> g_cmd_buffer;
static id<MTLComputeCommandEncoder> g_encoder;

int q4_gpu_begin_commands(void) {
    if (g_encoder) {
        [g_encoder endEncoding];
        g_encoder = nil;
    }
    g_cmd_buffer = [g_queue commandBuffer];
    return g_cmd_buffer ? 0 : -1;
}

int q4_gpu_flush_commands(void) {
    if (!g_encoder) return 0;
    [g_encoder endEncoding];
    g_encoder = nil;
    [g_cmd_buffer commit];
    [g_cmd_buffer waitUntilCompleted];
    return 0;
}

int q4_gpu_end_commands(void) {
    if (g_encoder) {
        [g_encoder endEncoding];
        g_encoder = nil;
    }
    if (g_cmd_buffer) {
        [g_cmd_buffer commit];
        [g_cmd_buffer waitUntilCompleted];
        g_cmd_buffer = nil;
    }
    return 0;
}

int q4_gpu_synchronize(void) {
    return q4_gpu_end_commands();
}

/* =========================================================================
 * Model Mapping.
 * ========================================================================= */

int q4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    g_model_map = model_map;
    g_model_size = model_size;

    /* Create a no-copy MTLBuffer from the mmap */
    g_model_buffer = [g_device newBufferWithBytesNoCopy:(void *)model_map
                                                 length:(NSUInteger)model_size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    return g_model_buffer ? 0 : -1;
}

int q4_gpu_set_model_fd(int fd) {
    (void)fd;
    return -1; /* Not used on Metal */
}

int q4_gpu_set_model_map_range(const void *model_map, uint64_t model_size,
                                uint64_t map_offset, uint64_t map_size) {
    (void)model_map;
    (void)model_size;
    (void)map_offset;
    (void)map_size;
    return -1;
}

int q4_gpu_cache_model_range(const void *model_map, uint64_t model_size,
                              uint64_t offset, uint64_t bytes, const char *label) {
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
    (void)label;
    return 1; /* No-op on Metal; mmap handles caching */
}

int q4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    (void)kv_cache_bytes;
    (void)context_bytes;
    return 0; /* Use direct buffers for now */
}

void q4_gpu_set_quality(bool quality) {
    (void)quality;
}

void q4_gpu_print_memory_report(const char *label) {
    (void)label;
    fprintf(stderr, "q4: Metal memory report (not implemented)\n");
}

/* =========================================================================
 * Kernel Helpers.
 * ========================================================================= */

static id<MTLComputePipelineState> make_pipeline(const char *name) {
    NSString *n = [NSString stringWithUTF8String:name];
    NSError *err = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:n];
    if (!fn) {
        fprintf(stderr, "q4: Metal function '%s' not found\n", name);
        return nil;
    }
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "q4: failed to create pipeline for '%s': %s\n", name,
                err ? [[err localizedDescription] UTF8String] : "unknown");
    }
    return pso;
}

static void ensure_encoder(const char *name) {
    if (!g_encoder) {
        g_cmd_buffer = [g_queue commandBuffer];
        g_encoder = [g_cmd_buffer computeCommandEncoder];
        [g_encoder pushGroup:n];
    }
}

/* Round up to SIMD width */
static inline NSUInteger round_up_simd(NSUInteger n) {
    return (n + 31) / 32 * 32;
}

/* =========================================================================
 * Embedding Lookup.
 * ========================================================================= */

int q4_gpu_embed_token_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint32_t       n_vocab,
        uint32_t       token,
        uint32_t       n_embd) {
    (void)model_map;
    (void)model_size;
    (void)weight_offset;
    (void)n_vocab;
    (void)token;
    (void)n_embd;
    /* For now, this is handled on the CPU side */
    return -1;
}

int q4_gpu_embed_tokens_tensor(
        q4_gpu_tensor       *out,
        const q4_gpu_tensor *tokens,
        const void          *model_map,
        uint64_t             model_size,
        uint64_t             weight_offset,
        uint32_t             n_vocab,
        uint32_t             n_tokens,
        uint32_t             n_embd) {
    (void)out;
    (void)tokens;
    (void)model_map;
    (void)model_size;
    (void)weight_offset;
    (void)n_vocab;
    (void)n_tokens;
    (void)n_embd;
    return -1;
}

/* =========================================================================
 * Core Kernels.
 * ========================================================================= */

int q4_gpu_matmul_q8_0_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok) {
    ensure_encoder("matmul_q8_0");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_dense_matmul_q8_0");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    /* Encode arguments */
    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];

    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t n_tok;
    } args = { (uint32_t)in_dim, (uint32_t)out_dim, n_tok };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(out_dim, n_tok, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        q4_gpu_tensor *gate,
        q4_gpu_tensor *up,
        q4_gpu_tensor *mid,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        float          clamp) {
    ensure_encoder("gate_up_swiglu");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_shared_gate_up_swiglu_q8_0");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:gate->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:up->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:mid->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:gate_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:up_offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];

    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        float clamp;
    } args = { (uint32_t)in_dim, (uint32_t)out_dim, clamp };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(out_dim, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_rms_norm_weight_rows_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *x,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint32_t       n,
        uint32_t       rows,
        float          eps) {
    ensure_encoder("rms_norm");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_rms_norm_weight_rows");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:arg_idx++];

    struct {
        uint32_t n;
        uint32_t rows;
        float eps;
    } args = { n, rows, eps };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_rope_full_tensor(
        q4_gpu_tensor *x,
        uint32_t       n_tok,
        uint32_t       n_head,
        uint32_t       head_dim,
        uint32_t       pos0,
        float          freq_base,
        float          freq_scale) {
    ensure_encoder("rope_full");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_rope_full");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];

    struct {
        uint32_t n_tok;
        uint32_t n_head;
        uint32_t head_dim;
        uint32_t pos0;
        float freq_base;
        float freq_scale;
    } args = { n_tok, n_head, head_dim, pos0, freq_base, freq_scale };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(n_head * n_tok, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_flash_attn_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       q_offset,
        uint64_t       k_offset,
        uint64_t       v_offset,
        const q4_gpu_tensor *kv_cache_k,
        const q4_gpu_tensor *kv_cache_v,
        uint32_t       n_q_heads,
        uint32_t       n_kv_heads,
        uint32_t       head_dim,
        uint32_t       n_tokens,
        uint32_t       pos0,
        uint32_t       kv_len,
        float          logit_softcap) {
    ensure_encoder("flash_attn");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_flash_attn_gqa_decode");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:q_offset atIndex:arg_idx++];
    [g_encoder setBuffer:kv_cache_k ? kv_cache_k->buffer : g_model_buffer
                             offset:k_offset atIndex:arg_idx++];
    [g_encoder setBuffer:kv_cache_v ? kv_cache_v->buffer : g_model_buffer
                             offset:v_offset atIndex:arg_idx++];

    struct {
        uint32_t n_q_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
        uint32_t n_tokens;
        uint32_t kv_len;
        uint32_t pos0;
        float logit_softcap;
    } args = { n_q_heads, n_kv_heads, head_dim, n_tokens, kv_len, pos0, logit_softcap };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(n_q_heads, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_kv_cache_store_tensor(
        const q4_gpu_tensor *k,
        const q4_gpu_tensor *v,
        q4_gpu_tensor       *cache_k,
        q4_gpu_tensor       *cache_v,
        uint32_t             pos,
        uint32_t             n_kv_heads,
        uint32_t             head_dim) {
    if (!cache_k || !cache_v || !k || !v) return -1;

    uint64_t offset = (uint64_t)pos * n_kv_heads * head_dim * sizeof(float);
    uint64_t bytes = (uint64_t)n_kv_heads * head_dim * sizeof(float);

    if (offset + bytes > cache_k->bytes || offset + bytes > cache_v->bytes) return -1;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:k->buffer sourceOffset:0
                toBuffer:cache_k->buffer destinationOffset:offset
                    size:bytes];
    [blit copyFromBuffer:v->buffer sourceOffset:0
                toBuffer:cache_v->buffer destinationOffset:offset
                    size:bytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return 0;
}

int q4_gpu_deltanet_step_tensor(
        q4_gpu_tensor *out,
        q4_gpu_tensor *state,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       a_offset,
        uint64_t       b_offset,
        uint64_t       dt_offset,
        const q4_gpu_tensor *x,
        uint32_t       n_embd,
        uint32_t       head_dim,
        uint32_t       n_kv_heads) {
    ensure_encoder("deltanet_step");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_deltanet_step");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:state->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:a_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:b_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:dt_offset atIndex:arg_idx++];

    struct {
        uint32_t n_embd;
        uint32_t head_dim;
        uint32_t n_kv_heads;
        uint32_t n_tokens;
        uint32_t q_per_kv;
    } args = { n_embd, head_dim, n_kv_heads, 1, n_embd / (head_dim * n_kv_heads) };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(n_kv_heads, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_deltanet_prefill_tensor(
        q4_gpu_tensor *out,
        q4_gpu_tensor *state,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       a_offset,
        uint64_t       b_offset,
        uint64_t       dt_offset,
        const q4_gpu_tensor *x,
        uint32_t       n_tokens,
        uint32_t       n_embd,
        uint32_t       head_dim,
        uint32_t       n_kv_heads,
        uint32_t       n_q_heads) {
    ensure_encoder("deltanet_prefill");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_deltanet_prefill");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:state->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:a_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:b_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:dt_offset atIndex:arg_idx++];

    struct {
        uint32_t n_embd;
        uint32_t head_dim;
        uint32_t n_kv_heads;
        uint32_t n_tokens;
        uint32_t q_per_kv;
    } args = { n_embd, head_dim, n_kv_heads, n_tokens, n_q_heads / n_kv_heads };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(n_kv_heads, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_add_tensor(
        q4_gpu_tensor *y,
        const q4_gpu_tensor *x,
        uint64_t       n,
        float          scale) {
    ensure_encoder("add_scalar");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_unary_add_scalar");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:y->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBytes:&scale length:sizeof(scale) atIndex:arg_idx++];

    struct {
        int32_t ne00;
        uint64_t nb1;
    } args = { (int32_t)n, 0 };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(1, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_silu_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *x,
        uint64_t       n) {
    ensure_encoder("silu");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_unary_silu");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];

    struct {
        int32_t ne00;
        uint64_t nb1;
    } args = { (int32_t)n, 0 };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(1, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_mul_rows_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *a,
        const q4_gpu_tensor *b,
        uint32_t       rows,
        uint32_t       cols) {
    ensure_encoder("mul_rows");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_mul_mv");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:a->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:b->buffer offset:0 atIndex:arg_idx++];

    struct {
        uint32_t rows;
        uint32_t cols;
    } args = { rows, cols };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_residual_add_tensor(
        q4_gpu_tensor *x,
        const q4_gpu_tensor *residual,
        uint64_t       n) {
    ensure_encoder("residual_add");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_residual_add");
    if (!pso) {
        /* Fallback: copy residual to a temp buffer and add on CPU */
        if (!x || !residual || n * sizeof(float) > x->bytes) return -1;
        float *xp = x->buffer.contents;
        const float *rp = residual->buffer.contents;
        for (uint64_t i = 0; i < n; i++) xp[i] += rp[i];
        return 0;
    }

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];
    [g_encoder setBuffer:residual->buffer offset:0 atIndex:arg_idx++];

    struct {
        uint64_t n;
    } args = { n };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(round_up_simd(n), 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_softmax_tensor(
        q4_gpu_tensor *x,
        uint32_t       rows,
        uint32_t       cols) {
    ensure_encoder("softmax");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_row_softmax");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:x->buffer offset:0 atIndex:arg_idx++];

    struct {
        uint32_t rows;
        uint32_t cols;
    } args = { rows, cols };
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}
