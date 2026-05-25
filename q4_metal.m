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
    uint64_t offset;  /* byte offset into buffer (0 for non-view tensors) */
    bool is_view;
} q4_gpu_tensor;

/* Argument struct shared with Metal kernels (must match q4_metal_args_mul_mv in common.metal) */
typedef struct {
    int32_t  ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t  ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t  ne0, ne1;
    int32_t  nr0;
    int16_t  r2, r3;
} q4_metal_args_mul_mv;

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

    /* First: inline common.metal at the top so all kernels can use it */
    NSString *common_path = [metal_dir stringByAppendingPathComponent:@"common.metal"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:common_path]) {
        NSString *common = [NSString stringWithContentsOfFile:common_path
                                                     encoding:NSUTF8StringEncoding error:nil];
        if (common) {
            [source appendString:common];
            [source appendString:@"\n"];
        }
    }

    /* Then append all other .metal files, stripping their #include "common.metal" */
    for (NSString *file in files) {
        if (![file hasSuffix:@".metal"]) continue;
        if ([file isEqualToString:@"common.metal"]) continue;

        NSString *path = [metal_dir stringByAppendingPathComponent:file];
        NSString *content = [NSString stringWithContentsOfFile:path
                                                      encoding:NSUTF8StringEncoding error:nil];
        if (content) {
            /* Remove #include "common.metal" since we inlined it above */
            content = [content stringByReplacingOccurrencesOfString:@"#include \"common.metal\""
                                                         withString:@"// common.metal inlined above"];
            [source appendFormat:@"\n// --- %@ ---\n", file];
            [source appendString:content];
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
    t->offset = 0;
    t->is_view = false;
    return t;
}

q4_gpu_tensor *q4_gpu_tensor_view(const q4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset + bytes > base->bytes) return NULL;

    q4_gpu_tensor *t = malloc(sizeof(*t));
    t->buffer = base->buffer;
    t->bytes = bytes;
    t->offset = base->offset + offset;
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

uint64_t q4_gpu_tensor_offset(const q4_gpu_tensor *tensor) {
    return tensor ? tensor->offset : 0;
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
    memcpy((uint8_t *)tensor->buffer.contents + tensor->offset + offset, data, (size_t)bytes);
    return 0;
}

int q4_gpu_tensor_read(const q4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || offset + bytes > tensor->bytes) return -1;
    memcpy(data, (const uint8_t *)tensor->buffer.contents + tensor->offset + offset, (size_t)bytes);
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
    g_cmd_buffer = nil;  /* Clear reference to committed buffer */
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

    id<MTLComputePipelineState> pso = make_pipeline("kernel_mul_mv_q8_0_f32");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    /* Q8_0 blocks are 32 elements, each is 18 bytes */
    NSUInteger weight_bytes_per_row = (in_dim / 32) * 18;
    NSUInteger input_bytes_per_row = in_dim * sizeof(float);

    q4_metal_args_mul_mv args = {0};
    args.ne00 = (int)in_dim;
    args.ne01 = (int)out_dim;
    args.ne02 = 1;
    args.ne10 = (int)in_dim;
    args.ne11 = (int)n_tok;
    args.ne12 = 1;
    args.nb00 = 18;
    args.nb01 = weight_bytes_per_row;
    args.nb02 = 0;
    args.nb03 = 0;
    args.nb10 = sizeof(float);
    args.nb11 = input_bytes_per_row;
    args.nb12 = 0;
    args.nb13 = 0;
    args.ne0 = (int)out_dim;
    args.ne1 = (int)n_tok;
    args.nr0 = 2;
    args.r2 = 1;
    args.r3 = 1;

    /* Q8_0 kernel: buf[0]=args, buf[1]=src0(weight), buf[2]=src1(input), buf[3]=dst */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];

    NSUInteger n_blocks = (out_dim + 1) / 2;  /* NR0=2 rows per threadgroup */
    MTLSize grid = MTLSizeMake(n_blocks, n_tok, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder setThreadgroupMemoryLength:256 atIndex:0];
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_matmul_q4_k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok) {
    ensure_encoder("matmul_q4_k");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_mul_mv_q4_k_f32");
    if (!pso) return -1;

    (void)model_map;
    (void)model_size;

    [g_encoder setComputePipelineState:pso];

    /* Q4_K blocks are 256 elements, each is 144 bytes */
    NSUInteger weight_bytes_per_row = (in_dim / 256) * 144;
    NSUInteger input_bytes_per_row = in_dim * sizeof(float);

    q4_metal_args_mul_mv args = {0};
    args.ne00 = (int)in_dim;
    args.ne01 = (int)out_dim;
    args.ne02 = 1;
    args.ne10 = (int)in_dim;
    args.ne11 = (int)n_tok;
    args.ne12 = 1;
    args.nb00 = 1;  /* not used by Q4_K kernel */
    args.nb01 = weight_bytes_per_row;
    args.nb02 = 0;
    args.nb03 = 0;
    args.nb10 = sizeof(float);
    args.nb11 = input_bytes_per_row;
    args.nb12 = 0;
    args.nb13 = 0;
    args.ne0 = (int)out_dim;
    args.ne1 = (int)n_tok;
    args.nr0 = 2;
    args.r2 = 1;
    args.r3 = 1;

    /* Q4_K kernel: buf[0]=src0(weight), buf[1]=src1(input), buf[2]=dst, buf[3]=args */
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:0];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:1];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:2];
    [g_encoder setBytes:&args length:sizeof(args) atIndex:3];

    NSUInteger n_blocks = (out_dim + 1) / 2;  /* NR0=2 rows per threadgroup */
    MTLSize grid = MTLSizeMake(n_blocks, n_tok, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);

    /* Set threadgroup memory length explicitly */
    [g_encoder setThreadgroupMemoryLength:256 atIndex:0];

    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];
    return 0;
}

int q4_gpu_matmul_q6_k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok) {
    ensure_encoder("matmul_q6_k");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_mul_mv_q6_k_f32");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    /* Q6_K blocks are 256 elements, each is 210 bytes */
    NSUInteger weight_bytes_per_row = (in_dim / 256) * 210;
    NSUInteger input_bytes_per_row = in_dim * sizeof(float);

    q4_metal_args_mul_mv args = {0};
    args.ne00 = (int)in_dim;
    args.ne01 = (int)out_dim;
    args.ne02 = 1;
    args.ne10 = (int)in_dim;
    args.ne11 = (int)n_tok;
    args.ne12 = 1;
    args.nb00 = 1;  /* not used by Q6_K kernel */
    args.nb01 = weight_bytes_per_row;
    args.nb02 = 0;
    args.nb03 = 0;
    args.nb10 = sizeof(float);
    args.nb11 = input_bytes_per_row;
    args.nb12 = 0;
    args.nb13 = 0;
    args.ne0 = (int)out_dim;
    args.ne1 = (int)n_tok;
    args.nr0 = 2;
    args.r2 = 1;
    args.r3 = 1;

    /* Q6_K kernel: buf[0]=src0(weight), buf[1]=src1(input), buf[2]=dst, buf[3]=args */
    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];

    NSUInteger n_blocks = (out_dim + 1) / 2;  /* NR0=2 rows per threadgroup */
    MTLSize grid = MTLSizeMake(n_blocks, n_tok, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder setThreadgroupMemoryLength:256 atIndex:0];
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

/* Forward declarations */
static int q4_gpu_matmul_q5_k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok);

int q4_gpu_matmul_any_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        uint32_t       weight_type,
        const q4_gpu_tensor *x,
        uint32_t       n_tok) {
    /* Tensor type enum values: Q4_TENSOR_Q4_K=12, Q4_TENSOR_Q5_K=13, Q4_TENSOR_Q6_K=14, Q4_TENSOR_Q8_0=8 */
    switch (weight_type) {
    case 12: return q4_gpu_matmul_q4_k_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
    case 13: return q4_gpu_matmul_q5_k_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
    case 14: return q4_gpu_matmul_q6_k_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
    case 8:  return q4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
    default: return -1;
    }
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

    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        float clamp;
    } args = { (uint32_t)in_dim, (uint32_t)out_dim, clamp };

    /* kernel_shared_gate_up_swiglu_q8_0:
       buf[0]=args, buf[1]=src0_gate(model), buf[2]=src0_up(model),
       buf[3]=src1(input), buf[4]=dst_gate, buf[5]=dst_up, buf[6]=dst_mid,
       buf[7]=clamp_value */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:gate_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:up_offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:gate->buffer offset:gate->offset atIndex:arg_idx++];
    [g_encoder setBuffer:up->buffer offset:up->offset atIndex:arg_idx++];
    [g_encoder setBuffer:mid->buffer offset:mid->offset atIndex:arg_idx++];
    [g_encoder setBytes:&clamp length:sizeof(clamp) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(out_dim, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder setThreadgroupMemoryLength:512 atIndex:0];
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

    id<MTLComputePipelineState> pso = make_pipeline("kernel_rms_norm_mul_f32");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    struct {
        int32_t  ne00;
        int32_t  ne00_t;
        uint64_t nb1;
        uint64_t nb2;
        uint64_t nb3;
        float    eps;
    } args = { (int32_t)n, (int32_t)n, n * sizeof(float), 0, 0, eps };

    /* kernel_rms_norm_mul_f32: buf[0]=args, buf[1]=src0(input), buf[2]=src1(weight), buf[3]=dst */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];

    /* Each threadgroup processes one row. Threadgroup size = min(n, 256) rounded up to 32. */
    NSUInteger tgsz = 32;
    while (tgsz < n && tgsz < 256) tgsz *= 2;
    MTLSize grid = MTLSizeMake(rows * tgsz, 1, 1);
    MTLSize group = MTLSizeMake(tgsz, 1, 1);
    [g_encoder setThreadgroupMemoryLength:tgsz * sizeof(float) atIndex:0];
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

    struct {
        int32_t  ne0;
        int32_t  ne1;
        uint64_t nb1;
        float    freq_base;
        float    freq_scale;
        uint32_t pos0;
        uint32_t head_dim;
        uint32_t n_head;
    } args = { (int32_t)head_dim, (int32_t)(n_tok * n_head), (uint64_t)(n_tok * n_head * head_dim * sizeof(float)),
               freq_base, freq_scale, pos0, head_dim, n_head };

    /* kernel_rope_full: buf[0]=args, buf[1]=src0, buf[2]=dst */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];  /* dst = src0 (in-place) */

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

    struct {
        uint32_t n_q_heads;
        uint32_t n_kv_heads;
        uint32_t head_dim;
        uint32_t n_tokens;
        uint32_t kv_len;
        uint32_t pos0;
        float    scale;
        float logit_softcap;
        uint64_t q_stride;
        uint64_t k_stride;
        uint64_t v_stride;
        uint64_t out_stride;
    } args = { n_q_heads, n_kv_heads, head_dim, n_tokens, kv_len, pos0,
               1.0f / head_dim, logit_softcap,
               (uint64_t)head_dim * sizeof(float),
               (uint64_t)n_kv_heads * head_dim * sizeof(float),
               (uint64_t)n_kv_heads * head_dim * sizeof(float),
               (uint64_t)head_dim * sizeof(float) };

    /* kernel_flash_attn_gqa_decode: buf[0]=args, buf[1]=Q, buf[2]=K_cache, buf[3]=V_cache, buf[4]=out */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:q_offset atIndex:arg_idx++];
    [g_encoder setBuffer:kv_cache_k ? kv_cache_k->buffer : g_model_buffer
                             offset:k_offset atIndex:arg_idx++];
    [g_encoder setBuffer:kv_cache_v ? kv_cache_v->buffer : g_model_buffer
                             offset:v_offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(n_q_heads, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    /* shmem needs scores[kv_len] + weights[kv_len+1] floats, rounded to 16-byte alignment */
    NSUInteger shmem_sz = ((2 * kv_len + 1) * sizeof(float) + 15) & ~15UL;
    if (shmem_sz == 0) shmem_sz = 16;
    [g_encoder setThreadgroupMemoryLength:shmem_sz atIndex:0];
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

    struct {
        uint32_t n_embd;
        uint32_t head_dim;
        uint32_t n_kv_heads;
        uint32_t n_tokens;
        uint32_t q_per_kv;
    } args = { n_embd, head_dim, n_kv_heads, 1, n_embd / (head_dim * n_kv_heads) };

    /* kernel_deltanet_step: buf[0]=args, buf[1]=x, buf[2]=a_gate, buf[3]=b_proj,
       buf[4]=dt_gate, buf[5]=state, buf[6]=out */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:a_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:b_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:dt_offset atIndex:arg_idx++];
    [g_encoder setBuffer:state->buffer offset:state->offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];

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
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];
    [g_encoder setBuffer:state->buffer offset:state->offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
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

    struct {
        int32_t ne00;
        uint64_t nb1;
    } args = { (int32_t)n, 0 };

    /* kernel_unary_add_scalar: buf[0]=args, buf[1]=src0, buf[2]=dst, buf[3]=scale */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:y->buffer offset:y->offset atIndex:arg_idx++];
    [g_encoder setBytes:&scale length:sizeof(scale) atIndex:arg_idx++];

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

    struct {
        int32_t ne00;
        uint64_t nb1;
    } args = { (int32_t)n, 0 };

    /* kernel_unary_silu: buf[0]=args, buf[1]=src0, buf[2]=dst */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(1, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_silu_clamped_mul_tensor(
        q4_gpu_tensor *out,
        const q4_gpu_tensor *gate,
        const q4_gpu_tensor *up,
        uint64_t       n,
        float          clamp) {
    ensure_encoder("silu_clamped_mul");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_silu_clamped_mul");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];
    [g_encoder setBuffer:gate->buffer offset:gate->offset atIndex:arg_idx++];
    [g_encoder setBuffer:up->buffer offset:up->offset atIndex:arg_idx++];
    [g_encoder setBytes:&clamp length:sizeof(clamp) atIndex:arg_idx++];
    [g_encoder setBytes:&n length:sizeof(n) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(round_up_simd(n), 1, 1);
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

    id<MTLComputePipelineState> pso = make_pipeline("kernel_elementwise_mul");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];
    [g_encoder setBuffer:a->buffer offset:a->offset atIndex:arg_idx++];
    [g_encoder setBuffer:b->buffer offset:b->offset atIndex:arg_idx++];

    uint64_t n = (uint64_t)rows * cols;
    [g_encoder setBytes:&n length:sizeof(n) atIndex:arg_idx++];

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
        /* Fallback: add on CPU with correct offsets */
        if (!x || !residual || n * sizeof(float) > x->bytes) return -1;
        float *xp = (float *)((uint8_t *)x->buffer.contents + x->offset);
        const float *rp = (const float *)((const uint8_t *)residual->buffer.contents + residual->offset);
        for (uint64_t i = 0; i < n; i++) xp[i] += rp[i];
        return 0;
    }

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:residual->buffer offset:residual->offset atIndex:arg_idx++];

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

    id<MTLComputePipelineState> pso = make_pipeline("kernel_soft_max_f32");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    struct {
        int32_t  ne00;
        int32_t  ne01;
        uint64_t nb1;
        float    scale;
    } args = { (int32_t)cols, (int32_t)rows, cols * sizeof(float), 1.0f };

    /* kernel_soft_max_f32: buf[0]=args, buf[1]=src0, buf[2]=dst */
    NSUInteger arg_idx = 0;
    [g_encoder setBytes:&args length:sizeof(args) atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];  /* dst = src0 (in-place) */

    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder setThreadgroupMemoryLength:128 atIndex:0];
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

/* =========================================================================
 * DeltaNet Operations.
 * ========================================================================= */

int q4_gpu_vec_matmul_q4k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x) {
    ensure_encoder("vec_matmul_q4k");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_vec_matmul_q4k");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];
    [g_encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:arg_idx++];
    [g_encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:arg_idx++];

    // Use more threads: 256 threads per group, each group handles 256 rows
    MTLSize grid = MTLSizeMake(round_up_simd(out_dim), 1, 1);
    MTLSize group = MTLSizeMake(256, 1, 1);
    [g_encoder setThreadgroupMemoryLength:in_dim * sizeof(float) atIndex:0];
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_deltanet_conv_split_tensor(
        const q4_gpu_tensor *qkv_raw,
        const q4_gpu_tensor *conv_buf,
        const q4_gpu_tensor *conv_buf_out,
        const void          *model_map,
        uint64_t             model_size,
        uint64_t             conv_w_offset,
        q4_gpu_tensor       *q_exp,
        q4_gpu_tensor       *k_exp,
        q4_gpu_tensor       *v_out,
        uint32_t             qkv_dim,
        uint32_t             n_k_groups,
        uint32_t             n_v_heads,
        uint32_t             head_k_dim,
        uint32_t             head_v_dim,
        uint32_t             repeat,
        uint32_t             conv_pos) {
    ensure_encoder("deltanet_conv_split");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_deltanet_conv_split");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:qkv_raw->buffer offset:qkv_raw->offset atIndex:arg_idx++];
    [g_encoder setBuffer:conv_buf->buffer offset:conv_buf->offset atIndex:arg_idx++];
    [g_encoder setBuffer:conv_buf_out->buffer offset:conv_buf_out->offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:conv_w_offset atIndex:arg_idx++];
    [g_encoder setBuffer:q_exp->buffer offset:q_exp->offset atIndex:arg_idx++];
    [g_encoder setBuffer:k_exp->buffer offset:k_exp->offset atIndex:arg_idx++];
    [g_encoder setBuffer:v_out->buffer offset:v_out->offset atIndex:arg_idx++];
    [g_encoder setBytes:&qkv_dim length:sizeof(qkv_dim) atIndex:arg_idx++];
    [g_encoder setBytes:&n_k_groups length:sizeof(n_k_groups) atIndex:arg_idx++];
    [g_encoder setBytes:&n_v_heads length:sizeof(n_v_heads) atIndex:arg_idx++];
    [g_encoder setBytes:&head_k_dim length:sizeof(head_k_dim) atIndex:arg_idx++];
    [g_encoder setBytes:&head_v_dim length:sizeof(head_v_dim) atIndex:arg_idx++];
    [g_encoder setBytes:&repeat length:sizeof(repeat) atIndex:arg_idx++];
    [g_encoder setBytes:&conv_pos length:sizeof(conv_pos) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(round_up_simd(qkv_dim), 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_deltanet_gate_transform_tensor(
        const q4_gpu_tensor *alpha_raw,
        const q4_gpu_tensor *beta_raw,
        q4_gpu_tensor       *gate_out,
        q4_gpu_tensor       *beta_out,
        const void          *model_map,
        uint64_t             model_size,
        uint64_t             dt_bias_offset,
        uint64_t             ssm_a_offset,
        uint32_t             n) {
    ensure_encoder("deltanet_gate");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_deltanet_gate_transform");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:alpha_raw->buffer offset:alpha_raw->offset atIndex:arg_idx++];
    [g_encoder setBuffer:beta_raw->buffer offset:beta_raw->offset atIndex:arg_idx++];
    [g_encoder setBuffer:gate_out->buffer offset:gate_out->offset atIndex:arg_idx++];
    [g_encoder setBuffer:beta_out->buffer offset:beta_out->offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:dt_bias_offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:ssm_a_offset atIndex:arg_idx++];
    [g_encoder setBytes:&n length:sizeof(n) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(round_up_simd(n), 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_delta_rule_tensor(
        q4_gpu_tensor *state,
        const q4_gpu_tensor *k_exp,
        const q4_gpu_tensor *v_raw,
        const q4_gpu_tensor *gate,
        const q4_gpu_tensor *beta,
        q4_gpu_tensor *output,
        uint32_t n_v_heads,
        uint32_t head_v_dim,
        uint32_t head_k_dim) {
    ensure_encoder("delta_rule");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_delta_rule");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:state->buffer offset:state->offset atIndex:arg_idx++];
    [g_encoder setBuffer:k_exp->buffer offset:k_exp->offset atIndex:arg_idx++];
    [g_encoder setBuffer:v_raw->buffer offset:v_raw->offset atIndex:arg_idx++];
    [g_encoder setBuffer:gate->buffer offset:gate->offset atIndex:arg_idx++];
    [g_encoder setBuffer:beta->buffer offset:beta->offset atIndex:arg_idx++];
    [g_encoder setBuffer:output->buffer offset:output->offset atIndex:arg_idx++];
    [g_encoder setBytes:&n_v_heads length:sizeof(n_v_heads) atIndex:arg_idx++];
    [g_encoder setBytes:&head_v_dim length:sizeof(head_v_dim) atIndex:arg_idx++];
    [g_encoder setBytes:&head_k_dim length:sizeof(head_k_dim) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(n_v_heads, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_deltanet_silu_rms_tensor(
        q4_gpu_tensor *inout,
        const q4_gpu_tensor *z_raw,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       ssm_norm_w_offset,
        uint32_t       n_v_heads,
        uint32_t       head_v_dim) {
    ensure_encoder("deltanet_silu_rms");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_deltanet_silu_rms");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:inout->buffer offset:inout->offset atIndex:arg_idx++];
    [g_encoder setBuffer:z_raw->buffer offset:z_raw->offset atIndex:arg_idx++];
    [g_encoder setBuffer:g_model_buffer offset:ssm_norm_w_offset atIndex:arg_idx++];
    [g_encoder setBytes:&n_v_heads length:sizeof(n_v_heads) atIndex:arg_idx++];
    [g_encoder setBytes:&head_v_dim length:sizeof(head_v_dim) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(n_v_heads, 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

int q4_gpu_vec_matmul_q5k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x) {
    ensure_encoder("vec_matmul_q5k");

    id<MTLComputePipelineState> pso = make_pipeline("kernel_vec_matmul_q5k");
    if (!pso) return -1;

    [g_encoder setComputePipelineState:pso];

    NSUInteger arg_idx = 0;
    [g_encoder setBuffer:g_model_buffer offset:weight_offset atIndex:arg_idx++];
    [g_encoder setBuffer:x->buffer offset:x->offset atIndex:arg_idx++];
    [g_encoder setBuffer:out->buffer offset:out->offset atIndex:arg_idx++];
    [g_encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:arg_idx++];
    [g_encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:arg_idx++];

    MTLSize grid = MTLSizeMake(round_up_simd(out_dim), 1, 1);
    MTLSize group = MTLSizeMake(32, 1, 1);
    [g_encoder dispatchThreads:grid threadsPerThreadgroup:group];

    return 0;
}

/* Batched Q5_K matmul wrapper (calls vec kernel per token). */
int q4_gpu_matmul_q5_k_tensor(
        q4_gpu_tensor *out,
        const void    *model_map,
        uint64_t       model_size,
        uint64_t       weight_offset,
        uint64_t       in_dim,
        uint64_t       out_dim,
        const q4_gpu_tensor *x,
        uint32_t       n_tok) {
    /* For n_tok=1, just call the vec kernel directly. */
    if (n_tok == 1) {
        return q4_gpu_vec_matmul_q5k_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x);
    }
    /* For n_tok > 1, we'd need a batched kernel. Fall back to CPU for now. */
    return -1;
}
