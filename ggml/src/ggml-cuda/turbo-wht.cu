// TurboQuant CUDA — GGML_OP_TURBO_WHT kernel
//
// Walsh-Hadamard Transform for pre-rotate-queries graph integration.
// Each thread handles one 128-element group (one attention head dimension).
// direction=0: forward rotation (signs1 → FWHT → signs2)
// direction=1: inverse rotation (signs2 → FWHT → signs1)
//
// Used by llama-graph.cpp build_attn() to rotate Q before attention
// and un-rotate output after attention. This moves O(n_ctx * d log d)
// cost to O(d log d) per query token.

#include "turbo-quant.cuh"

static __global__ void k_turbo_wht(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t ne00,     // innermost dimension (must be multiple of 128)
        const int64_t ne_total, // total number of 128-element groups
        const int     direction) {

    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne_total) return;

    // Each thread processes one 128-element group
    const int64_t offset = i * 128;
    float x[128];

    // Load
    for (int j = 0; j < 128; j++) {
        x[j] = src[offset + j];
    }

    // Transform
    if (direction == 0) {
        turbo_rotate_forward(x);
    } else {
        turbo_rotate_inverse(x);
    }

    // Store
    for (int j = 0; j < 128; j++) {
        dst[offset + j] = x[j];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] % 128 == 0);

    // DEBUG
    static int wht_call_count = 0;
    if (wht_call_count < 3) {
        int dir; memcpy(&dir, dst->op_params, sizeof(int));
        fprintf(stderr, "TURBO_WHT: dir=%d ne=[%lld,%lld,%lld,%lld] total_groups=%lld\n",
            dir, (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2], (long long)src0->ne[3],
            (long long)(ggml_nelements(src0)/128));
        wht_call_count++;
    }

    const float * src_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    // direction stored in op_params[0]
    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));
    GGML_ASSERT(direction == 0 || direction == 1);

    // Total elements / 128 = total groups
    const int64_t ne_total = ggml_nelements(src0) / 128;

    const int block_size = 128;
    const int grid_size = (ne_total + block_size - 1) / block_size;

    k_turbo_wht<<<grid_size, block_size, 0, ctx.stream()>>>(
        src_d, dst_d, src0->ne[0], ne_total, direction);
}
