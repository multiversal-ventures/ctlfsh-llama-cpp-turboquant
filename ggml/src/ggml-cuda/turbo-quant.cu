// TurboQuant CUDA kernels — turbo3/turbo4 set_rows (quantize) + dequant
//
// turbo3 needs a custom set_rows kernel because the WHT rotation spans
// 128-element groups (4 blocks of 32). The existing template processes
// one block at a time, which doesn't work for group-level rotation.
//
// turbo4 uses QK_TURBO4=128 (one block = one group), so a per-block
// template would work, but we use the same custom approach for consistency.

#include "turbo-quant.cuh"
#include "set-rows.cuh"

// ============================================================================
// turbo3 set_rows kernel — one thread per 128-element rotation group
// ============================================================================
template <typename idx_t>
static __global__ void k_set_rows_turbo3(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst,
        const int64_t ne00,       // row width in elements
        const int64_t ne01,       // number of rows per batch slice
        const int64_t ne02,
        const int64_t ne03,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,        // src0 strides in elements
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,        // src1 strides in elements
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,        // dst strides in bytes
        const int64_t nb2,
        const int64_t nb3) {

    const int64_t groups_per_row = ne00 / QK_TURBO3_GROUP;
    const int64_t total_groups = groups_per_row * ne01 * ne02 * ne03;
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= total_groups) return;

    // Decompose flat index into (group_in_row, row, batch dims)
    const int64_t i_grp = i % groups_per_row;
    const int64_t i_rem = i / groups_per_row;
    const int64_t i01   = i_rem % ne01;
    const int64_t i_rem2 = i_rem / ne01;
    const int64_t i02   = i_rem2 % ne02;
    const int64_t i03   = i_rem2 / ne02;

    // src1 lookup for destination row index
    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    // Source: 128 contiguous floats
    const float * src_grp = src0 + i01*s01 + i02*s02 + i03*s03 + i_grp * QK_TURBO3_GROUP;

    // Destination: 4 consecutive block_turbo3_0
    const int64_t blocks_per_row = ne00 / QK_TURBO3;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3);
    block_turbo3_0 * dst_grp = dst_row_ptr + i_grp * (QK_TURBO3_GROUP / QK_TURBO3);

    // DEBUG: dump first group's input and quantized output
    if (i == 0) {
        printf("TURBO3_DEBUG set_rows: ne00=%lld groups_per_row=%lld\n", (long long)ne00, (long long)groups_per_row);
        printf("TURBO3_DEBUG src[0..3]: %f %f %f %f\n", src_grp[0], src_grp[1], src_grp[2], src_grp[3]);
    }

    quantize_f32_turbo3_0_group(src_grp, dst_grp);

    if (i == 0) {
        // Read back: dequant first 4 elements
        float norm = __half2float(dst_grp[0].norm);
        uint8_t low2_0 = (dst_grp[0].qs[0] >> 0) & 0x3;
        uint8_t hi1_0 = (dst_grp[0].signs[0] >> 0) & 0x1;
        uint8_t idx0 = low2_0 | (hi1_0 << 2);
        printf("TURBO3_DEBUG quant: norm=%f idx[0]=%d centroid=%f\n", norm, idx0, TURBO_CENTROIDS_3BIT[idx0] * norm);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(blocks_per_row);
}

// ============================================================================
// turbo4 set_rows kernel — one thread per 128-element block
// ============================================================================
template <typename idx_t>
static __global__ void k_set_rows_turbo4(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo4_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3) {

    const int64_t blocks_per_row = ne00 / QK_TURBO4;
    const int64_t total_blocks = blocks_per_row * ne01 * ne02 * ne03;
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= total_blocks) return;

    const int64_t i_blk = i % blocks_per_row;
    const int64_t i_rem = i / blocks_per_row;
    const int64_t i01   = i_rem % ne01;
    const int64_t i_rem2 = i_rem / ne01;
    const int64_t i02   = i_rem2 % ne02;
    const int64_t i03   = i_rem2 / ne02;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src_blk = src0 + i01*s01 + i02*s02 + i03*s03 + i_blk * QK_TURBO4;

    block_turbo4_0 * dst_row_ptr = (block_turbo4_0 *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3);
    block_turbo4_0 * dst_blk = dst_row_ptr + i_blk;

    quantize_f32_turbo4_0_block(src_blk, dst_blk);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
}

// ============================================================================
// Host dispatch for turbo3 set_rows
// ============================================================================
template <typename idx_t>
static void set_rows_cuda_turbo3(
        const float * src0_d, const idx_t * src1_d, block_turbo3_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_TURBO3_GROUP == 0);

    const int64_t groups_per_row = ne00 / QK_TURBO3_GROUP;
    const int64_t total_groups = groups_per_row * ne01 * ne02 * ne03;

    if (total_groups == 0) return;

    const int block_size = 64;  // fewer threads — each does heavy work (128 floats)
    const int grid_size = (total_groups + block_size - 1) / block_size;

    k_set_rows_turbo3<<<grid_size, block_size, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12,
        nb01/sizeof(float), nb02/sizeof(float), nb03/sizeof(float),
        nb10/sizeof(idx_t), nb11/sizeof(idx_t), nb12/sizeof(idx_t),
        nb1, nb2, nb3);
}

// ============================================================================
// Host dispatch for turbo4 set_rows
// ============================================================================
template <typename idx_t>
static void set_rows_cuda_turbo4(
        const float * src0_d, const idx_t * src1_d, block_turbo4_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_TURBO4 == 0);

    const int64_t blocks_per_row = ne00 / QK_TURBO4;
    const int64_t total_blocks = blocks_per_row * ne01 * ne02 * ne03;

    if (total_blocks == 0) return;

    const int block_size = 64;
    const int grid_size = (total_blocks + block_size - 1) / block_size;

    k_set_rows_turbo4<<<grid_size, block_size, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12,
        nb01/sizeof(float), nb02/sizeof(float), nb03/sizeof(float),
        nb10/sizeof(idx_t), nb11/sizeof(idx_t), nb12/sizeof(idx_t),
        nb1, nb2, nb3);
}

// ============================================================================
// Public entry points called from set-rows.cu dispatch
// ============================================================================
template <typename idx_t>
void ggml_cuda_set_rows_turbo3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    set_rows_cuda_turbo3(
        src0_d, src1_d, (block_turbo3_0 *)dst->data,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1, nb2, nb3,
        ctx.stream());
}

template <typename idx_t>
void ggml_cuda_set_rows_turbo4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    set_rows_cuda_turbo4(
        src0_d, src1_d, (block_turbo4_0 *)dst->data,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1, nb2, nb3,
        ctx.stream());
}

// Explicit template instantiations
template void ggml_cuda_set_rows_turbo3<int32_t>(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *);
template void ggml_cuda_set_rows_turbo3<int64_t>(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *);
template void ggml_cuda_set_rows_turbo4<int32_t>(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *);
template void ggml_cuda_set_rows_turbo4<int64_t>(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *);

// ============================================================================
// turbo_split set_rows kernel — one thread per 128-element block
// ============================================================================
template <typename idx_t>
static __global__ void k_set_rows_turbo_split(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turbo_split_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3) {

    const int64_t blocks_per_row = ne00 / QK_TURBO_SPLIT;
    const int64_t total_blocks = blocks_per_row * ne01 * ne02 * ne03;
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= total_blocks) return;

    const int64_t i_blk = i % blocks_per_row;
    const int64_t i_rem = i / blocks_per_row;
    const int64_t i01   = i_rem % ne01;
    const int64_t i_rem2 = i_rem / ne01;
    const int64_t i02   = i_rem2 % ne02;
    const int64_t i03   = i_rem2 / ne02;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src_blk = src0 + i01*s01 + i02*s02 + i03*s03 + i_blk * QK_TURBO_SPLIT;

    block_turbo_split_0 * dst_row_ptr = (block_turbo_split_0 *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3);
    block_turbo_split_0 * dst_blk = dst_row_ptr + i_blk;

    quantize_f32_turbo_split_0_block(src_blk, dst_blk);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
}

// ============================================================================
// Host dispatch for turbo_split set_rows
// ============================================================================
template <typename idx_t>
static void set_rows_cuda_turbo_split(
        const float * src0_d, const idx_t * src1_d, block_turbo_split_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_TURBO_SPLIT == 0);

    const int64_t blocks_per_row = ne00 / QK_TURBO_SPLIT;
    const int64_t total_blocks = blocks_per_row * ne01 * ne02 * ne03;

    if (total_blocks == 0) return;

    const int block_size = 64;
    const int grid_size = (total_blocks + block_size - 1) / block_size;

    k_set_rows_turbo_split<<<grid_size, block_size, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12,
        nb01/sizeof(float), nb02/sizeof(float), nb03/sizeof(float),
        nb10/sizeof(idx_t), nb11/sizeof(idx_t), nb12/sizeof(idx_t),
        nb1, nb2, nb3);
}

template <typename idx_t>
void ggml_cuda_set_rows_turbo_split(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    set_rows_cuda_turbo_split(
        src0_d, src1_d, (block_turbo_split_0 *)dst->data,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1, nb2, nb3,
        ctx.stream());
}

template void ggml_cuda_set_rows_turbo_split<int32_t>(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *);
template void ggml_cuda_set_rows_turbo_split<int64_t>(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *);
