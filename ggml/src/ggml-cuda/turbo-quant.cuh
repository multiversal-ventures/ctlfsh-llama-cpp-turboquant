#pragma once

#include "common.cuh"

// ============================================================================
// TurboQuant CUDA — Constants and device functions for turbo3/turbo4
//
// Ported from Metal reference: ggml-metal.metal (kernel_set_rows_turbo,
// dequantize_turbo3_0, quantize_turbo4_0)
//
// turbo3: 3-bit MSE-only (3.5 bits/value, 4.6x compression vs fp16)
//   Block: 32 elements, 14 bytes. Rotation group: 128 elements (4 blocks).
//
// turbo4: 4-bit MSE+QJL (4.25 bits/value, 3.8x compression vs fp16)
//   Block: 128 elements, 68 bytes. Includes QJL residual signs.
// ============================================================================

// --- 3-bit Lloyd-Max centroids for N(0, 1/128) ---
__constant__ static const float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// --- 3-bit decision boundaries (midpoints between adjacent centroids) ---
__constant__ static const float TURBO_MIDPOINTS_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// --- 2-bit Lloyd-Max centroids for N(0, 1/sqrt(128)) ---
// Standard 2-bit centroids for N(0,1): +/-0.4528, +/-1.5104
// Scaled by sigma = 1/sqrt(128) = 0.08839
__constant__ static const float TURBO_CENTROIDS_2BIT[4] = {
    -0.133494f, -0.040023f, 0.040023f, 0.133494f
};

__constant__ static const float TURBO_MIDPOINTS_2BIT[3] = {
    -0.086759f, 0.0f, 0.086759f
};

// --- WHT rotation sign arrays (seed=42) ---
// These MUST match Metal turbo-wht.h and CPU ops.cpp exactly.
__constant__ static const float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

__constant__ static const float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// --- QJL sign arrays (seed=1042, for turbo4 residual projection) ---
__constant__ static const float TURBO_QJL_SIGNS1[128] = {
    1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
};

__constant__ static const float TURBO_QJL_SIGNS2[128] = {
    1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f
};

#define TURBO_INV_SQRT_128 0.08838834764831845f
#define TURBO_QJL_CONST    1.2533141373155003f  // sqrt(pi/2)

// ============================================================================
// Device helper: in-place Fast Walsh-Hadamard Transform (128 elements)
// O(n log n) = 896 FLOPs. FWHT is its own inverse (up to normalization).
// ============================================================================
static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; i++) {
        x[i] *= TURBO_INV_SQRT_128;
    }
}

// Forward rotation: signs1 → FWHT(includes 1/√d) → signs2
// turbo_fwht_128 already applies 1/√d normalization (line 107).
// Result: ||R(x)|| = ||x|| (orthogonal). Each coordinate ~ N(0,1/d).
static __device__ __forceinline__ void turbo_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
}

// Inverse rotation: signs2 → FWHT(includes 1/√d) → signs1
// Round-trip: R^{-1}(R(x)) = x (since FWHT applies 1/√d, two passes give 1/d × d = identity).
static __device__ __forceinline__ void turbo_rotate_inverse(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
}

// QJL forward rotation (seed=1042)
static __device__ __forceinline__ void turbo_qjl_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_QJL_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_QJL_SIGNS2[i];
}

// QJL inverse rotation
static __device__ __forceinline__ void turbo_qjl_rotate_inverse(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_QJL_SIGNS2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_QJL_SIGNS1[i];
}

// --- Nearest centroid lookup (3-bit, 7 comparisons) ---
static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MIDPOINTS_3BIT[0]) return 0;
    else if (val < TURBO_MIDPOINTS_3BIT[1]) return 1;
    else if (val < TURBO_MIDPOINTS_3BIT[2]) return 2;
    else if (val < TURBO_MIDPOINTS_3BIT[3]) return 3;
    else if (val < TURBO_MIDPOINTS_3BIT[4]) return 4;
    else if (val < TURBO_MIDPOINTS_3BIT[5]) return 5;
    else if (val < TURBO_MIDPOINTS_3BIT[6]) return 6;
    else                                    return 7;
}

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MIDPOINTS_2BIT[0]) return 0;
    else if (val < TURBO_MIDPOINTS_2BIT[1]) return 1;
    else if (val < TURBO_MIDPOINTS_2BIT[2]) return 2;
    else                                    return 3;
}

// ============================================================================
// Device quantize functions for set_rows integration
// ============================================================================

// Quantize 128 floats → 4 consecutive block_turbo3_0 (turbo3 set_rows)
static __device__ void quantize_f32_turbo3_0_group(
        const float * __restrict__ src,
        block_turbo3_0 * __restrict__ dst) {
    // Step 1: L2 norm
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3_GROUP; j++) {
        norm_sq += src[j] * src[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    // Step 2: normalize + WHT rotate
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward(x);

    // Step 3+4: quantize into 4 blocks of 32, accumulate recon norm
    float recon_norm_sq = 0.0f;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;  // 4

    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 * blk = &dst[b];
        const int off = b * QK_TURBO3;

        // Clear output
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk->qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk->signs[j] = 0;

        for (int j = 0; j < QK_TURBO3; j++) {
            float rv = x[off + j];
            uint8_t idx = turbo_nearest_centroid_3bit(rv);

            // Pack: lower 2 bits → qs, upper 1 bit → signs
            blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) {
                blk->signs[j / 8] |= (1 << (j % 8));
            }

            float c = TURBO_CENTROIDS_3BIT[idx];
            recon_norm_sq += c * c;
        }
    }

    // Step 5: norm correction — dequant gets exact original L2 norm for free
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst[b].norm = __float2half(corrected_norm);
    }
}

// Quantize 128 floats → 1 block_turbo4_0 (turbo4 set_rows)
static __device__ void quantize_f32_turbo4_0_block(
        const float * __restrict__ src,
        block_turbo4_0 * __restrict__ dst) {
    // Step 1: L2 norm + normalize
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4; j++) {
        norm_sq += src[j] * src[j];
    }
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst->norm = __float2half(norm);

    float x[128];
    for (int j = 0; j < 128; j++) {
        x[j] = src[j] * inv_norm;
    }

    // Step 2: WHT rotate
    turbo_rotate_forward(x);

    // Step 3: 3-bit PolarQuant (bit-packed into qs[48])
    // x[j] = rotated values, recon[j] = centroid of rotated values
    for (int j = 0; j < QK_TURBO4 * 3 / 8; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBO4 / 8; j++) dst->signs[j] = 0;

    float recon[128];
    for (int j = 0; j < 128; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(x[j]);
        recon[j] = TURBO_CENTROIDS_3BIT[idx];

        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos = bit_offset % 8;
        dst->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            dst->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
        }
    }

    // Step 4: residual in ROTATED space (both x and recon are rotated)
    // This differs from Metal which uses mixed space (normalized - recon).
    // Rotated-space residual is compatible with pre-rotate-queries:
    // dequant can reconstruct entirely in rotated space.
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        x[j] = x[j] - recon[j];  // rotated - rotated = rotated residual
        rnorm_sq += x[j] * x[j];
    }
    dst->rnorm = __float2half(sqrtf(rnorm_sq));

    // Step 5: QJL projection — 1-bit signs of random projection
    turbo_qjl_rotate_forward(x);
    for (int i = 0; i < 128; i++) {
        if (x[i] >= 0.0f) {
            dst->signs[i / 8] |= (1 << (i % 8));
        }
    }
}

// Quantize 128 floats → 1 block_turbo_split_0 (outlier-aware 32@3bit + 96@2bit)
static __device__ void quantize_f32_turbo_split_0_block(
        const float * __restrict__ src,
        block_turbo_split_0 * __restrict__ dst) {
    // Step 1: L2 norm
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO_SPLIT; j++) {
        norm_sq += src[j] * src[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    // Step 2: normalize + WHT rotate
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward(x);

    // Step 3: find top 32 by magnitude (outlier selection)
    // Partial selection sort: find 32 largest magnitudes
    float magnitudes[128];
    int indices[128];
    for (int j = 0; j < 128; j++) {
        magnitudes[j] = fabsf(x[j]);
        indices[j] = j;
    }
    for (int i = 0; i < 32; i++) {
        int max_idx = i;
        for (int j = i + 1; j < 128; j++) {
            if (magnitudes[j] > magnitudes[max_idx]) max_idx = j;
        }
        float tmp_m = magnitudes[i]; magnitudes[i] = magnitudes[max_idx]; magnitudes[max_idx] = tmp_m;
        int tmp_i = indices[i]; indices[i] = indices[max_idx]; indices[max_idx] = tmp_i;
    }

    // DIAGNOSTIC: fixed mask — first 32 channels are outliers, rest regular
    // This removes the per-block dynamic selection to test if the split mechanism works
    memset(dst->outlier_mask, 0, 16);
    dst->outlier_mask[0] = 0xFF;  // channels 0-7
    dst->outlier_mask[1] = 0xFF;  // channels 8-15
    dst->outlier_mask[2] = 0xFF;  // channels 16-23
    dst->outlier_mask[3] = 0xFF;  // channels 24-31

    float recon_norm_sq = 0.0f;
    int o_idx = 0, r_idx = 0;
    memset(dst->qs_outlier, 0, 12);
    memset(dst->qs_regular, 0, 36);

    for (int j = 0; j < 128; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(x[j]);
        if (j < 32) {
            // Outlier: pack into qs_outlier
            int bo = o_idx * 3;
            int byte_idx = bo / 8;
            int bit_pos = bo % 8;
            dst->qs_outlier[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < 12) {
                dst->qs_outlier[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
            }
            o_idx++;
        } else {
            // Regular: pack into qs_regular
            int bo = r_idx * 3;
            int byte_idx = bo / 8;
            int bit_pos = bo % 8;
            dst->qs_regular[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < 36) {
                dst->qs_regular[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
            }
            r_idx++;
        }
        recon_norm_sq += TURBO_CENTROIDS_3BIT[idx] * TURBO_CENTROIDS_3BIT[idx];
    }

    // Step 5: norm correction — dequant gets exact original L2 norm for free
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    dst->norm = __float2half(corrected_norm);
}

// Gemma 3 4B profiled channel permutation: top 32 variance channels → positions 0-31
// Generated from 2000-sample empirical profiling on T4 (2026-04-04)
// PERM[j] maps original channel j → permuted position (used in quantize)
// PERM_INV[p] maps permuted position p → original channel (used in dequant to find where to read)
static __device__ const int TURBO_SPLIT2_PERM[128] = {
    111,  70, 104,  71,  67, 105, 108,  64, 125, 106, 110,  66,  96, 107, 109,  99,
    101, 123,  97, 124, 103,  68,  69,  74,  75,  65, 126,  73, 122, 102,  78, 120,
     82,  80, 112, 100,  84, 116,  86,  77, 115,  85,  98,  72,  83,  79,  91, 127,
     93, 113, 117, 121,  81,  88,  76, 114,  87, 119,  89,  95,  40,  90,  92,  47,
     36,  35,  94,  41,  50, 118,  62,  33,   5,  48,  56,  38,  37,  49,  39,  58,
     51,  59,  45,  32,   0,  42,  61,  44,   9,  43,  60,  34,  14,  31,  29,  53,
      6,  52,   4,   1,   7,  63,  55,  10,  18,  57,  15,  23,  13,   2,   8,  22,
     11,  19,  16,  46,  54,  17,   3,  28,  12,  30,  20,  24,  25,  21,  27,  26
};

static __device__ const int TURBO_SPLIT2_PERM_INV[128] = {
     84,  99, 109, 118,  98,  72,  96, 100, 110,  88, 103, 112, 120, 108,  92, 106,
    114, 117, 104, 113, 122, 125, 111, 107, 123, 124, 127, 126, 119,  94, 121,  93,
     83,  71,  91,  65,  64,  76,  75,  78,  60,  67,  85,  89,  87,  82, 115,  63,
     73,  77,  68,  80,  97,  95, 116, 102,  74, 105,  79,  81,  90,  86,  70, 101,
      7,  25,  11,   4,  21,  22,   1,   3,  43,  27,  23,  24,  54,  39,  30,  45,
     33,  52,  32,  44,  36,  41,  38,  56,  53,  58,  61,  46,  62,  48,  66,  59,
     12,  18,  42,  15,  35,  16,  29,  20,   2,   5,   9,  13,   6,  14,  10,   0,
     34,  49,  55,  40,  37,  50,  69,  57,  31,  51,  28,  17,  19,   8,  26,  47
};

// Profiling: accumulate per-channel variance after WHT rotation
#ifndef TURBO_SPLIT2_PROFILE
#define TURBO_SPLIT2_PROFILE 0
#endif
#define TURBO_SPLIT2_PROFILE_N 2000

__device__ static float turbo_split2_var_accum[128] = {0};
__device__ static int   turbo_split2_var_count = 0;

// Quantize 128 floats → 1 block_turbo_split2_0 (2.5-bit split: 32@3bit + 96@2bit)
static __device__ void quantize_f32_turbo_split2_0_block(
        const float * __restrict__ src,
        block_turbo_split2_0 * __restrict__ dst) {
    // Step 1: L2 norm + normalize
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO_SPLIT2; j++) norm_sq += src[j] * src[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    // Step 2: Normalize + WHT rotate
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward(x);

    // Profiling: accumulate per-channel x^2 after WHT
#if TURBO_SPLIT2_PROFILE
    {
        for (int j = 0; j < 128; j++) {
            atomicAdd(&turbo_split2_var_accum[j], x[j] * x[j]);
        }
        int prev = atomicAdd(&turbo_split2_var_count, 1);
        if (prev == TURBO_SPLIT2_PROFILE_N - 1) {
            // Print variance ranking after N tokens
            printf("TURBO_SPLIT2_PROFILE: %d samples collected\n", TURBO_SPLIT2_PROFILE_N);
            printf("CHANNEL_VARIANCE:");
            for (int j = 0; j < 128; j++) {
                printf(" %.6f", turbo_split2_var_accum[j] / TURBO_SPLIT2_PROFILE_N);
            }
            printf("\n");
        }
    }
#endif

    // Step 3: Clear output arrays
    for (int j = 0; j < 12; j++) dst->qs_hi[j] = 0;
    for (int j = 0; j < 24; j++) dst->qs_lo[j] = 0;
    dst->padding[0] = 0;
    dst->padding[1] = 0;

    // Step 4: Permute channels — reorder by variance (high-variance first)
    // PERM maps: descending-variance-rank → original channel index
    // So permuted[p] = x[PERM[p]]: position p gets the value from original channel PERM[p]
    float xp[128];
    for (int j = 0; j < 128; j++) xp[j] = x[TURBO_SPLIT2_PERM[j]];

    // Step 5: Quantize permuted channels 0-31 (outliers) with 3-bit → qs_hi[12]
    float recon_norm_sq = 0.0f;
    for (int j = 0; j < 32; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(xp[j]);
        recon_norm_sq += TURBO_CENTROIDS_3BIT[idx] * TURBO_CENTROIDS_3BIT[idx];

        // 3-bit contiguous bit-pack
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos = bit_offset % 8;
        dst->qs_hi[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < 12) {
            dst->qs_hi[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
        }
    }

    // Step 6: Quantize permuted channels 32-127 (regular) with 2-bit → qs_lo[24]
    for (int j = 32; j < 128; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(xp[j]);
        recon_norm_sq += TURBO_CENTROIDS_2BIT[idx] * TURBO_CENTROIDS_2BIT[idx];

        // 2-bit pack: 4 values per byte
        int rpos = j - 32;  // 0-95
        dst->qs_lo[rpos / 4] |= (uint8_t)(idx << ((rpos % 4) * 2));
    }

    // Step 6: Norm correction
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    dst->norm = __float2half(corrected_norm);
}

// ============================================================================
// Op dispatch declarations
// ============================================================================
void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
