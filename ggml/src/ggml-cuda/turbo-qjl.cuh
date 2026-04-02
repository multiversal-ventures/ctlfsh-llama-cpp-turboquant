#pragma once

#include "common.cuh"
#include "turbo-quant.cuh"

// ============================================================================
// QJL reconstruction for turbo4 flash attention
//
// These functions are __noinline__ to prevent nvcc from inlining the
// 128-element FWHT butterfly into every FA template instantiation.
// The reverted commit 8c031cfac had these as __forceinline__ which
// tripled compile time and made 4-vCPU machines unresponsive.
//
// __noinline__ means nvcc compiles the function once per translation unit
// and generates a device function call — negligible overhead vs the
// 896 FLOPs of the FWHT itself.
// ============================================================================

// Centroids array accessible from FA templates
static __device__ const float TURBO_CENTROIDS_3BIT_QJL[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// QJL sign arrays for inverse reconstruction (seed=1042)
// Duplicated from turbo-quant.cuh for FA compilation units.
static __device__ const float TURBO_QJL_S1_FA[128] = {
    1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,
    1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,
    -1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,1,
    1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1
};
static __device__ const float TURBO_QJL_S2_FA[128] = {
    1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,1,
    -1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,1,-1,1,1,
    -1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,
    1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,1,1,-1
};

// Inverse QJL WHT: unpack signs → signs2 → FWHT → signs1
// __noinline__ is critical — prevents nvcc template bloat
static __device__ __noinline__ void turbo4_qjl_inverse_128(float * __restrict__ x) {
    // Apply signs2
    for (int i = 0; i < 128; i++) x[i] *= TURBO_QJL_S2_FA[i];

    // FWHT (self-inverse up to normalization)
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // Normalize: 1/sqrt(128)
    const float inv = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv;

    // Apply signs1
    for (int i = 0; i < 128; i++) x[i] *= TURBO_QJL_S1_FA[i];
}

// Reconstruct full turbo4 block (128 elements) with QJL.
// Returns centroid[j] + qjl_recon[j], all in rotated space, scaled by norm.
// __noinline__ — this calls turbo4_qjl_inverse_128 which is also noinline.
static __device__ __noinline__ void turbo4_dequant_block_qjl(
        const block_turbo4_0 * __restrict__ blk, float * __restrict__ out) {

    const float norm  = __half2float(blk->norm);
    const float rnorm = __half2float(blk->rnorm);
    const float qjl_scale = 1.2533141373155003f / 128.0f; // sqrt(pi/2) / d

    // Step 1: unpack QJL signs → ±1.0
    float qjl[128];
    for (int j = 0; j < 128; j++) {
        uint8_t bit = (blk->signs[j / 8] >> (j % 8)) & 0x1;
        qjl[j] = bit ? 1.0f : -1.0f;
    }

    // Step 2: inverse QJL WHT → residual estimate in rotated space
    turbo4_qjl_inverse_128(qjl);

    // Step 3: scale residual by sqrt(pi/2)/128 * rnorm
    for (int j = 0; j < 128; j++) {
        qjl[j] *= qjl_scale * rnorm;
    }

    // Step 4: centroid + QJL residual, all in rotated space, scaled by norm
    for (int j = 0; j < 128; j++) {
        const int bo = j * 3;
        uint16_t raw;
        memcpy(&raw, &blk->qs[bo / 8], sizeof(uint16_t));
        const uint8_t idx = (raw >> (bo % 8)) & 0x7;

        out[j] = (TURBO_CENTROIDS_3BIT_QJL[idx] + qjl[j]) * norm;
    }
}
