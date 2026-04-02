#pragma once

#include "common.cuh"

// ============================================================================
// QJL reconstruction for turbo4 flash attention — warp shuffle FWHT
//
// Previous approach used __noinline__ device functions with float[128] local
// arrays. This caused massive register spilling (1KB per thread) and dropped
// generation from 47 t/s to 2 t/s at 8K context.
//
// This version uses warp shuffle FWHT: each thread holds 4 elements in
// registers, butterfly stages use __shfl_xor_sync. Zero local memory,
// zero shared memory, ~28 shuffles per FWHT.
//
// Layout: thread t holds elements [4t, 4t+1, 4t+2, 4t+3].
// ============================================================================

// Centroids for FA dequant
static __device__ const float TURBO_CENTROIDS_3BIT_QJL[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// QJL sign arrays (seed=1042)
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

// ============================================================================
// Warp shuffle FWHT: 128-element transform using 4 registers per thread.
// Modifies r0..r3 in place. All 32 lanes must participate.
// ============================================================================
static __device__ __forceinline__ void turbo4_warp_fwht(
        float & r0, float & r1, float & r2, float & r3, const int lane) {
    // h=1: local butterfly (r0,r1) and (r2,r3)
    {
        float a = r0, b = r1;
        r0 = a + b; r1 = a - b;
        a = r2; b = r3;
        r2 = a + b; r3 = a - b;
    }
    // h=2: local butterfly (r0,r2) and (r1,r3)
    {
        float a = r0, b = r2;
        r0 = a + b; r2 = a - b;
        a = r1; b = r3;
        r1 = a + b; r3 = a - b;
    }
    // h=4,8,16,32,64: warp shuffle butterflies
    #pragma unroll
    for (int h = 4; h < 128; h *= 2) {
        const int mask = h / 4;
        const float p0 = __shfl_xor_sync(0xFFFFFFFF, r0, mask);
        const float p1 = __shfl_xor_sync(0xFFFFFFFF, r1, mask);
        const float p2 = __shfl_xor_sync(0xFFFFFFFF, r2, mask);
        const float p3 = __shfl_xor_sync(0xFFFFFFFF, r3, mask);
        if ((lane & mask) == 0) {
            r0 += p0; r1 += p1; r2 += p2; r3 += p3;
        } else {
            r0 = p0 - r0; r1 = p1 - r1; r2 = p2 - r2; r3 = p3 - r3;
        }
    }
}

// ============================================================================
// Warp-cooperative turbo4 QJL reconstruction.
// Each thread unpacks 4 QJL signs, does inverse QJL WHT via warp shuffles,
// adds centroid, scales by norm. Result in r0..r3 (4 elements per thread).
//
// Layout: thread t's r0..r3 = reconstructed K[4t], K[4t+1], K[4t+2], K[4t+3]
// ============================================================================
static __device__ __forceinline__ void turbo4_warp_dequant_block(
        const block_turbo4_0 * __restrict__ blk,
        float & r0, float & r1, float & r2, float & r3,
        const int lane) {

    const int base = lane * 4;

    // Step 1: Unpack 4 QJL signs → ±1.0
    {
        const uint8_t sbyte = __ldg(&blk->signs[base / 8]);
        const int off = base % 8;
        r0 = ((sbyte >> (off + 0)) & 1) ? 1.0f : -1.0f;
        r1 = ((sbyte >> (off + 1)) & 1) ? 1.0f : -1.0f;
        r2 = ((sbyte >> (off + 2)) & 1) ? 1.0f : -1.0f;
        r3 = ((sbyte >> (off + 3)) & 1) ? 1.0f : -1.0f;
    }

    // Step 2: Apply QJL signs2
    r0 *= TURBO_QJL_S2_FA[base + 0];
    r1 *= TURBO_QJL_S2_FA[base + 1];
    r2 *= TURBO_QJL_S2_FA[base + 2];
    r3 *= TURBO_QJL_S2_FA[base + 3];

    // Step 3: Warp shuffle FWHT
    turbo4_warp_fwht(r0, r1, r2, r3, lane);

    // Step 4: Normalize (1/sqrt(128)) and apply signs1
    const float inv = 0.08838834764831845f;
    r0 *= inv * TURBO_QJL_S1_FA[base + 0];
    r1 *= inv * TURBO_QJL_S1_FA[base + 1];
    r2 *= inv * TURBO_QJL_S1_FA[base + 2];
    r3 *= inv * TURBO_QJL_S1_FA[base + 3];

    // Step 5: Scale QJL residual by sqrt(pi/2)/128 * rnorm
    const float rnorm = __half2float(__ldg(&blk->rnorm));
    const float qjl_scale = 1.2533141373155003f / 128.0f * rnorm;
    r0 *= qjl_scale;
    r1 *= qjl_scale;
    r2 *= qjl_scale;
    r3 *= qjl_scale;

    // Step 6: Add centroid and scale by norm
    const float norm = __half2float(__ldg(&blk->norm));

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int pos = base + j;
        const int bo = pos * 3;
        uint16_t raw;
        memcpy(&raw, &blk->qs[bo / 8], sizeof(uint16_t));
        const uint8_t idx = (raw >> (bo % 8)) & 0x7;
        const float centroid = TURBO_CENTROIDS_3BIT_QJL[idx];

        float * r = (j == 0) ? &r0 : (j == 1) ? &r1 : (j == 2) ? &r2 : &r3;
        *r = (centroid + *r) * norm;
    }
}
