#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// TurboQuant 3-bit dequantize — 2 elements at a time
// 3-bit index = lower 2 bits from qs + upper 1 bit from signs
static const __device__ float TURBO_CENTROIDS_3BIT_DEQUANT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// QR_TURBO3 = 1: iqs is the element index, dequant returns (iqs, iqs+1)
#define QR_TURBO3 1
#define QI_TURBO3 (QK_TURBO3 / (2 * QR_TURBO3))

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    // iqs is the element position within the block (0, 2, 4, ..., 30)
    const int j0 = iqs;
    const int j1 = iqs + 1;

    // Extract 3-bit index for j0: lower 2 bits from qs, upper 1 bit from signs
    const uint8_t low2_0 = (x[ib].qs[j0 / 4] >> ((j0 % 4) * 2)) & 0x3;
    const uint8_t hi1_0  = (x[ib].signs[j0 / 8] >> (j0 % 8)) & 0x1;
    const uint8_t idx0   = low2_0 | (hi1_0 << 2);

    // Extract 3-bit index for j1
    const uint8_t low2_1 = (x[ib].qs[j1 / 4] >> ((j1 % 4) * 2)) & 0x3;
    const uint8_t hi1_1  = (x[ib].signs[j1 / 8] >> (j1 % 8)) & 0x1;
    const uint8_t idx1   = low2_1 | (hi1_1 << 2);

    v.x = TURBO_CENTROIDS_3BIT_DEQUANT[idx0] * norm;
    v.y = TURBO_CENTROIDS_3BIT_DEQUANT[idx1] * norm;
}

// turbo4: 3-bit PolarQuant + 1-bit QJL signs, QK=128
// Simplified dequant for get_rows (no inverse WHT — pre-rotate-queries handles it)
#define QR_TURBO4 1
#define QI_TURBO4 (QK_TURBO4 / (2 * QR_TURBO4))

static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float norm  = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const float qjl_scale = 1.2533141373155003f / 128.0f;  // sqrt(pi/2) / d

    // iqs is the element position (0, 2, 4, ..., 126)
    const int j0 = iqs;
    const int j1 = iqs + 1;

    // Unpack 3-bit index from bit-packed qs[48] for j0
    const int bo0 = j0 * 3;
    uint16_t raw0;
    memcpy(&raw0, &x[ib].qs[bo0 / 8], sizeof(uint16_t));
    const uint8_t idx0 = (raw0 >> (bo0 % 8)) & 0x7;

    // Unpack for j1
    const int bo1 = j1 * 3;
    uint16_t raw1;
    memcpy(&raw1, &x[ib].qs[bo1 / 8], sizeof(uint16_t));
    const uint8_t idx1 = (raw1 >> (bo1 % 8)) & 0x7;

    // PolarQuant centroid only for the per-element dequant path.
    // Full QJL reconstruction (inverse WHT) is done in the FA path
    // where we can process the full 128-element block at once.
    // The per-element path (get_rows) doesn't have block context.
    v.x = TURBO_CENTROIDS_3BIT_DEQUANT[idx0] * norm;
    v.y = TURBO_CENTROIDS_3BIT_DEQUANT[idx1] * norm;

    GGML_UNUSED(rnorm);
}
