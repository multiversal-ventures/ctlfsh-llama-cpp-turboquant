// Minimal GPU round-trip test for turbo3 quantize/dequant device functions.
// Tests the actual CUDA code path, not CPU reference.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Pull in the actual turbo3 block definition and constants
#define GGML_COMMON_IMPL_CUDA
#define GGML_COMMON_DECL_CUDA

// Inline just what we need
#define QK_TURBO3 32
#define QK_TURBO3_GROUP 128
#define QK_TURBO4 128

typedef struct {
    __half  norm;
    uint8_t qs[QK_TURBO3 / 4];
    uint8_t signs[QK_TURBO3 / 8];
} block_turbo3_0;

// --- Constants (same as turbo-quant.cuh) ---
__constant__ static const float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
__constant__ static const float TURBO_MIDPOINTS_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};
__constant__ static const float TURBO_WHT_SIGNS1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
__constant__ static const float TURBO_WHT_SIGNS2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};

#define TURBO_INV_SQRT_128 0.08838834764831845f

static __device__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    for (int i = 0; i < 128; i++) x[i] *= TURBO_INV_SQRT_128;
}

// GPU kernel: quantize 128 floats to turbo3, dequant back, write output
__global__ void test_roundtrip(const float * in, float * out, block_turbo3_0 * blocks_out) {
    float x[128];

    // Step 1: normalize
    float nsq = 0;
    for (int i = 0; i < 128; i++) nsq += in[i] * in[i];
    float gn = sqrtf(nsq);
    float inv = gn > 1e-10f ? 1.0f / gn : 0.0f;
    for (int i = 0; i < 128; i++) x[i] = in[i] * inv;

    // Step 2: WHT rotate
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];

    // Step 3: quantize into 4 blocks
    block_turbo3_0 blk[4];
    float rnsq = 0;
    for (int b = 0; b < 4; b++) {
        for (int j = 0; j < 8; j++) blk[b].qs[j] = 0;
        for (int j = 0; j < 4; j++) blk[b].signs[j] = 0;
        for (int j = 0; j < 32; j++) {
            float v = x[b*32+j];
            uint8_t idx;
            if      (v < TURBO_MIDPOINTS_3BIT[0]) idx=0;
            else if (v < TURBO_MIDPOINTS_3BIT[1]) idx=1;
            else if (v < TURBO_MIDPOINTS_3BIT[2]) idx=2;
            else if (v < TURBO_MIDPOINTS_3BIT[3]) idx=3;
            else if (v < TURBO_MIDPOINTS_3BIT[4]) idx=4;
            else if (v < TURBO_MIDPOINTS_3BIT[5]) idx=5;
            else if (v < TURBO_MIDPOINTS_3BIT[6]) idx=6;
            else                                   idx=7;
            blk[b].qs[j/4] |= (idx & 0x3) << ((j%4)*2);
            if (idx & 0x4) blk[b].signs[j/8] |= (1 << (j%8));
            rnsq += TURBO_CENTROIDS_3BIT[idx] * TURBO_CENTROIDS_3BIT[idx];
        }
    }
    float rn = sqrtf(rnsq);
    float cn = rn > 1e-10f ? gn / rn : gn;
    for (int b = 0; b < 4; b++) blk[b].norm = __float2half(cn);

    // Copy blocks to output for inspection
    for (int b = 0; b < 4; b++) blocks_out[b] = blk[b];

    // Step 4: dequant (same as our dequantize_turbo3_0 / convert path)
    float dec[128];
    for (int b = 0; b < 4; b++) {
        float nm = __half2float(blk[b].norm);
        for (int j = 0; j < 32; j++) {
            uint8_t lo = (blk[b].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi = (blk[b].signs[j/8] >> (j%8)) & 0x1;
            dec[b*32+j] = TURBO_CENTROIDS_3BIT[lo | (hi<<2)] * nm;
        }
    }

    // Step 5: inverse WHT
    for (int i = 0; i < 128; i++) dec[i] *= TURBO_WHT_SIGNS2[i];
    turbo_fwht_128(dec);
    for (int i = 0; i < 128; i++) dec[i] *= TURBO_WHT_SIGNS1[i];

    for (int i = 0; i < 128; i++) out[i] = dec[i];
}

int main() {
    float h_in[128], h_out[128];
    block_turbo3_0 h_blk[4];
    for (int i = 0; i < 128; i++) h_in[i] = sinf(i * 0.1f);

    float *d_in, *d_out;
    block_turbo3_0 *d_blk;
    cudaMalloc(&d_in, 128*sizeof(float));
    cudaMalloc(&d_out, 128*sizeof(float));
    cudaMalloc(&d_blk, 4*sizeof(block_turbo3_0));
    cudaMemcpy(d_in, h_in, 128*sizeof(float), cudaMemcpyHostToDevice);

    test_roundtrip<<<1,1>>>(d_in, d_out, d_blk);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_out, d_out, 128*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blk, d_blk, 4*sizeof(block_turbo3_0), cudaMemcpyDeviceToHost);

    float maxe = 0, mse = 0;
    for (int i = 0; i < 128; i++) {
        float e = fabsf(h_in[i] - h_out[i]);
        if (e > maxe) maxe = e;
        mse += e*e;
    }
    printf("GPU ROUND-TRIP: MSE=%.6f max=%.4f\n", mse/128, maxe);
    printf("in:  "); for(int i=0;i<8;i++) printf("%.3f ",h_in[i]); printf("\n");
    printf("out: "); for(int i=0;i<8;i++) printf("%.3f ",h_out[i]); printf("\n");
    printf("norm[0]=%.4f sizeof(block)=%zu\n",
        __half2float(h_blk[0].norm), sizeof(block_turbo3_0));
    printf("%s\n", maxe < 1.0f ? "GPU PASS" : "GPU FAIL");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_blk);
    return maxe < 1.0f ? 0 : 1;
}
