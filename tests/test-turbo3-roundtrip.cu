#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QK_TURBO3 32
#define QK_TURBO3_GROUP 128
typedef struct { __half norm; uint8_t qs[8]; uint8_t signs[4]; } block_turbo3_0;

static const float C[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static const float M[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};
static const float S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
static const float S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};

void fwht(float *x) {
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h*2)
            for (int j = i; j < i+h; j++) {
                float a=x[j], b=x[j+h]; x[j]=a+b; x[j+h]=a-b;
            }
    for (int i = 0; i < 128; i++) x[i] *= 0.08838834764831845f;
}

int main() {
    float in[128], rot[128], dec[128];
    for (int i = 0; i < 128; i++) in[i] = sinf(i * 0.1f);

    float nsq = 0;
    for (int i = 0; i < 128; i++) nsq += in[i]*in[i];
    float gn = sqrtf(nsq);
    for (int i = 0; i < 128; i++) rot[i] = in[i] / gn;

    // Forward rotation
    for (int i = 0; i < 128; i++) rot[i] *= S1[i];
    fwht(rot);
    for (int i = 0; i < 128; i++) rot[i] *= S2[i];

    // Quantize
    block_turbo3_0 blk[4];
    float rnsq = 0;
    for (int b = 0; b < 4; b++) {
        for (int j = 0; j < 8; j++) blk[b].qs[j] = 0;
        for (int j = 0; j < 4; j++) blk[b].signs[j] = 0;
        for (int j = 0; j < 32; j++) {
            float v = rot[b*32+j];
            uint8_t idx;
            if (v<M[0]) idx=0; else if (v<M[1]) idx=1; else if (v<M[2]) idx=2;
            else if (v<M[3]) idx=3; else if (v<M[4]) idx=4; else if (v<M[5]) idx=5;
            else if (v<M[6]) idx=6; else idx=7;
            blk[b].qs[j/4] |= (idx & 0x3) << ((j%4)*2);
            if (idx & 0x4) blk[b].signs[j/8] |= (1 << (j%8));
            rnsq += C[idx]*C[idx];
        }
    }
    float rn = sqrtf(rnsq);
    float cn = rn > 1e-10f ? gn/rn : gn;
    for (int b = 0; b < 4; b++) blk[b].norm = __float2half(cn);

    // Dequant + inverse rotation (full round-trip)
    for (int b = 0; b < 4; b++) {
        float nm = __half2float(blk[b].norm);
        for (int j = 0; j < 32; j++) {
            uint8_t lo = (blk[b].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi = (blk[b].signs[j/8] >> (j%8)) & 0x1;
            dec[b*32+j] = C[lo | (hi<<2)] * nm;
        }
    }
    for (int i = 0; i < 128; i++) dec[i] *= S2[i];
    fwht(dec);
    for (int i = 0; i < 128; i++) dec[i] *= S1[i];

    float maxe = 0, mse = 0;
    for (int i = 0; i < 128; i++) {
        float e = fabsf(in[i]-dec[i]);
        if (e > maxe) maxe = e;
        mse += e*e;
    }
    printf("ROUND-TRIP: MSE=%.6f max=%.4f\n", mse/128, maxe);
    printf("in:  "); for(int i=0;i<8;i++) printf("%.3f ",in[i]); printf("\n");
    printf("out: "); for(int i=0;i<8;i++) printf("%.3f ",dec[i]); printf("\n");
    printf("%s\n", maxe < 1.0f ? "PASS" : "FAIL");
    return maxe < 1.0f ? 0 : 1;
}
