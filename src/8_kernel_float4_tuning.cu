#include "sgemm.h"
#include <stdio.h>

template<int BN, int BM, int TN, int TM, int BK>
__global__ void sgemm_float4_tuning_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

   // left top corner of small tile in Tile
    int colIdx_smallTile = threadIdx.x * TN;
    int rowIdx_smallTile = threadIdx.y * TM;

    // move A,B,C to current tile position
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // shared memory of A and B
    __shared__ float sA_trans[BK][BM];
    __shared__ float sB[BK][BN];

    // number of threads in a block
    int numTotalThreads = blockDim.x * blockDim.y;
    // number of rows for loading A one time
    int sA_stride_size = numTotalThreads / (BK / 4);
    // number of rows for loading B one time
    int sB_stride_size = numTotalThreads / (BN / 4);

    // index of stride in shared memory
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int rowIdx_sA = localIdx / (BK / 4);
    int colIdx_sA = localIdx % (BK / 4);
    int rowIdx_sB = localIdx / (BN / 4);
    int colIdx_sB = localIdx % (BN / 4);

    // 初始化 Cvalue 為每個 thread 處理的多個結果
    float cvalue[TM][TN] = {0.0f};
    float avalue[TM] = {0.0f};
    float bvalue[TN] = {0.0f};

    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {
        // load A to shared memory
        #pragma unroll
        for (int i = 0; i < BM; i += sA_stride_size) {
            float4 temp = reinterpret_cast<float4 *>(&A[(i + rowIdx_sA) * K + colIdx_sA * 4])[0];
            sA_trans[colIdx_sA * 4 + 0][i + rowIdx_sA] = temp.x;
            sA_trans[colIdx_sA * 4 + 1][i + rowIdx_sA] = temp.y;
            sA_trans[colIdx_sA * 4 + 2][i + rowIdx_sA] = temp.z;
            sA_trans[colIdx_sA * 4 + 3][i + rowIdx_sA] = temp.w;
        }

        // load B to shared memory
        #pragma unroll
        for (int i = 0; i < BK; i += sB_stride_size) {
            reinterpret_cast<float4 *>(&sB[i + rowIdx_sB][colIdx_sB * 4])[0] 
                = reinterpret_cast<float4 *>(&B[(i + rowIdx_sB) * N + colIdx_sB * 4])[0];
        }

        __syncthreads();

        // move A and B to next position
        A += BK;
        B += BK * N;

        // calculate Cvalue
        #pragma unroll
        for (int e = 0; e < BK; ++e) {

            // load avalue and bvalue
            #pragma unroll
            for (int i = 0; i < TM; i+=4) {    // row
                reinterpret_cast<float4 *>(&avalue[i])[0] 
                    = reinterpret_cast<float4 *>(&sA_trans[e][rowIdx_smallTile + i])[0];
            }

            #pragma unroll
            for (int i = 0; i < TN; i+=4) {    // col
                reinterpret_cast<float4 *>(&bvalue[i])[0] 
                    = reinterpret_cast<float4 *>(&sB[e][colIdx_smallTile + i])[0];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {    // row
                #pragma unroll
                for (int j = 0; j < TN; j++) {   // col
                    cvalue[i][j] += avalue[i] * bvalue[j];
                }
            }
        }

        __syncthreads();
    }

    // write Cvalue to C
    #pragma unroll
    for (int i = 0; i < TM; i++) {    // row
        #pragma unroll
        for (int j = 0; j < TN; j+=4) {   // col
            // load C
            float4 temp = reinterpret_cast<float4 *>(&C[(rowIdx_smallTile + i) * N + (colIdx_smallTile + j)])[0];
            temp.x = alpha * cvalue[i][j] + beta * temp.x;
            temp.y = alpha * cvalue[i][j+1] + beta * temp.y;
            temp.z = alpha * cvalue[i][j+2] + beta * temp.z;
            temp.w = alpha * cvalue[i][j+3] + beta * temp.w;
            // write C
            reinterpret_cast<float4 *>(&C[(rowIdx_smallTile + i) * N + (colIdx_smallTile + j)])[0] = temp;
        }
    }
}


// BN: 128, BM: 128, TN: 8, TM: 8, BK: 16
// BN: 128, BM: 64, TN: 8, TM: 8, BK: 16
// BN: 128, BM: 32, TN: 8, TM: 8, BK: 32
// BN: 128, BM: 32, TN: 8, TM: 8, BK: 16
// BN: 64, BM: 128, TN: 8, TM: 8, BK: 16
// BN: 64, BM: 64, TN: 8, TM: 8, BK: 32
// BN: 64, BM: 64, TN: 8, TM: 8, BK: 16
// BN: 64, BM: 32, TN: 8, TM: 8, BK: 32
// BN: 64, BM: 32, TN: 8, TM: 8, BK: 16
// BN: 32, BM: 128, TN: 8, TM: 8, BK: 32
// BN: 32, BM: 128, TN: 8, TM: 8, BK: 16
// BN: 32, BM: 64, TN: 8, TM: 8, BK: 32
// BN: 32, BM: 64, TN: 8, TM: 8, BK: 16
// BN: 32, BM: 32, TN: 8, TM: 8, BK: 64
// BN: 32, BM: 32, TN: 8, TM: 8, BK: 32
// BN: 32, BM: 32, TN: 8, TM: 8, BK: 16

// BN: 128, BM: 128, TN: 8, TM: 8, BK: 16
void sgemm_tuning_128_128_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 128;
    int BM = 128;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<128,128,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 128, BM: 64, TN: 8, TM: 8, BK: 16
void sgemm_tuning_128_64_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 128;
    int BM = 64;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<128,64,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 128, BM: 32, TN: 8, TM: 8, BK: 32
void sgemm_tuning_128_32_8_8_32(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 128;
    int BM = 32;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<128,32,8,8,32><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 128, BM: 32, TN: 8, TM: 8, BK: 16
void sgemm_tuning_128_32_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 128;
    int BM = 32;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<128,32,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}


// BN: 64, BM: 128, TN: 8, TM: 8, BK: 16
void sgemm_tuning_64_128_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 64;
    int BM = 128;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<64,128,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 64, BM: 64, TN: 8, TM: 8, BK: 32
void sgemm_tuning_64_64_8_8_32(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 64;
    int BM = 64;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<64,64,8,8,32><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 64, BM: 64, TN: 8, TM: 8, BK: 16
void sgemm_tuning_64_64_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 64;
    int BM = 64;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<64,64,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 64, BM: 32, TN: 8, TM: 8, BK: 32
void sgemm_tuning_64_32_8_8_32(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 64;
    int BM = 32;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<64,32,8,8,32><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 64, BM: 32, TN: 8, TM: 8, BK: 16
void sgemm_tuning_64_32_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 64;
    int BM = 32;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<64,32,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 32, BM: 128, TN: 8, TM: 8, BK: 32
void sgemm_tuning_32_128_8_8_32(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 32;
    int BM = 128;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<32,128,8,8,32><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 32, BM: 128, TN: 8, TM: 8, BK: 16
void sgemm_tuning_32_128_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 32;
    int BM = 128;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<32,128,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}


// BN: 32, BM: 64, TN: 8, TM: 8, BK: 32
void sgemm_tuning_32_64_8_8_32(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 32;
    int BM = 64;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<32,64,8,8,32><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 32, BM: 64, TN: 8, TM: 8, BK: 16
void sgemm_tuning_32_64_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 32;
    int BM = 64;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<32,64,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 32, BM: 32, TN: 8, TM: 8, BK: 64
void sgemm_tuning_32_32_8_8_64(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 32;
    int BM = 32;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<32,32,8,8,64><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 32, BM: 32, TN: 8, TM: 8, BK: 32
void sgemm_tuning_32_32_8_8_32(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 32;
    int BM = 32;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<32,32,8,8,32><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

// BN: 32, BM: 32, TN: 8, TM: 8, BK: 16
void sgemm_tuning_32_32_8_8_16(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int BN = 32;
    int BM = 32;
    int TN = 8;
    int TM = 8;
    int blockDimX = BN / TN;
    int blockDimY = BM / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_tuning_kernel<32,32,8,8,16><<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}

void sgemm_float4_tuning(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // sgemm_tuning_128_128_8_8_16(A, B, C, M, N, K, alpha, beta);
    sgemm_tuning_128_64_8_8_16(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_128_32_8_8_32(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_128_32_8_8_16(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_64_128_8_8_16(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_64_64_8_8_32(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_64_64_8_8_16(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_64_32_8_8_32(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_64_32_8_8_16(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_32_128_8_8_32(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_32_128_8_8_16(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_32_64_8_8_32(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_32_64_8_8_16(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_32_32_8_8_64(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_32_32_8_8_32(A, B, C, M, N, K, alpha, beta);
    // sgemm_tuning_32_32_8_8_16(A, B, C, M, N, K, alpha, beta);
}