#include "sgemm.h"
#include <stdio.h>
#define TILE_SIZE 128
#define TM 8  // number of element same thread handle in y direction 
#define TN 8  // number of element same thread handle in x direction
#define BK 16 // block size in K direction

__global__ void sgemm_float4_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // left top corner of small tile in Tile
    int colIdx_smallTile = threadIdx.x * TN;
    int rowIdx_smallTile = threadIdx.y * TM;

    // move A,B,C to current tile position
    A += blockIdx.y * TILE_SIZE * K;
    B += blockIdx.x * TILE_SIZE;
    C += blockIdx.y * TILE_SIZE * N + blockIdx.x * TILE_SIZE;

    // shared memory of A and B
    __shared__ float sA_trans[BK][TILE_SIZE];
    __shared__ float sB[BK][TILE_SIZE];

    // number of threads in a block
    int numTotalThreads = blockDim.x * blockDim.y;
    // number of rows for loading A one time
    int sA_stride_size = numTotalThreads / (BK / 4);
    // number of rows for loading B one time
    int sB_stride_size = numTotalThreads / (TILE_SIZE / 4);

    // index of stride in shared memory
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int rowIdx_sA = localIdx / (BK / 4);
    int colIdx_sA = localIdx % (BK / 4);
    int rowIdx_sB = localIdx / (TILE_SIZE / 4);
    int colIdx_sB = localIdx % (TILE_SIZE / 4);

    // 初始化 Cvalue 為每個 thread 處理的多個結果
    float cvalue[TM][TN] = {0.0f};
    float avalue[TM] = {0.0f};
    float bvalue[TN] = {0.0f};

    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {
        // load A to shared memory
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += sA_stride_size) {
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

void sgemm_float4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int blockDimX = TILE_SIZE / TN;
    int blockDimY = TILE_SIZE / TM;
    dim3 dimBlock(blockDimX, blockDimY);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // int num_sms;
    // cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    // int n_threads = blockDimX * blockDimY;
    // int num_blocks_per_sm;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, sgemm_tile2d_kernel, n_threads, 0);
    // printf("num_blocks_per_sm: %d\n", num_blocks_per_sm);
    sgemm_float4_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}