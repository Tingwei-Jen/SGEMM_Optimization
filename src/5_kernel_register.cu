#include "sgemm.h"
#include <stdio.h>
#define TILE_SIZE 128
#define TM 8  // number of element same thread handle in y direction 
#define TN 8  // number of element same thread handle in x direction
#define BK 16 // block size in K direction

__global__ void sgemm_register_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // left top corner of small tile in Tile
    int colIdx_smallTile = threadIdx.x * TN;
    int rowIdx_smallTile = threadIdx.y * TM;

    // move A,B,C to current tile position
    A += blockIdx.y * TILE_SIZE * K;
    B += blockIdx.x * TILE_SIZE;
    C += blockIdx.y * TILE_SIZE * N + blockIdx.x * TILE_SIZE;

    // shared memory of A and B
    __shared__ float sA[TILE_SIZE][BK + 1];
    __shared__ float sB[BK][TILE_SIZE];

    // number of threads in a block
    int numTotalThreads = blockDim.x * blockDim.y;
    // number of rows for loading A one time
    int sA_stride_size = numTotalThreads / BK;
    // number of rows for loading B one time
    int sB_stride_size = numTotalThreads / TILE_SIZE;

    // index of stride in shared memory
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int rowIdx_sA = localIdx / BK;
    int colIdx_sA = localIdx % BK;
    int rowIdx_sB = localIdx / TILE_SIZE;
    int colIdx_sB = localIdx % TILE_SIZE;

    // 初始化 Cvalue 為每個 thread 處理的多個結果
    float cvalue[TM][TN] = {0.0f};
    float avalue[TM] = {0.0f};
    float bvalue[TN] = {0.0f};

    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {
        // load A to shared memory
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += sA_stride_size) {
            sA[i + rowIdx_sA][colIdx_sA] = A[(i + rowIdx_sA) * K + colIdx_sA];
        }

        // load B to shared memory
        #pragma unroll
        for (int i = 0; i < BK; i += sB_stride_size) {
            sB[i + rowIdx_sB][colIdx_sB] = B[(i + rowIdx_sB) * N + colIdx_sB];
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
            for (int i = 0; i < TM; i++) {    // row
                avalue[i] = sA[rowIdx_smallTile + i][e];
            }

            #pragma unroll
            for (int i = 0; i < TN; i++) {    // col
                bvalue[i] = sB[e][colIdx_smallTile + i];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {    // row
                #pragma unroll
                for (int j = 0; j < TN; j++) {   // col
                    // cvalue[i][j] += sA[rowIdx_smallTile + i][e] * sB[e][colIdx_smallTile + j];
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
        for (int j = 0; j < TN; j++) {   // col
                C[(rowIdx_smallTile + i) * N + (colIdx_smallTile + j)] = alpha * cvalue[i][j] + beta * C[(rowIdx_smallTile + i) * N + (colIdx_smallTile + j)];
        }
    }
}

void sgemm_register(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
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
    sgemm_register_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}