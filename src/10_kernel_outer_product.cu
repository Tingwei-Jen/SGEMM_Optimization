#include "sgemm.h"
#include <stdio.h>
#define TILE_SIZE_COL 128
#define TILE_SIZE_ROW 64
#define BK 64 // block size in K direction

__global__ void sgemm_outer_product_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    int tx = threadIdx.x;

    // move A,B,C to current tile position
    A += blockIdx.y * TILE_SIZE_ROW * K;
    B += blockIdx.x * TILE_SIZE_COL;
    C += blockIdx.y * TILE_SIZE_ROW * N + blockIdx.x * TILE_SIZE_COL;

    // shared memory of A   
    __shared__ float sA[TILE_SIZE_ROW][BK];

    int numTotalThreads = blockDim.x;
    int sA_stride_size = numTotalThreads / BK;
    int rowIdx_sA = tx / BK;
    int colIdx_sA = tx % BK;

    // outer product result
    float cvalue[TILE_SIZE_ROW] = {0.0f};

    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {

        // load A to shared memory
        #pragma unroll
        for (int row = 0; row < TILE_SIZE_ROW; row += sA_stride_size) {
            sA[row + rowIdx_sA][colIdx_sA] = A[(row + rowIdx_sA) * K + colIdx_sA];
        }

        __syncthreads();

        // move to tx column
        float *bp = &B[tx];

        // calculate cvalue
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            float bvalue = *bp;

            // outer product
            #pragma unroll
            for (int e = 0; e < TILE_SIZE_ROW; e++ ){
                cvalue[e] += sA[e][i] * bvalue;
            }

            bp += N;
        }
        __syncthreads();

        // move A and B to next position
        A += BK;
        B += BK * N;
    }

    // write Cvalue to C
    // move to tx column
    #pragma unroll
    for (int i = 0; i < TILE_SIZE_ROW; i++) {
        int index = tx + i * N;
        C[index] = alpha * cvalue[i] + beta * C[index];
    }
}

void sgemm_outer_product(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(TILE_SIZE_COL, 1);
    dim3 dimGrid((N + TILE_SIZE_COL - 1) / TILE_SIZE_COL, (M + TILE_SIZE_ROW - 1) / TILE_SIZE_ROW);

    // int num_sms;
    // cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    // int n_threads = TILE_SIZE_COL;
    // int num_blocks_per_sm;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, sgemm_outer_product_kernel, n_threads, 0);
    // printf("num_blocks_per_sm: %d\n", num_blocks_per_sm);
    // printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y);
    sgemm_outer_product_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}