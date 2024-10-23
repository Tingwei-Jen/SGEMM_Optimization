#include "sgemm.h"
#include <stdio.h>
#define TILE_SIZE 128
#define BK 32 // block size in K direction
#define TN 4 // number of element same thread handle in x direction

__global__ void sgemm_outer_product_float4_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // local thread index
    int tx = threadIdx.x;
    int localColIdx = tx * TN;

    // move A,B,C to current tile position
    A += blockIdx.y * TILE_SIZE * K;
    B += blockIdx.x * TILE_SIZE * TN;
    C += blockIdx.y * TILE_SIZE * N + blockIdx.x * TILE_SIZE * TN;

    // shared memory of A   
    __shared__ float sA[TILE_SIZE][BK];

    int numTotalThreads = blockDim.x;
    int sA_stride_size = numTotalThreads / (BK /4);
    int rowIdx_sA = tx / (BK / 4);
    int colIdx_sA = tx % (BK / 4);
    // printf("sA_stride_size= %d, rowIdx_sA = %d, colIdx_sA = %d\n", sA_stride_size, rowIdx_sA, colIdx_sA);
    // outer product result
    float cvalue[TILE_SIZE][TN] = {0.0f};

    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {

        // load A to shared memory
        #pragma unroll
        for (int row = 0; row < TILE_SIZE; row += sA_stride_size) {
            reinterpret_cast<float4 *>(&sA[row + rowIdx_sA][colIdx_sA * 4])[0] = 
                reinterpret_cast<float4 *>(&A[(row + rowIdx_sA) * K + colIdx_sA * 4])[0];
        }
        
        __syncthreads();

        // move to tx column
        // float4 bp = reinterpret_cast<float4 *>(&B[localColIdx])[0];

        // calculate cvalue
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            #pragma unroll
            for (int j = 0; j < TN/4; j++) {
                float4 bp = reinterpret_cast<float4 *>(&B[(i * N + localColIdx) + j * 4])[0];
                float bvalue0 = bp.x;
                float bvalue1 = bp.y;
                float bvalue2 = bp.z;
                float bvalue3 = bp.w;

                // outer product
                #pragma unroll
                for (int e = 0; e < TILE_SIZE; e++ ){
                    cvalue[e][j * 4 + 0] += sA[e][i] * bvalue0;
                    cvalue[e][j * 4 + 1] += sA[e][i] * bvalue1;
                    cvalue[e][j * 4 + 2] += sA[e][i] * bvalue2;
                    cvalue[e][j * 4 + 3] += sA[e][i] * bvalue3;
                }
            }
        }
        __syncthreads();

        // move A and B to next position
        A += BK;
        B += BK * N;
    }

    // write Cvalue to C
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        int index = localColIdx + i * N;
        #pragma unroll
        for (int j = 0; j < TN; j+=4) {
            float4 temp = reinterpret_cast<float4 *>(&C[index + j])[0];
            temp.x = alpha * cvalue[i][j] + beta * temp.x;
            temp.y = alpha * cvalue[i][j + 1] + beta * temp.y;
            temp.z = alpha * cvalue[i][j + 2] + beta * temp.z;
            temp.w = alpha * cvalue[i][j + 3] + beta * temp.w;
            reinterpret_cast<float4 *>(&C[index + j])[0] = temp;
        }
    }
}

void sgemm_outer_product_float4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(TILE_SIZE, 1);
    dim3 dimGrid((N + (TILE_SIZE * TN) - 1) / (TILE_SIZE * TN), (M + TILE_SIZE - 1) / TILE_SIZE);

    // int num_sms;
    // cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    // int n_threads = TILE_SIZE;
    // int num_blocks_per_sm;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, sgemm_outer_product_float4_kernel, n_threads, 0);
    // printf("num_blocks_per_sm: %d\n", num_blocks_per_sm);
    // printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y);
    sgemm_outer_product_float4_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);
}