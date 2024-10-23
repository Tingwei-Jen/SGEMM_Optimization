#include "sgemm.h"
#include <stdio.h>
#define BLOCKDIM_X 32
#define BLOCKDIM_Y 4
#define TN 4  // number of element same thread handle in x direction
#define TM 16 // number of element same thread handle in y direction
#define BK 8  // block size in K direction

__global__ void sgemm_reorder_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tileX = BLOCKDIM_X * TN;
    int tileY = BLOCKDIM_Y * TM;

    // move A,B,C to current tile position
    A += blockIdx.y * tileY * K;
    B += blockIdx.x * tileX;
    C += blockIdx.y * tileY * N + blockIdx.x * tileX;

    // shared memory of A and B
    __shared__ float sA_trans[BK][BLOCKDIM_Y * TM];
    __shared__ float sB[BK][BLOCKDIM_X * TN];

    // // number of threads in a block
    int numTotalThreads = blockDim.x * blockDim.y;
    // number of rows for loading A one time
    int sA_stride_size = numTotalThreads / (BK / 4);
    // number of rows for loading B one time
    int sB_stride_size = numTotalThreads / (tileX / 4);

    // index of stride in shared memory
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int rowIdx_sA = localIdx / (BK / 4);
    int colIdx_sA = localIdx % (BK / 4);
    int rowIdx_sB = localIdx / (tileX / 4);
    int colIdx_sB = localIdx % (tileX / 4);

    // 初始化 Cvalue 為每個 thread 處理的多個結果
    float cvalue[TM][TN] = {0.0f};
    float avalue[TM] = {0.0f};
    float bvalue[TN] = {0.0f};

    // left top corner of small tile (1* 4) in Tile
    int colIdx = tx * TN;

    for (int bk = 0; bk < K; bk += BK) {
        // load A to shared memory
        for (int i = 0; i < tileY; i += sA_stride_size) {
            float4 temp = reinterpret_cast<float4 *>(&A[(i + rowIdx_sA) * K + colIdx_sA * 4])[0];
            sA_trans[colIdx_sA * 4 + 0][i + rowIdx_sA] = temp.x;
            sA_trans[colIdx_sA * 4 + 1][i + rowIdx_sA] = temp.y;
            sA_trans[colIdx_sA * 4 + 2][i + rowIdx_sA] = temp.z;
            sA_trans[colIdx_sA * 4 + 3][i + rowIdx_sA] = temp.w;
        }

        // load B to shared memory
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

            #pragma unroll
            for (int i = 0; i < TM; i++) {    // row
                avalue[i] = sA_trans[e][i * blockDim.y + ty];
            }

            #pragma unroll
            for (int i = 0; i < TN; i++) {    // col
                bvalue[i] = sB[e][colIdx + i];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {    // row
                #pragma unroll
                for (int j = 0; j < TN; j++) {   // col
                    cvalue[i][j] += avalue[i] * bvalue[j];
                }
            }
        }
    }

    // write Cvalue to C
    #pragma unroll
    for (int i = 0; i < TM; i++) {    // row
        // load C from global memory
        float4 temp = reinterpret_cast<float4 *>(&C[(i * 4 + ty) * N + colIdx])[0];
        // write C to global memory
        temp.x = cvalue[i][0];
        temp.y = cvalue[i][1];
        temp.z = cvalue[i][2];
        temp.w = cvalue[i][3];
        reinterpret_cast<float4 *>(&C[(i * 4 + ty) * N + colIdx])[0] = temp;
    }
}

void sgemm_reorder(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);
    // printf("dimBlock.x = %d, dimBlock.y = %d\n", dimBlock.x, dimBlock.y);

    int tileX = BLOCKDIM_X * TN;
    int tileY = BLOCKDIM_Y * TM;
    dim3 dimGrid((N + tileX - 1) / tileX, (M + tileY - 1) / tileY);
    // printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y);

    // int num_sms;
    // cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    // int n_threads = BLOCKDIM_X * BLOCKDIM_Y;
    // int num_blocks_per_sm;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, sgemm_reorder_kernel, n_threads, 0);
    // printf("num_blocks_per_sm: %d\n", num_blocks_per_sm);

    sgemm_reorder_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N, K, alpha, beta);

}