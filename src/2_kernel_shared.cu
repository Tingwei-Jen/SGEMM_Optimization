#include "sgemm.h"
#include <stdio.h>
#define BLOCK_DIM 16

__global__ void sgemm_shared_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // local thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // move A, B, C to current tile position
    A += blockIdx.y * BLOCK_DIM * K;
    B += blockIdx.x * BLOCK_DIM;
    C += blockIdx.y * BLOCK_DIM * N + blockIdx.x * BLOCK_DIM;

    __shared__ float sA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sB[BLOCK_DIM][BLOCK_DIM];

    float sum = 0.f;

    #pragma unroll
    for (int bk = 0; bk < K; bk += BLOCK_DIM) {

        sA[ty][tx] = A[ty * K + tx];  // load A to shared memory
        sB[ty][tx] = B[ty * N + tx];  // load B to shared memory

        __syncthreads();

        A += BLOCK_DIM;
        B += BLOCK_DIM * N;

        #pragma unroll
        for (int e = 0; e < BLOCK_DIM; ++e) {
            float a = sA[ty][e];
            float b = sB[e][tx];
            sum += a * b;
        }

	    __syncthreads();
    }

    C[ty * N + tx] = alpha * sum + beta * C[ty * N + tx];
}

void sgemm_shared(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    sgemm_shared_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}