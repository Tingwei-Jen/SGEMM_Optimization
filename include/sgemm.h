#ifndef SGEMM_H
#define SGEMM_H
#include <cuda_fp16.h>

/**
 * @brief Performs single-precision general matrix multiplication (SGEMM) on the GPU.
 *
 * This function computes the matrix product of two matrices A and B, and adds the result to matrix C,
 * scaled by the factors alpha and beta respectively. The operation performed is:
 * 
 *     C = alpha * (A * B) + beta * C
 *
 * @param A Pointer to the first input matrix (of size M x K).
 * @param B Pointer to the second input matrix (of size K x N).
 * @param C Pointer to the output matrix (of size M x N).
 * @param N Number of rows in matrices A and C.
 * @param M Number of columns in matrices B and C.
 * @param K Number of columns in matrix A and number of rows in matrix B.
 * @param alpha Scalar multiplier for the product of matrices A and B.
 * @param beta Scalar multiplier for the matrix C.
 */
void sgemm_naive(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_shared(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_tile2d(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_padding(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_register(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_float4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_float4_prefetch(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_float4_tuning(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);

// void sgemm_reorder(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
// void sgemm_outer_product(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
// void sgemm_outer_product_float4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);

#endif // SGEMM_H
