#include <iostream>
#include <Eigen/Dense>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
#include "sgemm.h"

class SGEMMProfiler {
public:
    SGEMMProfiler(const int M, const int N, const int K, const int testIter);
    ~SGEMMProfiler();

    typedef void (*sgemm_function)(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
    void profiling(sgemm_function sgemm_impl);

private:
    
    cudaEvent_t m_start, m_stop;
    int m_M, m_N, m_K;
    int m_testIter;
	float *m_hA, *m_hB, *m_hC, *m_hC_cublas;
	float *m_dA, *m_dB, *m_dC, *m_dC_cublas;
};

SGEMMProfiler::SGEMMProfiler(const int M, const int N, const int K, const int testIter)
    : m_M(M), m_N(N), m_K(K), m_testIter(testIter) {
	
    checkCudaErrors(cudaEventCreate(&m_start));
    checkCudaErrors(cudaEventCreate(&m_stop));

    // allocation of host memory
	m_hA = (float *)malloc(m_M * m_K * sizeof(float));
	m_hB = (float *)malloc(m_K * m_N * sizeof(float));
	m_hC = (float *)malloc(m_M * m_N * sizeof(float));
    m_hC_cublas = (float *)malloc(m_M * m_N * sizeof(float));

	// allocation of device memory
	checkCudaErrors(cudaMalloc((void **)&m_dA, m_M * m_K * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&m_dB, m_K * m_N * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&m_dC, m_M * m_N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&m_dC_cublas, m_M * m_N * sizeof(float)));

    // ramdom init
	for (int i = 0; i < m_M * m_K; ++i) {
		// random value between -1 and 1
        m_hA[i] = 2.f * (rand() / (float)RAND_MAX) - 1.f;
	}

	for (int i = 0; i < m_K * m_N; ++i) {
		// random value between -1 and 1
        m_hB[i] = 2.f * (rand() / (float)RAND_MAX) - 1.f;
	}

	for (int i = 0; i < m_M * m_N; ++i) {
		// random value between -1 and 1
        m_hC[i] = 2.f * (rand() / (float)RAND_MAX) - 1.f;
	}    

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(m_dA, m_hA, m_M * m_K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_dB, m_hB, m_K * m_N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_dC, m_hC, m_M * m_N * sizeof(float), cudaMemcpyHostToDevice));
}

SGEMMProfiler::~SGEMMProfiler() {
    // free host memory
    free(m_hA);
    free(m_hB);
    free(m_hC);

    // free device memory
    checkCudaErrors(cudaFree(m_dA));
    checkCudaErrors(cudaFree(m_dB));
    checkCudaErrors(cudaFree(m_dC));

    // finalize cuda event
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
}

void SGEMMProfiler::profiling(sgemm_function sgemm_impl) {

    // warm-up
    sgemm_impl(m_dA, m_dB, m_dC, m_M, m_N, m_K, 1.f, 0.f);

    // start cuda event
    cudaEventRecord(m_start);
    for (int i = 0; i < m_testIter; i++) {
        sgemm_impl(m_dA, m_dB, m_dC, m_M, m_N, m_K, 1.f, 0.f);
    }

    // event record
    cudaEventRecord(m_stop);
    checkCudaErrors(cudaEventSynchronize(m_stop));

    // get elapsed time
    float elapsed_time_msed_event = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed_event, m_start, m_stop);
    elapsed_time_msed_event /= 100;
    printf("CUDA event estimated - elapsed %.6f ms \n", elapsed_time_msed_event);

    // get gflops
    float gflops = 2.f * (float)m_M * (float)m_N * (float)m_K / (elapsed_time_msed_event / 1000.f) / 1e9;
    printf("GFLOPS: %.6f\n", gflops);

    // copy data from device to host
    cudaMemcpy(m_hC, m_dC, m_M * m_N * sizeof(float), cudaMemcpyDeviceToHost);

    // cublas comparison ///////////////////////////////////////////////////////////
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // start cuda event
    cudaEventRecord(m_start);

    // Perform matrix multiplication with CUBLAS
	float alpha = 1.f;
	float beta = 0.f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_M, m_N, m_K, &alpha, m_dB, m_M, m_dA, m_K, &beta, m_dC_cublas, m_M);

    // event record
    cudaEventRecord(m_stop);
    checkCudaErrors(cudaEventSynchronize(m_stop));

    // // Calculate the elapsed time
    float elapsed_time_msed_event_cublas = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed_event_cublas, m_start, m_stop);
    printf("CUBLAS event estimated - elapsed %.6f ms \n", elapsed_time_msed_event_cublas);

    float gflops_cublas = 2.f * (float)m_M * (float)m_N * (float)m_K / (elapsed_time_msed_event_cublas / 1000.f) / 1e9;
    printf("GFLOPS CUBLAS: %.6f\n", gflops_cublas);

    // copy data from device to host
    cudaMemcpy(m_hC_cublas, m_dC_cublas, m_M * m_N * sizeof(float), cudaMemcpyDeviceToHost);

    // compare the results
    float error = 0.f;

    for (int i = 0; i < m_M * m_N; i++) {
        // printf("m_hC[%d]: %.6f, m_hC_cublas[%d]: %.6f\n", i, m_hC[i], i, m_hC_cublas[i]);
        error += m_hC[i] - m_hC_cublas[i];
    }

    printf("Error: %.6f\n", error);
    cublasDestroy(handle);
}


int main(int argc, char* argv[]) {

    int start = 9;
    int end = 12;

    for (int i = start; i <= end; i++) {
        int M = 1 << i;
        int N = 1 << i;
        int K = 1 << i;

        printf("%d x %d x %d\n", M, N, K);
        SGEMMProfiler profiler(M, N, K, 100);

        // printf("SGEMM Naive: \n");
        // profiler.profiling(sgemm_naive);
        // printf("\n");

        // printf("SGEMM Shared: \n");
        // profiler.profiling(sgemm_shared);
        // printf("\n");

        // printf("SGEMM Tile2d: \n");
        // profiler.profiling(sgemm_tile2d);
        // printf("\n");

        // printf("SGEMM Padding: \n");
        // profiler.profiling(sgemm_padding);
        // printf("\n");

        // printf("SGEMM Register: \n");
        // profiler.profiling(sgemm_register);
        // printf("\n");

        printf("SGEMM Float4: \n");
        profiler.profiling(sgemm_float4);
        printf("\n");

        printf("SGEMM Float4 + Prefetch: \n");
        profiler.profiling(sgemm_float4_prefetch);
        printf("\n");

        printf("SGEMM float4 tuning: \n");
        profiler.profiling(sgemm_float4_tuning);
        printf("\n");
    }

    return 0;
}
