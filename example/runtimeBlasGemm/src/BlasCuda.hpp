#pragma once
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include "BLAS.hpp"
#include <alpaka/alpaka.hpp>
#include <cublas_v2.h>
#include <type_traits>
#include <cassert>
#include <iostream>

#define CUBLAS_CHECK(err)                                                                      \
    do {                                                                                      \
        if((err) != CUBLAS_STATUS_SUCCESS) {                                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ':' << __LINE__ << '\n';           \
            std::abort();                                                                     \
        }                                                                                     \
    } while (0)

class BlasCuda
{
public:
    explicit BlasCuda(alpaka::QueueCudaRtNonBlocking queue)
        : m_queue{queue}
    {
        CUBLAS_CHECK(cublasCreate(&m_handle));
        CUBLAS_CHECK(cublasSetStream(m_handle, m_queue.getNativeHandle()));
    }

    ~BlasCuda() { cublasDestroy(m_handle); }

    template <typename T>
    inline void gemm(
        cublasOperation_t transA,
        cublasOperation_t transB,
        int m, int n, int k,
        const T* alpha,
        const T* A, int lda,
        const T* B, int ldb,
        const T* beta,
        T* C, int ldc)
    {
        static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,
                      "BlasCuda::gemm only supports float or double.");

        if constexpr(std::is_same_v<T,float>)
        {
            CUBLAS_CHECK(cublasSgemm(
                m_handle, transA, transB,
                m, n, k,
                alpha, A, lda,
                       B, ldb,
                beta,  C, ldc));
        }
        else
        {
            CUBLAS_CHECK(cublasDgemm(
                m_handle, transA, transB,
                m, n, k,
                alpha, A, lda,
                       B, ldb,
                beta,  C, ldc));
        }
    }

private:
    alpaka::QueueCudaRtNonBlocking m_queue;
    cublasHandle_t                 m_handle{};
};

namespace traits
{
    template <>
    class Blas<alpaka::TagGpuCudaRt>
    {
    public:
        using Impl = BlasCuda;
    };
} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
