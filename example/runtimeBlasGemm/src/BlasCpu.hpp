#pragma once
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

#include "BLAS.hpp"
#include <alpaka/alpaka.hpp>
#ifdef USE_CBLAS
#    include <cblas.h>
#else
#    include <cassert>
     enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
#endif
#include <type_traits>

class BlasCpu
{
public:
    BlasCpu(auto) {}

    template <typename T>
    inline void gemm(
        CBLAS_TRANSPOSE transA,
        CBLAS_TRANSPOSE transB,
        int m, int n, int k,
        const T* alpha,
        const T* A, int lda,
        const T* B, int ldb,
        const T* beta,
        T* C, int ldc)
    {
#ifdef USE_CBLAS
        if constexpr(std::is_same_v<T,float>)
            cblas_sgemm(CblasRowMajor, transA, transB,
                         m, n, k,
                         *alpha, A, lda,
                                  B, ldb,
                         *beta,  C, ldc);
        else
            cblas_dgemm(CblasRowMajor, transA, transB,
                         m, n, k,
                         *alpha, A, lda,
                                  B, ldb,
                         *beta,  C, ldc);
#else
        assert(transA == CblasNoTrans && transB == CblasNoTrans &&
               "Fallback only supports non-transposed matrices");
        for (int row = 0; row < m; ++row)
            for (int col = 0; col < n; ++col) {
                T acc = T{0};
                for (int p = 0; p < k; ++p)
                    acc += A[row*lda + p] * B[p*ldb + col];
                C[row*ldc + col] = (*alpha)*acc + (*beta)*C[row*ldc + col];
            }
#endif
    }
};

namespace traits
{
    template <>
    class Blas<alpaka::TagCpuSerial>
    {
    public:
        using Impl = BlasCpu;
    };
} // namespace traits

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
