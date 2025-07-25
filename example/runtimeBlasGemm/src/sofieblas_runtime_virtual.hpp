#include <alpaka/alpaka.hpp>

#include <cublas_v2.h>
#include <rocblas/rocblas.h>

#include <memory>

enum class sofieblas_backend
{
    CUDA,
    HIP
};

struct IBlasRuntime
{
    virtual ~IBlasRuntime() = default;
    virtual void gemm(
        bool tA,
        bool tB,
        int m,
        int n,
        int k,
        float const& alpha,
        float const* A,
        int lda,
        float const* B,
        int ldb,
        float const& beta,
        float* C,
        int ldc)
        = 0;
};

class BlasCuda final : public IBlasRuntime
{
    cublasHandle_t h_{};

public:
    BlasCuda()
    {
        cublasCreate(&h_);
    }

    ~BlasCuda() override
    {
        cublasDestroy(h_);
    }

    void gemm(
        bool tA,
        bool tB,
        int m,
        int n,
        int k,
        float const& a,
        float const* A,
        int lda,
        float const* B,
        int ldb,
        float const& b,
        float* C,
        int ldc) override
    {
        cublasSgemm(
            h_,
            tA ? CUBLAS_OP_T : CUBLAS_OP_N,
            tB ? CUBLAS_OP_T : CUBLAS_OP_N,
            m,
            n,
            k,
            &a,
            A,
            lda,
            B,
            ldb,
            &b,
            C,
            ldc);
    }
};

class BlasHip final : public IBlasRuntime
{
    rocblas_handle h_{};

public:
    BlasHip()
    {
        rocblas_create_handle(&h_);
    }

    ~BlasHip() override
    {
        rocblas_destroy_handle(h_);
    }

    void gemm(
        bool tA,
        bool tB,
        int m,
        int n,
        int k,
        float const& a,
        float const* A,
        int lda,
        float const* B,
        int ldb,
        float const& b,
        float* C,
        int ldc) override
    {
        rocblas_sgemm(
            h_,
            tA ? rocblas_operation_transpose : rocblas_operation_none,
            tB ? rocblas_operation_transpose : rocblas_operation_none,
            m,
            n,
            k,
            &a,
            A,
            lda,
            B,
            ldb,
            &b,
            C,
            ldc);
    }
};

inline std::unique_ptr<IBlasRuntime> make_blas(sofieblas_backend b)
{
    if(b == sofieblas_backend::CUDA)
        return std::make_unique<BlasCuda>();
    return std::make_unique<BlasHip>();
}
