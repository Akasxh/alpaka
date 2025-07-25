#include "sofieblas_runtime_virtual.hpp"

#include <cstdlib> // getenv
#include <iostream>

int main(int argc, char** argv)
{
    sofieblas_backend backend = sofieblas_backend::CUDA; // default
    if(char const* e = std::getenv("SOFIE_BACKEND"))
        backend = (std::string(e) == "HIP") ? sofieblas_backend::HIP : sofieblas_backend::CUDA;

    auto blas = make_blas(backend); // runtime pick

    constexpr int m = 128, n = 128, k = 128, lda = m, ldb = k, ldc = m;
    float alpha = 1.0f, beta = 0.0f;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr; // dev ptrs (allocate as you wish)

    blas->gemm(false, false, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc);

    std::cout << "Ran GEMM on " << (backend == sofieblas_backend::CUDA ? "CUDA" : "HIP") << "\n";
}
