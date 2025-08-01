#include "blas.hpp"
#include <alpaka/alpaka.hpp>
#include <iostream>
#include <cmath>

int main()
{
    using Tag = alpaka::TagCpuSerial; // swap to TagGpuCudaRt for CUDA
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<Tag, Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::NonBlocking>;

    auto const devAcc = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0);
    Queue q(devAcc);

    auto const devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);

    constexpr int m = 3, n = 3, k = 3;
    using BufHost = alpaka::Buf<alpaka::DevCpu, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;

    BufHost A(alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(m*k)));
    BufHost B(alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(k*n)));
    BufHost C(alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(m*n)));
    BufHost gold(alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(m*n)));

    for(int i=0;i<m*k;++i) A[i] = static_cast<float>(i+1);
    for(int i=0;i<k*n;++i) B[i] = 0.f;
    for(int i=0;i<k && i<n;++i) B[i*n+i] = 1.f;
    for(int i=0;i<m*n;++i) C[i]=0.f, gold[i]=0.f;

    const float alpha=1.f, beta=0.f;
    for(int r=0;r<m;++r)
        for(int c=0;c<n;++c)
            for(int p=0;p<k;++p)
                gold[r*n+c] += alpha*A[r*k+p]*B[p*n+c];

    BufAcc dA(alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(m*k)));
    BufAcc dB(alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(k*n)));
    BufAcc dC(alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(m*n)));

    alpaka::memcpy(q, dA, A);
    alpaka::memcpy(q, dB, B);
    alpaka::memset(q, dC, std::uint8_t{0});

    Blas<Tag> blas{q};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    auto op = CUBLAS_OP_N;
#else
    auto op = CblasNoTrans;
#endif
    blas.gemm(op, op,
              m, n, k,
              &alpha,
              alpaka::getPtrNative(dA), k,
              alpaka::getPtrNative(dB), n,
              &beta,
              alpaka::getPtrNative(dC), n);
    alpaka::wait(q);

    alpaka::memcpy(q, C, dC);
    alpaka::wait(q);

    std::cout << "Gold result:\n";
    for(int r=0;r<m;++r){ for(int c=0;c<n;++c) std::cout<<gold[r*n+c]<<' '; std::cout<<'\n'; }
    std::cout << "\nComputed result:\n";
    for(int r=0;r<m;++r){ for(int c=0;c<n;++c) std::cout<<C[r*n+c]<<' '; std::cout<<'\n'; }

    bool ok = true;
    for(int i=0;i<m*n;++i) if(std::fabs(C[i]-gold[i])>1e-5f) ok=false;
    return ok?0:1;
}
