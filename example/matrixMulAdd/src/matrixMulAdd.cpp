#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <cublas_v2.h>
#include <chrono>
#include <iostream>
#include <cstdlib>


//! Kernel summing all four elements of a 2x2 matrix
struct SumKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* C, T* result) const
    {
        T sum = T{0};
        for(int i = 0; i < 4; ++i)
        {
            sum += C[i];
        }
        result[0] = sum;
    }
};

template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&, int iterations) -> int
{
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    using DevAcc = alpaka::Dev<Acc>;
    using DevHost = alpaka::DevCpu;

    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    Queue queue(devAcc);

    Idx const numElements = 4;
    alpaka::Vec<Dim, Idx> const extent(numElements);

    using Data = float;
    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;

    BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostB(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostC1(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostC2(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostSum1(alpaka::allocBuf<Data, Idx>(devHost, alpaka::Vec<Dim, Idx>(1)));
    BufHost bufHostSum2(alpaka::allocBuf<Data, Idx>(devHost, alpaka::Vec<Dim, Idx>(1)));

    // initialize 2x2 matrices
    bufHostA[0] = 1.f; bufHostA[1] = 2.f; bufHostA[2] = 3.f; bufHostA[3] = 4.f;
    bufHostB[0] = 5.f; bufHostB[1] = 6.f; bufHostB[2] = 7.f; bufHostB[3] = 8.f;
    bufHostC1[0] = bufHostC1[1] = bufHostC1[2] = bufHostC1[3] = 0.f;
    bufHostC2[0] = bufHostC2[1] = bufHostC2[2] = bufHostC2[3] = 0.f;
    bufHostSum1[0] = 0.f;
    bufHostSum2[0] = 0.f;

    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccC1(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccC2(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccSum1(alpaka::allocBuf<Data, Idx>(devAcc, alpaka::Vec<Dim, Idx>(1)));
    BufAcc bufAccSum2(alpaka::allocBuf<Data, Idx>(devAcc, alpaka::Vec<Dim, Idx>(1)));

    alpaka::memcpy(queue, bufAccA, bufHostA);
    alpaka::memcpy(queue, bufAccB, bufHostB);
    alpaka::memcpy(queue, bufAccC1, bufHostC1);
    alpaka::memcpy(queue, bufAccC2, bufHostC2);
    alpaka::memcpy(queue, bufAccSum1, bufHostSum1);
    alpaka::memcpy(queue, bufAccSum2, bufHostSum2);

    SumKernel sumKernel;

    alpaka::KernelCfg<Acc> cfgSum = {alpaka::Vec<Dim, Idx>(1), Idx{1}};
    auto const workDivSum = alpaka::getValidWorkDiv(cfgSum, devAcc, sumKernel, alpaka::getPtrNative(bufAccC1), alpaka::getPtrNative(bufAccSum1));

    auto stream = alpaka::getNativeHandle(queue);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    double totalMul1 = 0.0, totalAdd1 = 0.0, totalMul2 = 0.0, totalAdd2 = 0.0, totalRuntime = 0.0;

    for(int i = 0; i < iterations; ++i)
    {
        auto startTotal = std::chrono::high_resolution_clock::now();

        alpaka::wait(queue);
        auto startMul1 = std::chrono::high_resolution_clock::now();
        float alpha1 = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    2,
                    2,
                    2,
                    &alpha1,
                    alpaka::getPtrNative(bufAccA),
                    2,
                    alpaka::getPtrNative(bufAccB),
                    2,
                    &beta,
                    alpaka::getPtrNative(bufAccC1),
                    2);
        alpaka::wait(queue);
        auto endMul1 = std::chrono::high_resolution_clock::now();

        auto startAdd1 = std::chrono::high_resolution_clock::now();
        alpaka::exec<Acc>(queue, workDivSum, sumKernel, alpaka::getPtrNative(bufAccC1), alpaka::getPtrNative(bufAccSum1));
        alpaka::wait(queue);
        auto endAdd1 = std::chrono::high_resolution_clock::now();

        alpaka::memcpy(queue, bufHostSum1, bufAccSum1);
        alpaka::wait(queue);
        float alpha2 = bufHostSum1[0];

        auto startMul2 = std::chrono::high_resolution_clock::now();
        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    2,
                    2,
                    2,
                    &alpha2,
                    alpaka::getPtrNative(bufAccA),
                    2,
                    alpaka::getPtrNative(bufAccB),
                    2,
                    &beta,
                    alpaka::getPtrNative(bufAccC2),
                    2);
        alpaka::wait(queue);
        auto endMul2 = std::chrono::high_resolution_clock::now();

        auto startAdd2 = std::chrono::high_resolution_clock::now();
        alpaka::exec<Acc>(queue, workDivSum, sumKernel, alpaka::getPtrNative(bufAccC2), alpaka::getPtrNative(bufAccSum2));
        alpaka::wait(queue);
        auto endAdd2 = std::chrono::high_resolution_clock::now();

        auto endTotal = std::chrono::high_resolution_clock::now();

        totalMul1 += std::chrono::duration<double>(endMul1 - startMul1).count();
        totalAdd1 += std::chrono::duration<double>(endAdd1 - startAdd1).count();
        totalMul2 += std::chrono::duration<double>(endMul2 - startMul2).count();
        totalAdd2 += std::chrono::duration<double>(endAdd2 - startAdd2).count();
        totalRuntime += std::chrono::duration<double>(endTotal - startTotal).count();
    }

    alpaka::memcpy(queue, bufHostC1, bufAccC1);
    alpaka::memcpy(queue, bufHostC2, bufAccC2);
    alpaka::memcpy(queue, bufHostSum1, bufAccSum1);
    alpaka::memcpy(queue, bufHostSum2, bufAccSum2);
    alpaka::wait(queue);

    cublasDestroy(handle);

    std::cout << "Average time GEMM1: " << totalMul1 / iterations << "s" << std::endl;
    std::cout << "Average time Add1: " << totalAdd1 / iterations << "s" << std::endl;
    std::cout << "Average time GEMM2: " << totalMul2 / iterations << "s" << std::endl;
    std::cout << "Average time Add2: " << totalAdd2 / iterations << "s" << std::endl;
    std::cout << "Average total time: " << totalRuntime / iterations << "s" << std::endl;

    std::cout << "Result matrix after GEMM2:" << std::endl;
    std::cout << bufHostC2[0] << " " << bufHostC2[1] << std::endl;
    std::cout << bufHostC2[2] << " " << bufHostC2[3] << std::endl;
    std::cout << "Sum1: " << bufHostSum1[0] << ", Sum2: " << bufHostSum2[0] << std::endl;

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    int iterations = 100;
    if(argc > 1)
    {
        iterations = std::atoi(argv[1]);
    }
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag, iterations); });
}

