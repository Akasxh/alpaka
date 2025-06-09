#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>

//! Kernel performing multiplication of two fixed 2x2 matrices
struct MatrixMulKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* A, T const* B, T* C) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if(idx < 4)
        {
            auto const row = idx / 2;
            auto const col = idx % 2;
            T sum = A[row * 2 + 0] * B[0 * 2 + col] + A[row * 2 + 1] * B[1 * 2 + col];
            C[idx] = sum;
        }
    }
};

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
auto example(TAccTag const&) -> int
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
    BufHost bufHostC(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostSum(alpaka::allocBuf<Data, Idx>(devHost, alpaka::Vec<Dim, Idx>(1)));

    // initialize 2x2 matrices
    bufHostA[0] = 1.f; bufHostA[1] = 2.f; bufHostA[2] = 3.f; bufHostA[3] = 4.f;
    bufHostB[0] = 5.f; bufHostB[1] = 6.f; bufHostB[2] = 7.f; bufHostB[3] = 8.f;
    bufHostC[0] = bufHostC[1] = bufHostC[2] = bufHostC[3] = 0.f;
    bufHostSum[0] = 0.f;

    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccC(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccSum(alpaka::allocBuf<Data, Idx>(devAcc, alpaka::Vec<Dim, Idx>(1)));

    alpaka::memcpy(queue, bufAccA, bufHostA);
    alpaka::memcpy(queue, bufAccB, bufHostB);
    alpaka::memcpy(queue, bufAccC, bufHostC);
    alpaka::memcpy(queue, bufAccSum, bufHostSum);

    MatrixMulKernel mulKernel;
    SumKernel sumKernel;

    alpaka::KernelCfg<Acc> cfgMul = {extent, Idx{1}};
    auto const workDivMul = alpaka::getValidWorkDiv(cfgMul, devAcc, mulKernel, alpaka::getPtrNative(bufAccA), alpaka::getPtrNative(bufAccB), alpaka::getPtrNative(bufAccC));

    alpaka::KernelCfg<Acc> cfgSum = {alpaka::Vec<Dim, Idx>(1), Idx{1}};
    auto const workDivSum = alpaka::getValidWorkDiv(cfgSum, devAcc, sumKernel, alpaka::getPtrNative(bufAccC), alpaka::getPtrNative(bufAccSum));

    auto const startTotal = std::chrono::high_resolution_clock::now();

    // multiplication kernel
    alpaka::wait(queue);
    auto startMul = std::chrono::high_resolution_clock::now();
    alpaka::exec<Acc>(queue, workDivMul, mulKernel, alpaka::getPtrNative(bufAccA), alpaka::getPtrNative(bufAccB), alpaka::getPtrNative(bufAccC));
    alpaka::wait(queue);
    auto endMul = std::chrono::high_resolution_clock::now();

    // addition kernel
    auto startAdd = std::chrono::high_resolution_clock::now();
    alpaka::exec<Acc>(queue, workDivSum, sumKernel, alpaka::getPtrNative(bufAccC), alpaka::getPtrNative(bufAccSum));
    alpaka::wait(queue);
    auto endAdd = std::chrono::high_resolution_clock::now();

    alpaka::memcpy(queue, bufHostC, bufAccC);
    alpaka::memcpy(queue, bufHostSum, bufAccSum);
    alpaka::wait(queue);

    auto endTotal = std::chrono::high_resolution_clock::now();

    std::cout << "Time for multiplication: " << std::chrono::duration<double>(endMul - startMul).count() << "s" << std::endl;
    std::cout << "Time for addition: " << std::chrono::duration<double>(endAdd - startAdd).count() << "s" << std::endl;
    std::cout << "Total time: " << std::chrono::duration<double>(endTotal - startTotal).count() << "s" << std::endl;

    std::cout << "Result matrix:" << std::endl;
    std::cout << bufHostC[0] << " " << bufHostC[1] << std::endl;
    std::cout << bufHostC[2] << " " << bufHostC[3] << std::endl;
    std::cout << "Sum of result elements: " << bufHostSum[0] << std::endl;

    return EXIT_SUCCESS;
}

int main()
{
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}

