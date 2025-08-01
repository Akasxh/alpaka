#pragma once

namespace traits
{
    template <typename TTag>
    class Blas;         // forward declaration
} // namespace traits

template <typename TTag>
using Blas = typename traits::Blas<TTag>::Impl;   // convenient alias
