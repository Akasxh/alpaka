cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(_TARGET_NAME runtimeBlasGemm)
project(${_TARGET_NAME} LANGUAGES CXX)

set(BLAS_LIBS)
if(alpaka_ACC_GPU_CUDA_ONLY_MODE)
    find_package(CUDA REQUIRED)
    list(APPEND BLAS_LIBS ${CUDA_LIBRARIES} cublas)
endif()
if(alpaka_ACC_GPU_HIP_ONLY_MODE)
    find_package(rocblas REQUIRED CONFIG)
    list(APPEND BLAS_LIBS rocblas)
endif()

if(NOT BLAS_LIBS)
    message(WARNING "Skipping build of '${_TARGET_NAME}' because no GPU backend is enabled.")
    return()
endif()

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

alpaka_add_executable(
    ${_TARGET_NAME}
    src/runtimeBlasGemm.cpp)

target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka ${BLAS_LIBS})

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
