cmake_minimum_required(VERSION 3.24)
project(squid5090 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_ARCH "120a" CACHE STRING "Target CUDA architecture (default: 120a for RTX 5090 Blackwell)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

find_package(CUDAToolkit REQUIRED)

add_executable(squid5090
    src/main.cpp
    src/safetensors.cpp
    src/tokenizer.cpp
    src/server.cpp
    src/weights.cu
    src/kernels_gemv.cu
    src/kernels_inkling.cu
    src/kernels_dspark.cu
    src/engine.cu
)

find_package(Threads REQUIRED)

target_include_directories(squid5090 PRIVATE src)

target_compile_options(squid5090 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo --expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
)

target_link_libraries(squid5090 PRIVATE
    CUDA::cudart
    CUDA::cublas
    Threads::Threads
)
