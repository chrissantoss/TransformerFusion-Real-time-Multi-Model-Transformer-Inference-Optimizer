cmake_minimum_required(VERSION 3.18)
project(transformer_fusion CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Add this before find_package(CUDA REQUIRED)
set(CUDAToolkit_ROOT "/usr/local/cuda")  # or "/opt/homebrew/opt/cuda" for M1
find_package(CUDA REQUIRED)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Add executable
add_executable(transformer_demo
    src/demo.cu
    src/fusion_kernel.cu
)

# Set CUDA architecture
set_target_properties(transformer_demo PROPERTIES
    CUDA_ARCHITECTURES "75;80;86"  # Adjust based on your GPU
)

# Enable compiler optimizations
target_compile_options(transformer_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
) 