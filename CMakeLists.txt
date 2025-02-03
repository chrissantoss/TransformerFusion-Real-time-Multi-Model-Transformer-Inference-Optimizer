cmake_minimum_required(VERSION 3.18)
project(transformer_fusion CXX)

set(CMAKE_CXX_STANDARD 17)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Add executable with .cpp files instead of .cu
add_executable(transformer_demo
    src/demo.cpp  # Changed from .cu
    src/fusion_kernel.cpp  # Changed from .cu
    src/transformer_fusion.cpp  # Add this line
)

# Enable compiler optimizations
target_compile_options(transformer_demo PRIVATE -O3) 