cmake_minimum_required(VERSION 3.18)
project(transformer_fusion CXX)

set(CMAKE_CXX_STANDARD 17)

# Enable Objective-C++
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Add executable
add_executable(transformer_demo
    src/demo.cpp
    src/fusion_kernel.mm
    src/transformer_fusion.cpp
)

# Enable compiler optimizations
target_compile_options(transformer_demo PRIVATE -O3)

# Add Metal framework
find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(MPS_LIBRARY MetalPerformanceShaders)

target_link_libraries(transformer_demo
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${MPS_LIBRARY}
)

# Add Metal shader compilation
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/transformer.metallib
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/src/transformer.metal -o ${CMAKE_CURRENT_BINARY_DIR}/transformer.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/transformer.air -o ${CMAKE_CURRENT_BINARY_DIR}/transformer.metallib
)

add_custom_target(metal_shader ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/transformer.metallib) 