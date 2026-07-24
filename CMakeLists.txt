# Alternative to the Makefile for environments that prefer CMake:
#   cmake -S . -B build && cmake --build build -j
cmake_minimum_required(VERSION 3.18)
project(cuda_npp_edge_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(edge_pipeline
  src/main.cc
  src/cli.cc
  src/image.cc
  src/otsu.cc
  src/gpu_pipeline.cu
  src/kernels.cu
)

target_include_directories(edge_pipeline PRIVATE include src third_party)
target_compile_options(edge_pipeline PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
target_link_libraries(edge_pipeline PRIVATE
  CUDA::nppc CUDA::nppisu CUDA::nppicc CUDA::nppif CUDA::nppim CUDA::nppist
  CUDA::cudart
)
set_target_properties(edge_pipeline PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# Host-only tests; these build and run without a GPU.
add_executable(host_tests
  tests/host_tests.cc
  src/cli.cc
  src/image.cc
  src/otsu.cc
)
target_include_directories(host_tests PRIVATE include third_party)
target_compile_options(host_tests PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME host_tests COMMAND host_tests)
