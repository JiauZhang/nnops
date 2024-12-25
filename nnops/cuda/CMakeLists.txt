find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

target_sources(
    nnops PUBLIC
    device.cpp
)

target_include_directories(nnops PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(nnops PRIVATE CUDA::cudart)

add_library(
    nnops_cuda SHARED
    dummy.cu
)

target_compile_options(nnops_cuda PUBLIC -arch=sm_50)
set_target_properties(nnops_cuda PROPERTIES CUDA_ARCHITECTURES "50")
