find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

target_sources(
    nnops PUBLIC
    device.cpp
)

add_library(
    nnops_cuda SHARED
    device.cu copy.cu
)

target_include_directories(nnops_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(nnops_cuda PRIVATE CUDA::cudart)
target_compile_options(nnops_cuda PRIVATE -arch=sm_50)
set_target_properties(nnops_cuda PROPERTIES CUDA_ARCHITECTURES "50")

target_link_libraries(nnops PRIVATE nnops_cuda)