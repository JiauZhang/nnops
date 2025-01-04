cmake_minimum_required(VERSION 3.15)

project(nnops_cpp_tests)

find_package(GTest REQUIRED)

set(NNOPS_CPP_TEST_SOURCES
    test_tensor.cpp
)

add_executable(cpp_tests ${NNOPS_CPP_TEST_SOURCES})
target_link_libraries(cpp_tests PUBLIC nnops nnops_cuda GTest::GTest GTest::Main)
