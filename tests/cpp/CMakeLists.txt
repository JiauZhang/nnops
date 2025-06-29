cmake_minimum_required(VERSION 3.15)

if (WIN32)
    set(GTest_DIR "F:/vcpkg/installed/x64-windows/share/gtest")
endif()

project(nnops_cpp_tests)

enable_testing()
find_package(GTest CONFIG REQUIRED)
set(GTEST_LIBS GTest::gtest GTest::gtest_main)

set(NNOPS_CPP_TEST_SOURCES
    test_tensor.cpp
)

add_executable(cpp_tests ${NNOPS_CPP_TEST_SOURCES})
target_link_libraries(cpp_tests PUBLIC nnops ${GTEST_LIBS})
