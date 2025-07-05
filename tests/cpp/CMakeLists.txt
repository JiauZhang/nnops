cmake_minimum_required(VERSION 3.15)

if (WIN32 AND NOT DEFINED GTest_DIR)
    set(GTest_DIR "$ENV{VCPKG_ROOT}/installed/x64-windows/share/gtest")
    message(STATUS "GTest_DIR = ${GTest_DIR}")
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
