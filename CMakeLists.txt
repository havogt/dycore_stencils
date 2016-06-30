cmake_minimum_required( VERSION 2.8.8 )

project( HACKATHON )
enable_language( CXX )
enable_testing()

set (ADDITIONAL_CXX_FLAGS "-std=c++11")

find_package(CUDA REQUIRED)
set (ADDITIONAL_NVCC_FLAGS "-std=c++11")
set( CUDA_ARCH "sm_35" CACHE STRING "Compute capability for CUDA" )

SET(CUDA_SEPARABLE_COMPILATION ON)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
set(exe_LIBS "${CUDA_CUDART_LIBRARY}" "${exe_LIBS}" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${ADDITIONAL_NVCC_FLAGS}"
"-arch=${CUDA_ARCH}" "-Xcudafe" "--diag_suppress=dupl_calling_convention" "-Xcudafe" "--diag_suppress=code_is_unreachable" "-Xcudafe" "--diag_suppress=implicit_return_from_non_void_function" "-Xcudafe" "--diag_suppress=calling_convention_not_allowed" "-Xcudafe" "--diag_suppress=conflicting_calling_conventions")

# find GTest
find_package(GTest)
if (GTEST_FOUND)
  include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
else()
  message("GTest not found. GTest targets will not be compiled. (Try setting GTEST_ROOT in the env.)")
endif()
set(exe_LIBS -lpthread ${exe_LIBS} ${GTEST_BOTH_LIBRARIES})

cuda_add_executable( horizontal_diffusion horizontal_diffusion.cu horizontal_diffusion_benchmark.cu Options.cpp ${GTEST_FILES})
target_link_libraries(horizontal_diffusion ${exe_LIBS} )

#add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_cuda" "77 48 61")
#add_test( tests.${srcfile}_cuda ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/${srcfile}_cuda" "63 55 43")

