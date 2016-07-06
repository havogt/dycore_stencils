
cuda_add_executable( fast_waves_uv  fast_waves_uv_benchmark.cu ../Options.cpp  ${COMMON_INC} ${GTEST_FILES})
target_link_libraries(fast_waves_uv ${exe_LIBS} )

