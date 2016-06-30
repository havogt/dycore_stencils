
cuda_add_executable( vertical_advection vertical_advection.cu vertical_advection_benchmark.cu ../Options.cpp  ${COMMON_INC} ${GTEST_FILES})
target_link_libraries(vertical_advection ${exe_LIBS} )

