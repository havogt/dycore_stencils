
cuda_add_executable( horizontal_diffusion horizontal_diffusion.cu horizontal_diffusion_benchmark.cu ../Options.cpp  ${COMMON_INC} ${GTEST_FILES})
target_link_libraries(horizontal_diffusion ${exe_LIBS} )

