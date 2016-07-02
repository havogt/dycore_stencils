#pragma once

#include "../Definitions.hpp"
#include <iostream>
#include <cuda.h>
#include "../domain.hpp"
#include "../repository.hpp"
#include "../timer_cuda.hpp"

__global__
void cukernel(Real* in, Real* out, const int, const int, const int);

void launch_kernel(repository& repo, timer_cuda* );


