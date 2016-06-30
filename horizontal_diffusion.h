#pragma once

#include "Definitions.h"
#include <iostream>
#include <cuda.h>
#include "domain.hpp"
#include "repository.hpp"

__global__
void cukernel(Real* in, Real* out, const int, const int, const int);

void launch_kernel(repository& repo);


