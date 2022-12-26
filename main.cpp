// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include "include/data_generation.cuh"

#define MIN_VALUE -100
#define MAX_VALUE 100

int main()
{
    int N = 1000;
    int dim = 3;
    float* tab = generateData(N, dim, MIN_VALUE, MAX_VALUE);

    return 0;
}