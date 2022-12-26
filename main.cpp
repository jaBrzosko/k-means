// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/data_generation.cuh"
#include "include/solve_cpu.h"

#define MIN_VALUE -100
#define MAX_VALUE 100

int main()
{
    int N = 1000;
    int dim = 2;
    int k = 4;
    float* tab = generateData(N, dim, MIN_VALUE, MAX_VALUE);

    float* kCPU = solveCPU(tab, N, dim, k);
    for(int i = 0; i < k; i++)
    {
        std::cout << "Centroid number " << i << std::endl;
        for(int j = 0; j < dim; j++)
        {
            std::cout << kCPU[i + dim * j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}