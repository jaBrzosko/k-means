// Include C++ header files.
#include <iostream>
#include <chrono>

// Include local CUDA header files.
#include "include/data_generation.cuh"
#include "include/solve_cpu.h"
#include "include/solve_gpu.cuh"

#define MIN_VALUE 0
#define MAX_VALUE 100

using namespace std::chrono;

int main()
{
    int N = 10000;
    int dim = 4;
    int k = 8;
    float* tab = generateData(N, dim, MIN_VALUE, MAX_VALUE);
    
    auto start = high_resolution_clock::now();
    float* kCPU = solveCPU(tab, N, dim, k);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "CPU time duration in microseconds: " << duration.count() << std::endl;
    
    start = high_resolution_clock::now();
    float* kGPU = solveGPU(tab, N, dim, k);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "GPU time duration in microseconds: " << duration.count() << std::endl;

    for(int i = 0; i < k; i++)
    {
        std::cout << "Centroid number " << i << std::endl;
        std::cout << "CPU: " << std::endl;
        for(int j = 0; j < dim; j++)
        {
            std::cout << kCPU[i + dim * j] << " ";
        }
        std::cout << std::endl << "GPU: " << std::endl;
        for(int j = 0; j < dim; j++)
        {
            std::cout << kGPU[i + dim * j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}