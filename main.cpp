// Include C++ header files.
#include <iostream>
#include <chrono>

// Include local CUDA header files.
#include "include/data_generation.cuh"
#include "include/solve_cpu.h"
#include "include/solve_gpu.cuh"
#include "include/solve_gpu2.cuh"

#define MIN_VALUE 0
#define MAX_VALUE 100
#define DIFF_EPS 0.00001f

using namespace std::chrono;

int main()
{
    int N = 100000;
    int dim = 8;
    int k = 32;
    float* tab = generateData(N, dim, MIN_VALUE, MAX_VALUE);
    
    auto start = high_resolution_clock::now();
    float* kCPU = solveCPU(tab, N, dim, k);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "CPU time duration: " << duration.count() / 1000 << "ms" << std::endl;
    
    start = high_resolution_clock::now();
    float* kGPU = solveGPU(tab, N, dim, k);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "GPU1 time duration: " << duration.count() / 1000 << "ms" << std::endl;

    start = high_resolution_clock::now();
    float* kGPU2 = solveGPU(tab, N, dim, k);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "GPU2 time duration: " << duration.count() / 1000 << "ms" << std::endl;

    int okCount = 0;
    for(int i = 0; i < k; i++)
    {
        std::cout << "Centroid number " << i << std::endl;
        bool isOk = true;
        for(int j = 0; j < dim; j++)
        {
            if(abs(kCPU[i + dim * j] - kGPU[i + dim * j]) > DIFF_EPS || abs(kCPU[i + dim * j] - kGPU2[i + dim * j]) > DIFF_EPS)
            {
                isOk = false;
                break;
            }
        }
        if(isOk)
        {
            std::cout << "\033[1;32mOK\033[0m" << std::endl;
            okCount++;
        }
        else
        {
            std::cout << "\033[1;31m---WRONG---" << std::endl;

            std::cout << "CPU: ";
            for(int j = 0; j < dim; j++)
            {
                std::cout << kCPU[i + dim * j] << " ";
            }

            std::cout << std::endl << "GPU1: ";
            for(int j = 0; j < dim; j++)
            {
                std::cout << kGPU[i + dim * j] << " ";
            }

            std::cout << std::endl << "GPU2: ";
            for(int j = 0; j < dim; j++)
            {
                std::cout << kGPU2[i + dim * j] << " ";
            }
            std::cout << "\033[0m" << std::endl;
        }
    }

    if(okCount == k)
    {
        std::cout << "\033[1;32mEvery centorid matches " << okCount << "/" << k << "\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[1;31mSome centroids are not the same " << okCount << "/" << k << "\033[0m" << std::endl;
    }

    return 0;
}