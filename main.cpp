// Include C++ header files.
#include <iostream>
#include <chrono>

// Include local CUDA header files.
#include "include/data_generation.cuh"
#include "include/solve_cpu.h"
#include "include/solve_gpu.cuh"
#include "include/solve_gpu2.cuh"

#define MIN_VALUE 0
#define MAX_VALUE 1
#define DIFF_EPS 0.001f

using namespace std::chrono;

float myAbs(float a)
{
    if(a < 0)
        return -a;
    return a;
}

int main()
{
    int N = 100000;
    int dim = 32;
    int k = 32;
    float* tab = generateData(N, dim, MIN_VALUE, MAX_VALUE);
    
    auto start = high_resolution_clock::now();
    float* kCPU = solveCPU(tab, N, dim, k);
    auto stop = high_resolution_clock::now();
    auto durationCPU = duration_cast<microseconds>(stop - start);
    
    start = high_resolution_clock::now();
    float* kGPU = solveGPU(tab, N, dim, k);
    stop = high_resolution_clock::now();
    auto durationGPU1 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    float* kGPU2 = solveGPU2(tab, N, dim, k);
    stop = high_resolution_clock::now();
    auto durationGPU2 = duration_cast<microseconds>(stop - start);

    int okCount = 0;
    for(int i = 0; i < k; i++)
    {
        std::cout << "Centroid number " << i << std::endl;
        bool isOk = true;
        for(int j = 0; j < dim; j++)
        {
            if(myAbs(kCPU[i + k * j] - kGPU[i + k * j]) > DIFF_EPS || myAbs(kCPU[i + k * j] - kGPU2[i + k * j]) > DIFF_EPS)
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
                std::cout << kCPU[i + k * j] << " ";
            }

            std::cout << std::endl << "GPU1: ";
            for(int j = 0; j < dim; j++)
            {
                std::cout << kGPU[i + k * j] << " ";
            }

            std::cout << std::endl << "GPU2: ";
            for(int j = 0; j < dim; j++)
            {
                std::cout << kGPU2[i + k * j] << " ";
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
    std::cout << std::endl;
    std::cout << "CPU time duration: " << durationCPU.count() / 1000 << "ms" << std::endl;
    std::cout << "GPU1 time duration: " << durationGPU1.count() / 1000 << "ms" << std::endl;
    std::cout << "GPU2 time duration: " << durationGPU2.count() / 1000 << "ms" << std::endl;
    std::cout << std::endl;

    return 0;
}