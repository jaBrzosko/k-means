// Include C++ header files.
#include <iostream>
#include <chrono>
#include <iomanip>

// Include local CUDA header files.
#include "include/data_generation.cuh"
#include "include/solve_cpu.h"
#include "include/solve_gpu.cuh"
#include "include/solve_gpu2.cuh"
#include "include/analysis.h"

#define MIN_VALUE 0
#define MAX_VALUE 1
#define DIFF_EPS 0.001f
#define MAX_DIM 32

using namespace std::chrono;

float myAbs(float a)
{
    if(a < 0)
        return -a;
    return a;
}

int main(int argc, char** argv)
{
    int N = 100000;
    int dim = 8;
    int k = 8;

    if(argc > 1)
    {
        N = atoi(argv[1]);
        if(N <= 0)
        {
            std::cerr << "Parameter for N is not valid!" << std::endl;
            exit(1);
        }
    }
    
    if(argc > 2)
    {
        k = atoi(argv[2]);
        if(k <= 0)
        {
            std::cerr << "Parameter for k is not valid!" << std::endl;
            exit(1);
        }
    }
    
    if(argc > 3)
    {
        dim = atoi(argv[3]);
        if(dim <= 0 || dim > MAX_DIM)
        {
            std::cerr << "Parameter for n is not valid!" << std::endl;
            exit(1);
        }
    }

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

    std::cout << std::endl << "Parameters:" << std::endl;
    std::cout << "N=" << N << " k=" << k << " n=" << dim << std::endl;
    std::cout << std::endl;

    std::cout << "CPU time duration: " << durationCPU.count() / 1000 << "ms" << std::endl;
    std::cout << "GPU1 time duration: " << durationGPU1.count() / 1000 << "ms" << std::endl;
    std::cout << "GPU2 time duration: " << durationGPU2.count() / 1000 << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "CPU  squared error: " <<  std::fixed << std::setprecision(3) << squaredError(kCPU, tab, N, dim, k) << std::endl;
    std::cout << "GPU1 squared error: " << std::fixed << std::setprecision(3) << squaredError(kGPU, tab, N, dim, k) << std::endl;
    std::cout << "GPU2 squared error: " << std::fixed << std::setprecision(3) << squaredError(kGPU2, tab, N, dim, k) << std::endl;
    std::cout << std::endl;


    return 0;
}

