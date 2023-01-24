#include <cfloat>
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template<int dim>
__global__ void divNewTab(float* centroid, float* newCentroid, int* count, int k)
{
    unsigned int kIndex   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int dimIndex = threadIdx.y + blockIdx.y * blockDim.y;

    // Don't go over boundaries
    if(kIndex >= k || dimIndex >= dim) return;

    int cnt = count[kIndex];
    if(cnt != 0)
        centroid[kIndex + dimIndex * k] = newCentroid[kIndex + dimIndex * k] / cnt;
}

// if index changes value over neighbors update cellStart
__global__ void prepareCellStart(int* indices, int* cellStart, int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N) return;
    if(tid == 0)
        cellStart[indices[0]] = 0;
    else if(indices[tid] != indices[tid - 1])
        cellStart[indices[tid]] = tid;
}

__global__ void resetIndicesAndCopyMembership(int* indices, int* membership, int* temp, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N) return;
    indices[tid] = tid;
    temp[tid] = membership[tid];
}

template<int dim> 
__global__ void calculateBestDistance(float* tab, float* kTab, float* kNewTab, int* membership, int* kCount, int* changed, int N, int k)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N)
        return;
    float bestDistance = FLT_MAX;
    int bestCentroid = -1;
    
    // Check every cluster
    for(int i = 0; i < k; i++)
    {
        // Calculate distance to this cluster
        float d = 0;
        for(int j = 0; j < dim; j++)
        {
            float point = tab[tid + j * N];
            float centroid = kTab[i + j * k];
            d += (point - centroid) * (point - centroid);
        }

        // Check if it is closer
        if(d < bestDistance)
        {
            bestDistance = d;
            bestCentroid = i;
        }
    }

    changed[tid] = 0;
    // Save better cluster
    if(membership[tid] != bestCentroid)
    {
        membership[tid] = bestCentroid;
        changed[tid] = 1;
    }
    atomicAdd(&kCount[bestCentroid], 1);
    for(int i = 0; i < dim; i++)
    {
        atomicAdd(&kNewTab[bestCentroid + i * k], tab[tid + i * N]);
    }
}

void runCalculateBestDistance(float* d_tab, float* d_centroid, float* d_newCentroid, int* d_membership, int* d_count, int* d_changed, int N, int k, int dim, int gridN, int block)
{
    switch(dim)
    {
        case 1:
        {
            calculateBestDistance<1><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 2:
        {
            calculateBestDistance<2><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 3:
        {
            calculateBestDistance<3><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 4:
        {
            calculateBestDistance<4><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 5:
        {
            calculateBestDistance<5><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 6:
        {
            calculateBestDistance<6><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 7:
        {
            calculateBestDistance<7><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 8:
        {
            calculateBestDistance<8><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 9:
        {
            calculateBestDistance<9><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 10:
        {
            calculateBestDistance<10><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 11:
        {
            calculateBestDistance<11><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 12:
        {
            calculateBestDistance<12><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 13:
        {
            calculateBestDistance<13><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 14:
        {
            calculateBestDistance<14><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 15:
        {
            calculateBestDistance<15><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 16:
        {
            calculateBestDistance<16><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 17:
        {
            calculateBestDistance<17><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 18:
        {
            calculateBestDistance<18><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 19:
        {
            calculateBestDistance<19><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 20:
        {
            calculateBestDistance<20><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 21:
        {
            calculateBestDistance<21><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 22:
        {
            calculateBestDistance<22><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 23:
        {
            calculateBestDistance<23><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 24:
        {
            calculateBestDistance<24><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 25:
        {
            calculateBestDistance<25><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 26:
        {
            calculateBestDistance<26><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 27:
        {
            calculateBestDistance<27><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 28:
        {
            calculateBestDistance<28><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 29:
        {
            calculateBestDistance<29><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 30:
        {
            calculateBestDistance<30><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 31:
        {
            calculateBestDistance<31><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 32:
        {
            calculateBestDistance<32><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
    }
}

void runDivNewTab(float* d_centroid, float* d_newCentroid, int* d_count, int k, int dim, dim3 gridK, dim3 blockK)
{

    switch(dim)
    {
        case 1:
        {
            divNewTab<1><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 2:
        {
            divNewTab<2><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 3:
        {
            divNewTab<3><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 4:
        {
            divNewTab<4><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 5:
        {
            divNewTab<5><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 6:
        {
            divNewTab<6><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 7:
        {
            divNewTab<7><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 8:
        {
            divNewTab<8><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 9:
        {
            divNewTab<9><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 10:
        {
            divNewTab<10><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 11:
        {
            divNewTab<11><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 12:
        {
            divNewTab<12><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 13:
        {
            divNewTab<13><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 14:
        {
            divNewTab<14><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 15:
        {
            divNewTab<15><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 16:
        {
            divNewTab<16><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 17:
        {
            divNewTab<17><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 18:
        {
            divNewTab<18><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 19:
        {
            divNewTab<19><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 20:
        {
            divNewTab<20><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 21:
        {
            divNewTab<21><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 22:
        {
            divNewTab<22><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 23:
        {
            divNewTab<23><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 24:
        {
            divNewTab<24><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 25:
        {
            divNewTab<25><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 26:
        {
            divNewTab<26><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 27:
        {
            divNewTab<27><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 28:
        {
            divNewTab<28><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 29:
        {
            divNewTab<29><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 30:
        {
            divNewTab<30><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 31:
        {
            divNewTab<31><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
        case 32:
        {
            divNewTab<32><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            break;
        }
    }
}

float* solveGPU(float* h_tab, int N, int dim, int k)
{
    // Allocate memory and prepare data
    float *d_centroid, *d_tab, *d_newCentroid;
    int *d_membership, *d_changed;
    int *d_count;
    float *h_centroid = new float[dim * k];

    cudaMalloc(&d_centroid, dim * k * sizeof(float));
    cudaMalloc(&d_newCentroid, dim * k * sizeof(float));
    cudaMalloc(&d_tab, dim * N * sizeof(float));
    cudaMalloc(&d_membership, N * sizeof(int));
    cudaMalloc(&d_changed, N * sizeof(int));
    cudaMalloc(&d_count, k * sizeof(int));



    cudaMemcpy(d_tab, h_tab, N * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize centroid positions as first k point in tab
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            h_centroid[i + k * j] = h_tab[i + N * j];
        }
    }
    cudaMemcpy(d_centroid, h_centroid, k * dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Solve k-means
    int block = 1024;
    int gridN = N / 1024 + (N % 1024 == 0 ? 0 : 1);
    
    dim3 gridK(k / 32 + (k % 32 == 0 ? 0 : 1), dim / 32 + (dim % 32 == 0 ? 0 : 1), 1);
    dim3 blockK(32, 32, 1);

    int total = 0;
    // Main loop
    while(total <= 10000)
    {
        total++;
        // Calculate distances between all points and all centroids
        cudaMemset(d_newCentroid, 0, k * dim * sizeof(float));
        cudaMemset(d_count, 0, k * sizeof(int));
        
        runCalculateBestDistance(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed,
                                N, k, dim, gridN, block);

        int totalChanged = thrust::reduce(thrust::device, d_changed, d_changed + N, 0);
        if(!totalChanged)
            break;

        runDivNewTab(d_centroid, d_newCentroid, d_count, k, dim, gridK, blockK);
    }
    std::cout << "Total loops for GPU1: " <<  total << std::endl;

    cudaMemcpy(h_centroid, d_centroid, dim * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_centroid);
    cudaFree(d_newCentroid);
    cudaFree(d_tab);
    cudaFree(d_membership);
    cudaFree(d_changed);
    cudaFree(d_count);

    return h_centroid;
}