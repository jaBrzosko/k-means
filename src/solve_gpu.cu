#include <cfloat>
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>



__global__ void divNewTab(float* centroid, float* newCentroid, int* count, int k, int dim)
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

__global__ void calculateBestDistance(float* tab, float* kTab, float* kNewTab, int* membership, int* kCount, int* changed, int N, int k, int dim)
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
        calculateBestDistance<<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k, dim);
        // {
        //     int* debug = new int[k];
        //     cudaMemcpy(debug, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);
        //     for(int i = 0; i < k; i++)
        //     {
        //         std::cout << i << ") " << debug[i] << std::endl;
        //     }
        // }
        int totalChanged = thrust::reduce(thrust::device, d_changed, d_changed + N, 0);
        //std::cout << "Total changed " << totalChanged << std::endl;
        if(!totalChanged)
            break;
        divNewTab<<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k, dim);
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