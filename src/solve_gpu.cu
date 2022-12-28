#include <cfloat>
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

__global__ void findNewMean(float* position, float* centroids, int* indices, int* start, int N, int k, int dim)
{
    unsigned int kIndex   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int dimIndex = threadIdx.y + blockIdx.y * blockDim.y;

    // Don't go over boundaries
    if(kIndex >= k || dimIndex >= dim) return;

    float localSum = 0;
    int myStart = start[kIndex];
    int myEnd = start[kIndex + 1];

    position += dimIndex * N;

    for(int i = myStart; i < myEnd; i++)
    {
        localSum += position[indices[i]];
    }
    centroids[dimIndex + kIndex * dim] = localSum / (myEnd - myStart);
}

// if index changes value over neighbors update cellStart
__global__ void prepareCellStart(int* indices, int* cellStart)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0)
        cellStart[indices[0]] = 0;
    else if(indices[tid] != indices[tid - 1])
        cellStart[indices[tid]] = tid;
}

__global__ void resetIndicesAndCopyMembership(int* indices, int* membership, int* temp, int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N)
        return;
    indices[tid] = tid;
    temp[tid] = membership[tid];
}

__global__ void calculateBestDistance(float* tab, float* kTab, int* membership, int* changed, int N, int k, int dim)
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

    // Save better cluster
    if(bestCentroid != membership[tid])
    {
        membership[tid] = bestCentroid;
        changed[tid] = 1;
    }
    else
    {
        changed[tid] = 0;
    }
}

float* solveGPU(float* h_tab, int N, int dim, int k)
{
    // Allocate memory and prepare data
    float *d_centroid, *d_tab;
    int *d_membership, *d_changed, *d_index, *d_tempMembership;
    int *d_start;
    float *h_centroid = new float[dim * k];

    cudaMalloc(&d_centroid, dim * k * sizeof(float));
    cudaMalloc(&d_tab, dim * N * sizeof(float));
    cudaMalloc(&d_membership, N * sizeof(int));
    cudaMalloc(&d_tempMembership, N * sizeof(int));
    cudaMalloc(&d_changed, N * sizeof(int));
    cudaMalloc(&d_index, N * sizeof(int));
    cudaMalloc(&d_start, k * sizeof(int));
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
    
    dim3 gridK(k / 32 + (k % 1024 == 0 ? 0 : 1), dim / 32 + (dim % 1024 == 0 ? 0 : 1), 1);
    dim3 blockK(32, 32, 1);
    
    int total = 0;
    // Main loop
    while(1)
    {
        total++;
        // Calculate distances between all points and all centroids
        calculateBestDistance<<<gridN, block>>>(d_tab, d_centroid, d_membership, d_changed, N, k, dim);
        int totalChanged = thrust::reduce(thrust::device, d_changed, d_changed + N, 0);
        //std::cout << "Total changed " << totalChanged << std::endl;
        if(!totalChanged)
            break;

        // Find new centroid positions
        resetIndicesAndCopyMembership<<<gridN, block>>>(d_index, d_membership, d_tempMembership, N);
        thrust::sort_by_key(thrust::device, d_tempMembership, d_tempMembership + N, d_index);
        prepareCellStart<<<gridN, block>>>(d_index, d_start);
        findNewMean<<<gridK, blockK>>>(d_tab, d_centroid, d_index, d_start, N, k, dim);
    }
    std::cout << "Total loops for GPU: " <<  total << std::endl;

    cudaMemcpy(h_centroid, d_centroid, dim * k * sizeof(float), cudaMemcpyDeviceToHost);

    return h_centroid;
}