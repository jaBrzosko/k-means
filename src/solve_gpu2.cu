#include <cfloat>
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template<int dim>
__global__ void divNewTab2(float* centroid, float* newCentroid, int* count, int k)
{
    unsigned int kIndex   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int dimIndex = threadIdx.y + blockIdx.y * blockDim.y;

    // Don't go over boundaries
    if(kIndex >= k || dimIndex >= dim) return;

    int cnt = count[kIndex];
    if(cnt != 0)
        centroid[kIndex + dimIndex * k] = newCentroid[kIndex + dimIndex * k] / cnt;
}

__global__ void initProperTables(float* tab, float* kTab, float* kNewTab, int* membership, int* kCount, int N, int k, int dim)
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
    
    membership[tid] = bestCentroid;
    atomicAdd(&kCount[bestCentroid], 1);

    for(int i = 0; i < dim; i++)
    {
        atomicAdd(&kNewTab[bestCentroid + i * k], tab[tid + i * N]);
    }
}

template<int dim> 
__global__ void calculateBestDistance2(float* tab, float* kTab, float* kNewTab, int* membership, int* kCount, int* changed, int N, int k)
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
    int previousCentroid = membership[tid];
    if(previousCentroid != bestCentroid)
    {
        membership[tid] = bestCentroid;
        changed[tid] = 1;
        
        // Add one to new centroid
        atomicAdd(&kCount[bestCentroid], 1);
        atomicAdd(&kCount[previousCentroid], -1);
        for(int i = 0; i < dim; i++)
        {
            float v = tab[tid + i * N];
            atomicAdd(&kNewTab[bestCentroid + i * k], v);
            atomicAdd(&kNewTab[previousCentroid + i * k], -v);
        }
        return;
    }
    changed[tid] = 0;
}

void runCudaFunctions(float* d_tab, float* d_centroid, float* d_newCentroid, int* d_membership, int* d_count, int* d_changed, 
                int N, int k, int dim, int gridN, dim3 gridK, int block, dim3 blockK)
{
    switch(dim)
    {
        case 1:
        {
            divNewTab2<1><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<1><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 2:
        {
            divNewTab2<2><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<2><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 3:
        {
            divNewTab2<3><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<3><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 4:
        {
            divNewTab2<4><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<4><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 5:
        {
            divNewTab2<5><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<5><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 6:
        {
            divNewTab2<6><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<6><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 7:
        {
            divNewTab2<7><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<7><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 8:
        {
            divNewTab2<8><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<8><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 9:
        {
            divNewTab2<9><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<9><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 10:
        {
            divNewTab2<10><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<10><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 11:
        {
            divNewTab2<11><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<11><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 12:
        {
            divNewTab2<12><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<12><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 13:
        {
            divNewTab2<13><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<13><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 14:
        {
            divNewTab2<14><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<14><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 15:
        {
            divNewTab2<15><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<15><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 16:
        {
            divNewTab2<16><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<16><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 17:
        {
            divNewTab2<17><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<17><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 18:
        {
            divNewTab2<18><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<18><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 19:
        {
            divNewTab2<19><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<19><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 20:
        {
            divNewTab2<20><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<20><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 21:
        {
            divNewTab2<21><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<21><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 22:
        {
            divNewTab2<22><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<22><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 23:
        {
            divNewTab2<23><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<23><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 24:
        {
            divNewTab2<24><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<24><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 25:
        {
            divNewTab2<25><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<25><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 26:
        {
            divNewTab2<26><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<26><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 27:
        {
            divNewTab2<27><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<27><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 28:
        {
            divNewTab2<28><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<28><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 29:
        {
            divNewTab2<29><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<29><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 30:
        {
            divNewTab2<30><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<30><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 31:
        {
            divNewTab2<31><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<31><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
        case 32:
        {
            divNewTab2<32><<<gridK, blockK>>>(d_centroid, d_newCentroid, d_count, k);
            calculateBestDistance2<32><<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k);
            break;
        }
    }
}
float* solveGPU2(float* h_tab, int N, int dim, int k)
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

    // Prepare init data
    int total = 1;
    cudaMemset(d_count, 0, k * sizeof(int));
    initProperTables<<<gridN, block>>>(d_tab, d_centroid, d_newCentroid, d_membership, d_count, N, k, dim);

    // Main loop
    while(total <= 10000)
    {
        total++;
        // Calculate distances between all points and all centroids
        runCudaFunctions(d_tab, d_centroid, d_newCentroid, d_membership, d_count, d_changed, N, k, dim, gridN, gridK, block, blockK);

        int totalChanged = thrust::reduce(thrust::device, d_changed, d_changed + N, 0);
        if(!totalChanged)
            break;
    }
    std::cout << "Total loops for GPU2: " <<  total << std::endl;

    cudaMemcpy(h_centroid, d_centroid, dim * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_centroid);
    cudaFree(d_newCentroid);
    cudaFree(d_tab);
    cudaFree(d_membership);
    cudaFree(d_changed);
    cudaFree(d_count);

    return h_centroid;
}