float* solveGPU(float* tab, int N, int dim, int k)
{
    // Allocate memory and prepare data
    float *d_centroid, *d_tab;
    float *h_centroid = new float[dim * k];

    cudaMalloc(&d_centroid, dim * k * sizeof(float));
    cudaMalloc(&d_tab, dim * N * sizeof(float));
    cudaMemcpy(d_tab, h_tab, N * dim * sizeof(float));
    
    cudaMemcpy(h_centroid, d_centroid, dim * k * sizeof(float), cudaMemcpyDeviceToHost);

    return h_centroid;
}