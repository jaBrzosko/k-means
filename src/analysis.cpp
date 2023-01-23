
float getDistanceAnalysis(int pIndex, int kIndex, float* tab, float* kTab, int N, int dim, int k)
{
    float ret = 0;
    for(int i = 0; i < dim; i++)
    {
        float point = tab[pIndex + i * N];
        float centroid = kTab[kIndex + i * k];
        ret += (point - centroid) * (point - centroid);
    }
    return ret;
}

// https://www.saedsayad.com/clustering_kmeans.htm
float squaredError(float* kTab, float* tab, int N, int dim, int k)
{
    float ret = 0;
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < k; j++)
        {
            float v = getDistanceAnalysis(i, j, tab, kTab, N, dim, k);
            ret += v;
        }
    }


    return ret;
}