#include <cfloat>
#include <iostream>

float getDistance(int pIndex, int kIndex, float* tab, float* kTab, int dim, int N, int k)
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

float* solveCPU(float* tab, int N, int dim, int k)
{
    float* centroids = new float[k * dim];
    float* tempCentroids = new float[k * dim];
    int* membership = new int[N];
    int* membershipCount = new int[k];
    // Choose first k points as initial centroids
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            centroids[i + k * j] = tab[i + N * j];
        }
    }

    int total = 0;
    int count = 1;
    while(count)
    {
        total++;
        count = 0;
        // For every point in data set
        for(int i = 0; i < N; i++)
        {
            float distance = FLT_MAX;
            int bestCentroid = -1;
            // For every centroid
            for(int j = 0; j < k; j++)
            {
                // Calculate distance between point and centroid
                float d = getDistance(i, j, tab, centroids, dim, N, k);
                // Check which centroid is closer
                if(d < distance)
                {
                    distance = d;
                    bestCentroid = j;
                }
            }
            // Different centroid was found
            if(membership[i] != bestCentroid)
            {
                count++;
                membership[i] = bestCentroid;
            }
        }
        
        // Reset centroid position accumulators
        for(int i = 0; i < k * dim; i++)
        {
            tempCentroids[i] = 0;
        }
        // Reset centroid counter accumulators
        for(int i = 0; i < k; i++)
        {
            membershipCount[i] = 0;
        }

        for(int i = 0; i < N; i++)
        {
            int myK = membership[i];
            membershipCount[myK]++;
            for(int j = 0; j < dim; j++)
            {
                tempCentroids[myK + k * j] += tab[i + N * j];
            }
        }

        //std::cout << "Loop number " << total << std::endl;
        // Apply new centroid positions
        for(int i = 0; i < k; i++)
        {
            int cnt = membershipCount[i];
            //std::cout << "Centroid " << i << " has " << cnt << " members" << std::endl;
            for(int j = 0; j < dim; j++)
            {
                centroids[i] = tempCentroids[i] / cnt;
            }
        }
    }
    std::cout << "Total loops for CPU: " <<  total << std::endl;
    return centroids;
}