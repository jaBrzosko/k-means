#define KEY 104729
#define mapRange(a1,a2,b1,b2,s) (b1 + (s-a1)*(b2-b1)/(a2-a1))

float* generateData(int N, int dim, int minValue, int maxValue, int seed = 0)
{
    srand(seed);
    float* tab = new float[N * dim];
    for(int i = 0; i < N * dim; i++)
    {
        float v = rand() % KEY;
        tab[i] = mapRange(0, KEY - 1, minValue, maxValue, v);
    }
    return tab;
}