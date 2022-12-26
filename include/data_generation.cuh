// Generate float table with values varying from minValue to maxValue.
// Seed is passed to RNG
// Result table will be of N * dim lenght with next values of object stored every N element
float* generateData(int N, int dim, int minValue, int maxValue, int seed = 0);