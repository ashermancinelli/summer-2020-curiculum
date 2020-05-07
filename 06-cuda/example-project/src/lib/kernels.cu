
#include <cuda.h>

__global__
void saxpy(int n, double d, double* A, double* B)
{
    int tid = threadIdx.x + blockIdx.x * blockIdx.y;
    if (tid < n)
        A[i] = B[i] * d;
}
