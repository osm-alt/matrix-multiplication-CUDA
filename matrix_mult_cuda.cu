%%cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int j, int k, int l) {
  
// Calculate the row index of the P element and M
int Row = blockIdx.y*blockDim.y+threadIdx.y;

// Calculate the column index of P and N
int Col = blockIdx.x*blockDim.x+threadIdx.x;

if ((Row < j) && (Col < l)) {
float Pvalue = 0;
// each thread computes one element of the block sub-matrix
for (int _k = 0; _k < k; ++_k) {
Pvalue += d_M[Row*k+_k]*d_N[_k*l+Col];
}

d_P[Row*l+Col] = Pvalue;
}
}


int main(int argc, char *argv[])
{
    clock_t t1 = clock();
    int j = 1000;
    int k = 500;
    int l = 800;
     float *h_M = (float*)malloc(j*k*sizeof(float));
     float *h_N = (float*)malloc(k*l*sizeof(float));
     float *h_P = (float*)malloc(j*l*sizeof(float));
 
    int i;
 
    for(i = 0; i < j * k; i++)
    {
        h_M[i] = 2;
    } 
  
    for(i = 0; i < k * l; i++)
    {
        h_N[i] = 3;
    }  
 
 
    int Msize = j * k * sizeof(float);
    int Nsize = k * l * sizeof(float);
    int Psize = j * l * sizeof(float);

    float *d_M, *d_N, *d_P;
    cudaMalloc((void **) &d_M, Msize); 
    cudaMalloc((void **) &d_N, Nsize);
    cudaMalloc((void **) &d_P, Psize);

    cudaMemcpy(d_M, h_M, Msize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, Nsize, cudaMemcpyHostToDevice);

    dim3 DimGrid((l-1)/32 + 1, (j-1)/32 + 1, 1);
    dim3 DimBlock(32, 32, 1);
 
    MatrixMulKernel<<<DimGrid, DimBlock>>>(d_M, d_N, d_P, j, k, l);
 cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__);
        exit(-1);
    }

   cudaMemcpy(h_P, d_P, Psize, cudaMemcpyDeviceToHost);
 
   cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
 
  free(h_M);
 free(h_N);
 free(h_P);

clock_t t2 = clock();
    printf("Elapsed time = % 5.3f seconds\n", (float)(t2 - t1) / CLOCKS_PER_SEC);
    return 0;
}