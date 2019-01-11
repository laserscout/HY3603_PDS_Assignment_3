/**********************************************************************
 *
 * hashing3D.h --  Hashing the 3D space into d*d*d boxes
 * 
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef HASHING3D_H
#define HASHING3D_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error \"%s\" at %s:%d\n", \
    cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void hashingKernel(float *v, int N, int d, float ***boxes, int *boxesSizes);

__global__ void InitBoxesSizes(int *boxesSizes, int N);

void hashing3D(float *v, int N, int d, float ***boxes, int *boxesSizes);

#endif /* HASHING3D_H */