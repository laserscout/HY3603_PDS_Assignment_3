/**********************************************************************
 *
 * cuRandFloat.h -- Generate an arbitrary amount of random floats
 *
 * Stolen from the CUDA Toolkit documentation
 * https://docs.nvidia.com/cuda/curand
 * 
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef CU_RAND_FLOAT_H
#define CU_RAND_FLOAT_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error \"%s\" at %s:%d\n", \
    cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error \"%s\" at %s:%d\n", \
    cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int randFloat(float **out, int size);

#endif /* CU_RAND_FLOAT_H */

