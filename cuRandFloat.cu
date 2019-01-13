/**********************************************************************
 *
 * cuRandFloat.cu -- Generate an arbitrary amount of random floats
 *
 * Stolen from the CUDA Toolkit documentation
 * https://docs.nvidia.com/cuda/curand
 * 
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include "cuRandFloat.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#define AEM 8263-6698
#define DIM 3

int randFloat(float **out, float **d_out, int size)
{
    size_t n = DIM*size;
    size_t dataSize = n*sizeof(float);
    curandGenerator_t gen;
    float *data, *d_data;
    static int offset = 0;

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc(&d_data, dataSize));
    data = (float *)malloc(dataSize);
    if(data == NULL) {
      printf("Error allocating data");
      exit(1);
    }

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, AEM));

    /* Set offset */
    CURAND_CALL(curandSetPseudoRandomGeneratorOffset(gen, offset));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, d_data, n));

    /*                           WARNING!!                            */
    /* There is no way in my knowledge to have curand include 0 in its
       generation and exclude 1. I just wanted to play with curand in
       this assignment, that's why i kept it. Normal rand from the CPU
       is the better opption here */

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(data, d_data, dataSize), cudaMemcpyDeviceToHost));

    offset += n;
    *d_out = d_data;
    *out = data;

    CURAND_CALL(curandDestroyGenerator(gen));
    return EXIT_SUCCESS;
}