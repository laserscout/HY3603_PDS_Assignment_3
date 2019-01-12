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

int randFloat(float **out, int size)
{
    size_t n = 1<<size;
    size_t d = 3;          // Number of dimentions for each float
    size_t outSize = n*d*sizeof(float);
    curandGenerator_t gen;
    float *devData;

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc(&devData, outSize));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                AEM));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
    
    *out = devData;

    // /* Copy device memory to host */
    // CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
    //     cudaMemcpyDeviceToHost));

    // /* Show result */
    // for(i = 0; i < n; i++) {
    //     printf("%1.4f ", hostData[i]);
    // }
    // printf("\n");

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    return EXIT_SUCCESS;
}