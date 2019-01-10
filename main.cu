/**********************************************************************
 *
 * main.cu -- main function for the NN in cuda
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
//#include "cuNearestNeighborHelper.h"
#include "cuRandFloat.h"

#define DIM 3

int main (int argc, char *argv[]) {

  float *Q, *C;
  int NC,NQ;

  if (argc != 3) {
    printf("Usage: %s arg1 arg2\n  where NC=2^arg1 and NQ=2^arg2\n",
	   argv[0]);
    exit(1);
  }

  NC = 1<<atoi(argv[1]);
  NQ = 1<<atoi(argv[2]);
  
  size_t threadsPerBlock;
  size_t numberOfBlocks;

  int deviceId;
  cudaDeviceProp props;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);
  
  threadsPerBlock = 8*props.warpSize;
  numberOfBlocks  = 50*props.multiProcessorCount;

  randFloat(&Q, NQ);
  randFloat(&C, NC);

  CUDA_CALL(cudaMemPrefetchAsync(Q, NQ, cudaCpuDeviceId));
  CUDA_CALL(cudaMemPrefetchAsync(C, NC, cudaCpuDeviceId));

  /* Show result */
  printf("===Q vector===\n");
  for(int i = 0; i < NQ ; i++){
    for (int d=0; d<DIM; d++)
      printf("%1.4f ", Q[i*DIM+d]);
    printf("\n");
  }
  printf("===C vector===\n");
  for(int i = 0; i < NC ; i++){
    for (int d=0; d<DIM; d++)
      printf("%1.4f ", C[i*DIM+d]);
    printf("\n");
  }

  /* Cleanup */
  CUDA_CALL(cudaFree(Q));
  CUDA_CALL(cudaFree(C));

  return 0;
}