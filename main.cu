/**********************************************************************
 *
 * main.cu -- main function for the NN in CUDA
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
#include "hashing3D.h"

#define DIM 3

int main (int argc, char *argv[]) {

  float *Q, *C;
//   float ***S, ***Q_to_S;
  int NC,NQ, d;
//   int *S_Sizes, *Q_to_S_Sizes;

  if (argc != 4) {
    printf("Usage: %s arg1 arg2 arg3\n  where NC=2^arg1, NQ=2^arg2 and d=2^arg3\n",
	   argv[0]);
    exit(1);
  }

  NC = 1<<atoi(argv[1]);
  NQ = 1<<atoi(argv[2]);
  d  = 1<<atoi(argv[3]);
  
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
  printf(" ======Q vector====== \n");
  for(int i = 0; i < NQ ; i++){
    for (int d=0; d<DIM; d++)
      printf("%1.4f ", Q[i*DIM+d]);
    printf("\n");
  }
  printf(" ======C vector====== \n");
  for(int i = 0; i < NC ; i++){
    for (int d=0; d<DIM; d++)
      printf("%1.4f ", C[i*DIM+d]);
    printf("\n");
  }

  float ***S, ***Q_to_S;
  int *S_Sizes, *Q_to_S_Sizes;
  int d3 = d*d*d;
  
  // Hashing C into d*d*d boxes
  hashing3D(C, NC, d, &S, &S_Sizes, numberOfBlocks, threadsPerBlock);

  // Hashing Q into d*d*d boxes
  hashing3D(Q, NQ, d, &Q_to_S, &Q_to_S_Sizes, numberOfBlocks, threadsPerBlock);
    
  CUDA_CALL(cudaMemPrefetchAsync(S, d3 * sizeof(float **) + NC * sizeof(float*), cudaCpuDeviceId));
  CUDA_CALL(cudaMemPrefetchAsync(Q_to_S, d3 * sizeof(float **) + NQ * sizeof(float*), cudaCpuDeviceId));
  CUDA_CALL(cudaMemPrefetchAsync(S_Sizes, d3, cudaCpuDeviceId));  
  CUDA_CALL(cudaMemPrefetchAsync(Q_to_S_Sizes, d3, cudaCpuDeviceId));

  /* Show result */
  printf("\nd=%d\n\n",d);
    
  printf(" ======S vector======\n");
  for(int boxid=0;boxid<d*d*d;boxid++){
      for(int i = 0; i < S_Sizes[boxid] ; i++) {
        for (int d=0; d<DIM; d++)
          printf("%1.4f ", S[boxid][i][d]);
      	printf("\n");
  	  }
      printf(" S_Sizes[%d]=%d\n",boxid,S_Sizes[boxid]);
  }
  printf(" ===Q_to_S  vector===\n");
  for(int boxid=0;boxid<d*d*d;boxid++){
      for(int i = 0; i < Q_to_S_Sizes[boxid] ; i++) {
        for (int d=0; d<DIM; d++)
          printf("%1.4f ", Q_to_S[boxid][i][d]);
      	printf("\n");
      }
      printf(" Q_to_S_Sizes[%d]=%d\n",boxid,Q_to_S_Sizes[boxid]);
  }

  /* Cleanup */
  CUDA_CALL(cudaFree(Q));
  CUDA_CALL(cudaFree(C));
  CUDA_CALL(cudaFree(Q_to_S));
  CUDA_CALL(cudaFree(S));
  CUDA_CALL(cudaFree(Q_to_S_Sizes));
  CUDA_CALL(cudaFree(S_Sizes));

  return 0;
}