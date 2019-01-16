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
#include <string.h>
#include "cuRandFloat.h"
#include "hashing3D.h"
#include "cpuValidation.h"
#include "cuNearestNeighbor.h"
#include "cuNearestNeighbor2ndPass.h"

#define DIM 3

int main (int argc, char *argv[]) {

  float *Q, *C, *d_Q, *d_C;
  size_t QSize, CSize;
  int *S, *d_S, *P, *d_P;
  int NC, NQ, d;
  int SDim;
  cudaError_t err;
  char verboseFlag = 0;
  char noValidationFlag = 0;

  if (argc < 4) {
    printf("Usage: %s [flags] arg1 arg2 arg3\n  where NC=2^arg1, NQ=2^arg2 and d=2^arg3\n",
	   argv[0]);
    exit(1);
  }

  for(int i=1; i<argc; i++) {
    if (strcmp(argv[i], "-v") == 0)
      {                 
        verboseFlag = 1; // use only with small NC NQ and d
      }
    if (strcmp(argv[i], "--novalidation") == 0)
      {                 
        noValidationFlag = 1; // Do not run the slow validation in the end
      }
    if (strncmp(argv[i], "-", 1) != 0) {
      NC = 1<<atoi(argv[i]);
      NQ = 1<<atoi(argv[i+1]);
      d  = 1<<atoi(argv[i+2]);
      break;
    }
      
  }
  
  size_t threadsPerBlock, warp;
  size_t numberOfBlocks, multiP;

  int deviceId;
  cudaDeviceProp props;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);

  warp = props.warpSize;
  multiP = props.multiProcessorCount;
  threadsPerBlock = 8*warp;
  numberOfBlocks  = 5*multiP;

  randFloat(&Q, &d_Q, NQ);
  QSize = DIM * NQ * sizeof(float);
  randFloat(&C, &d_C, NC);
  CSize = DIM * NC * sizeof(float);
  CUDA_CALL(cudaDeviceSynchronize());

  if(verboseFlag == 1) {
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
  }

  cudaEvent_t startOfHashing, startOfFirstRun, startOfSecondRun, stop;
  cudaEventCreate(&startOfHashing);
  cudaEventCreate(&startOfFirstRun);
  cudaEventCreate(&startOfSecondRun);
  cudaEventCreate(&stop);

  cudaEventRecord(startOfHashing);

  // Hashing C into d*d*d boxes
  hashing3D(C, d_C, CSize, NC, d, &S, &d_S, 
  		numberOfBlocks, threadsPerBlock);

  int *QBoxIdToCheck, *d_QBoxIdToCheck;
  hashing3D(Q, d_Q, QSize, NQ, d, &P, &d_P, &QBoxIdToCheck, &d_QBoxIdToCheck,
	    numberOfBlocks, threadsPerBlock);

  if(verboseFlag == 1){
    /* Show result */
    printf("\nd=%d\n\n",d);
    printf(" ====new Q vector==== \n");
    for(int i = 0; i < NQ ; i++){
      for (int d=0; d<DIM; d++)
	printf("%1.4f ", Q[i*DIM+d]);
      printf("| Belongs to box:%d\n",QBoxIdToCheck[i]);
    }
    printf(" ======S vector====== \n");
    for(int boxid=0;boxid<d*d*d;boxid++){
      SDim = S[boxid+1]-S[boxid];
      printf("Box%d size=%d\n", boxid, SDim);
      for(int i = S[boxid] ; i < S[boxid+1] ; i++){
        for (int d=0; d<DIM; d++)
          printf("%1.4f ", C[ i*DIM +d ]);
        printf("\n");
      }
    }
  }

  cudaEventRecord(startOfFirstRun);

  int *neighbor, *d_neighbor;
  char *d_checkOutside;
  size_t neighborSize = NQ * sizeof(int);
  size_t checkOutsideSize = NQ * sizeof(char);
  
  CUDA_CALL(cudaMalloc(&d_neighbor,neighborSize));
  neighbor = (int *)malloc(neighborSize);
  if(neighbor == NULL) {
    printf("Error allocating neighbor");
    exit(1);
  }

  CUDA_CALL(cudaMalloc(&d_checkOutside,checkOutsideSize));
  
  cudaEventRecord(startOfSecondRun);

  cuNearestNeighbor<<<numberOfBlocks, threadsPerBlock>>>
    (d_C,d_S,d_Q,NQ,d_QBoxIdToCheck,d,d_neighbor,d_checkOutside);

  cudaEventRecord(stop);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  
  CUDA_CALL(cudaDeviceSynchronize());
    
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, startOfHashing, startOfFirstRun);
  printf("Duration of Q and C hashing: %1.6fms\n",milliseconds);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, startOfFirstRun, startOfSecondRun);
  printf("Duration of the first run of the kernel: %1.6fms\n",milliseconds);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, startOfSecondRun, stop);
  printf("Duration of the second run of the kernel: %1.6fms\n",milliseconds);


  CUDA_CALL(cudaMemcpy(neighbor, d_neighbor, neighborSize, cudaMemcpyDeviceToHost));
  
  if(verboseFlag == 1) {
    printf(" ==== Neighbors! ==== \n");
    for(int i = 0; i < NQ ; i++)
    	printf("> Q[%d] -> C[%d]\n",i,neighbor[i]);
  }

  cuNearestNeighbor2ndPass<<<numberOfBlocks*10, 27>>>
    (d_C,d_S,d_Q,NQ,d_QBoxIdToCheck,d,d_neighbor,d_checkOutside);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }

  
  if(verboseFlag == 1) {
    printf(" ==== Neighbors! ==== \n");
    for(int i = 0; i < NQ ; i++)
    	printf(">> Q[%d] -> C[%d]\n",i,neighbor[i]);
  }

  if(noValidationFlag==0) {
    CUDA_CALL(cudaMemcpy(neighbor, d_neighbor, neighborSize, cudaMemcpyDeviceToHost));
    /* Validating the NN results */
    cpuValidation(Q, NQ, C, NC, neighbor, verboseFlag);
  }
  
  /* Cleanup */
  CUDA_CALL(cudaFree(d_Q));
  CUDA_CALL(cudaFree(d_C));
  CUDA_CALL(cudaFree(d_S));
  CUDA_CALL(cudaFree(d_QBoxIdToCheck));
  free(Q);
  free(C);
  free(S);
  free(QBoxIdToCheck);
  
  return 0;
}