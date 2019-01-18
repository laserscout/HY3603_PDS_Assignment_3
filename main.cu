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
#include "gpuValidation.h"
#include "cuNearestNeighbor.h"
#include "cuNearestNeighbor2ndPass.h"

#define DIM 3

int main (int argc, char *argv[]) {

  float *Q, *C, *d_Q, *d_C;
  int *S, *d_S, *P, *d_P, *QBoxIdToCheck, *d_QBoxIdToCheck;
  int NC, NQ, d, SDim;
  cudaError_t err;
  char verboseFlag = 0;
  char noValidationFlag = 0;

  // Parsing input arguments
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
    
  // Initializing
    
  int d3 = d*d*d;
  size_t QSize = DIM * NQ * sizeof(float);
  size_t CSize = DIM * NC * sizeof(float);
  size_t QBoxIdToCheckSize = NQ * sizeof(int);
  size_t SSize = (d3+1) * sizeof(int); 
  
  // CUDA Device setup
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
    
  // Timers setup
  cudaEvent_t startOfHashing, startOfFirstRun, startOfSecondRun, stop;
  cudaEventCreate(&startOfHashing);
  cudaEventCreate(&startOfFirstRun);
  cudaEventCreate(&startOfSecondRun);
  cudaEventCreate(&stop);

  // Create input Data
  randFloat(&Q, &d_Q, NQ);
  randFloat(&C, &d_C, NC);
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

  // Aranging C and Q into d*d*d boxes
  cudaEventRecord(startOfHashing);
  hashing3D(C, &d_C, CSize, NC, d, &S, &d_S, 
  		numberOfBlocks, threadsPerBlock);

  hashing3D(Q, &d_Q, QSize, NQ, d, &P, &d_P, &QBoxIdToCheck, &d_QBoxIdToCheck,
	    numberOfBlocks, threadsPerBlock);

  if(verboseFlag == 1){
    /* Show result */
    CUDA_CALL(cudaMemcpy(Q, d_Q, QSize, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(QBoxIdToCheck, d_QBoxIdToCheck, QBoxIdToCheckSize, cudaMemcpyDeviceToHost));
    printf("\nd=%d\n\n",d);
    printf(" ====new Q vector==== \n");
    for(int i = 0; i < NQ ; i++){
      for (int d=0; d<DIM; d++)
	printf("%1.4f ", Q[i*DIM+d]);
      printf("| Belongs to box:%d\n",QBoxIdToCheck[i]);
    }
    CUDA_CALL(cudaMemcpy(C, d_C, CSize, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(S, d_S, SSize, cudaMemcpyDeviceToHost));
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

  // First Run of Nearest neighbor function
  cudaEventRecord(startOfFirstRun);

  int *neighbor, *d_neighbor;
  int *d_checkOutside;
  size_t neighborSize = NQ * sizeof(int);
  size_t checkOutsideSize = (NQ+1) * sizeof(int);
  
  CUDA_CALL(cudaMalloc(&d_neighbor,neighborSize));
  neighbor = (int *)malloc(neighborSize);
  if(neighbor == NULL) {
    printf("Error allocating neighbor");
    exit(1);
  }
  CUDA_CALL(cudaMalloc(&d_checkOutside,checkOutsideSize));
  
  cuNearestNeighbor<<<numberOfBlocks, threadsPerBlock>>>
    (d_C,d_S,d_Q,NQ,d_QBoxIdToCheck,d,d_neighbor,d_checkOutside);
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  

  CUDA_CALL(cudaDeviceSynchronize());
/*
  CUDA_CALL(cudaMalloc(&d_checkOutside,checkOutsideSize));
  
  cuWhichCheckOutside<<<numberOfBlocks, threadsPerBlock>>>
    (d_Q, NQ, d_checkOutside, d_QtoCheckOutside);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  

  CUDA_CALL(cudaDeviceSynchronize());
*/
  if(verboseFlag == 1) {
    CUDA_CALL(cudaMemcpy(neighbor, d_neighbor, neighborSize, cudaMemcpyDeviceToHost));
    printf(" ==== Neighbors! ==== \n");
    for(int i = 0; i < NQ ; i++)
    	printf("> Q[%d] -> C[%d]\n",i,neighbor[i]);
  }

  // Second Run of Nearest neighbor function
  cudaEventRecord(startOfSecondRun);

  cuNearestNeighbor2ndPass<<<numberOfBlocks, threadsPerBlock/8>>>
    (d_C,d_S,d_Q,NQ,d_QBoxIdToCheck,d,d_neighbor,d_checkOutside);
    
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  
  CUDA_CALL(cudaDeviceSynchronize());

  if(verboseFlag == 1) {
    CUDA_CALL(cudaMemcpy(neighbor, d_neighbor, neighborSize, cudaMemcpyDeviceToHost));
    printf(" ==== Neighbors! ==== \n");
    for(int i = 0; i < NQ ; i++)
    	printf(">> Q[%d] -> C[%d]\n",i,neighbor[i]);
  }
    
  // THE END
  cudaEventRecord(stop);
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

  if(noValidationFlag==0) {
    /* Validating the NN results */
    // CUDA_CALL(cudaMemcpy(neighbor, d_neighbor, neighborSize, cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(C, d_C, CSize, cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(Q, d_Q, QSize, cudaMemcpyDeviceToHost));
    // cpuValidation(Q, NQ, C, NC, neighbor, verboseFlag);
    gpuValidation(d_Q, NQ, d_C, NC, d_neighbor, verboseFlag, numberOfBlocks, threadsPerBlock);
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