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
<<<<<<< HEAD
#include "cpuValidation.h"
=======
#include "cuNearestNeighbor.h"
>>>>>>> master

#define DIM 3

int main (int argc, char *argv[]) {

  float *Q, *C, *d_Q, *d_C;
  size_t QSize, CSize;
  int *S, *d_S, *P, *d_P;
  int NC, NQ, d;
  int SDim;
  cudaError_t err;

  if (argc != 4) {
    printf("Usage: %s arg1 arg2 arg3\n  where NC=2^arg1, NQ=2^arg2 and d=2^arg3\n",
	   argv[0]);
    exit(1);
  }

  NC = 1<<atoi(argv[1]);
  NQ = 1<<atoi(argv[2]);
  d  = 1<<atoi(argv[3]);
  
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

  // Hashing C into d*d*d boxes
  hashing3D(C, d_C, CSize, NC, d, &S, &d_S, numberOfBlocks, threadsPerBlock);

  int *QBoxIdToCheck, *d_QBoxIdToCheck;
  hashing3D(Q, d_Q, QSize, NQ, d, &P, &d_P, &QBoxIdToCheck, &d_QBoxIdToCheck,
	    numberOfBlocks, threadsPerBlock);

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
    SDim = (S[boxid+1]-S[boxid])/3;
    printf("Box%d size=%d\n", boxid, SDim);
      for(int i = 0; i < SDim ; i++){
        for (int d=0; d<DIM; d++)
          printf("%1.4f ", C[ S[boxid] +i*DIM +d ]);
        printf("\n");
      }
  }

  int *neighbor, *d_neighbor;
  size_t neighborSize = NQ * sizeof(int);
  
  CUDA_CALL(cudaMalloc(&d_neighbor,neighborSize));
  neighbor = (int *)malloc(neighborSize);
  if(neighbor == NULL) {
    printf("Error allocating neighbor");
    exit(1);
  }

  cuNearestNeighbor<<<numberOfBlocks, threadsPerBlock>>>
    (d_C,d_S,d_Q,NQ,d_QBoxIdToCheck,d,d_neighbor);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(neighbor, d_neighbor, neighborSize, cudaMemcpyDeviceToHost));

  printf(" ==== Neighbors! ==== \n");
  for(int i = 0; i < NQ ; i++){
    for (int d=0; d<DIM; d++)
      printf("%1.4f ", Q[i*DIM+d]);
    printf("-> ");
    for (int d=0; d<DIM; d++)
      printf("%1.4f ", C[neighbor[i]+d]);
    printf("\n");
  }

  /* Validating the NN results */
  cpuValidation(Q, NQ, C, NC, neighbor);
  
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