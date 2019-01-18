/**********************************************************************
 *
 * gpuValidation.c -- gpuValidation function for the NNs of Q as 
 * 					  calculated via CUDA
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/
#include "gpuValidation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 3


__global__
void validationKernel(float *Q, int NQ, float *C, int NC, int *results, char verboseFlag)
{
  int proccess = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  float NNdist, dist;
  int NNidx;

  float xQ, yQ, zQ;
  float xC, yC, zC;

  for(int idx=proccess; idx<NQ; idx+=stride) {
    NNdist=1.000000;
    NNidx=-1;
    xQ = Q[idx * DIM + 0];
    yQ = Q[idx * DIM + 1];
    zQ = Q[idx * DIM + 2];

    for(int j = 0; j < NC; j++) {
      dist = 0.000000;
      xC = C[j * DIM + 0];
      yC = C[j * DIM + 1];
      zC = C[j * DIM + 2];

      dist = (xQ-xC)*(xQ-xC) + (yQ-yC)*(yQ-yC) + (zQ-zC)*(zQ-zC);
      dist = sqrtf(dist);

      if(dist<NNdist) {
        NNdist = dist;
        NNidx = j;
      }     
    } // End of going through all C for the one q

    if(results[idx] != NNidx) {
      printf("     ! ! ! VALIDATION FAILED ! ! !\n");
      printf("-> On Q[%d]: (%1.4f, %1.4f, %1.4f)\n",idx, xQ, yQ, zQ);
      if(results[idx] == -1)
        printf("\nAlgorithm did not manage to locate a neighbor within the Primary nor the Secondary Candidates\n\n\n");
      else {
        printf("Algorithm found C[%d] as the NN, while in fact it was C[%d].\n", results[idx], NNidx);
        printf("      (%1.4f, %1.4f, %1.4f)                 (%1.4f, %1.4f, %1.4f)\n\n",
         C[results[idx]*DIM],C[results[idx]*DIM+1],C[results[idx]*DIM+2],
         C[NNidx*DIM],C[NNidx*DIM+1],C[NNidx*DIM+2]);
      }
      if(verboseFlag==0) // If verboseFlag is enabled quit as soon as you find the first miscalculated NN
        return;
    }

  } // End of going through all Q;

}

int gpuValidation(float *Q, int NQ, float *C, int NC, int *results, char verboseFlag, size_t numberOfBlocks, size_t threadsPerBlock)
{

  if(verboseFlag==1)
    printf("\n\n ====== Begining validation of results on the GPU ======\n\n");

  validationKernel<<<numberOfBlocks, threadsPerBlock>>>
    (Q, NQ, C, NC, results, verboseFlag);
      
  return 0;
}