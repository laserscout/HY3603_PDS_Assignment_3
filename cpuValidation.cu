/**********************************************************************
 *
 * cpuValidation.c -- cpuValidation function for the NNs of Q as 
 * 					  calculated via CUDA
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/
#include "cpuValidation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 3

int cpuValidation(float *Q, int NQ, float *C, int NC, int *results, char verboseFlag)
{
  float NNdist, dist;
  int NNidx;
  int flag = 0;

  float xQ, yQ, zQ;
  float xC, yC, zC;

  if(verboseFlag==1)
    printf("\n\n ====== Begining validation of results ======\n\n");

  for(int i = 0; i < NQ; i++) {
    NNdist=1.000000;
    xQ = Q[i * DIM + 0];
    yQ = Q[i * DIM + 1];
    zQ = Q[i * DIM + 2];

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

    if(results[i] != NNidx) {
      printf("     ! ! ! VALIDATION FAILED ! ! !\n");
      printf("-> On Q[%d]: (%1.4f, %1.4f, %1.4f)\n",i, xQ, yQ, zQ);
      printf("Algorithm found C[%d] as the NN, while in fact it was C[%d].\n", results[i], NNidx);
      printf("      (%1.4f, %1.4f, %1.4f)                 (%1.4f, %1.4f, %1.4f)\n\n",
	     C[results[i]*DIM],C[results[i]*DIM+1],C[results[i]*DIM+2],
	     C[NNidx*DIM],C[NNidx*DIM+1],C[NNidx*DIM+2]);
      flag = 1;
      if(verboseFlag==0)
	return flag;
    }

  } // End of going through all Q;

  if(verboseFlag==0) // If it reached here with the verbose flag then it hasn't retunred 1 (look 3 lines up)
    printf("     ! ! ! VALIDATION SUCCEEDED ! ! !\n\n");
      
  return flag;
}