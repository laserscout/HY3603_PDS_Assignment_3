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

void cpuValidation(float *Q, int NQ, float *C, int NC, int *results)
{
	float NNdist, dist;
	int NNidx;

	float xQ, yQ, zQ;
	float xC, yC, zC;
    
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
		}

		if(results[i]/3 != NNidx) {

			printf("     ! ! ! VALIDATION FAILED ! ! !\n");
            printf("-> On Q[%d]: (%1.4f, %1.4f, %1.4f)\n",i, xQ, yQ, zQ);
			printf("Algorithm found C[%d] as the NN, while in fact it was C[%d].\n\n", results[i]/3, NNidx);
		}

	}
    printf("\n");
}