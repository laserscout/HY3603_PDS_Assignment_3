/**********************************************************************
 *
 * cuNearestNeighbor2ndPass.h -- Kernel 2nd pass for the find 
 * 									of the nearest neighbor
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef CU_NEAREST_NEIGHBOR_2ND_PASS_H
#define CU_NEAREST_NEIGHBOR_2ND_PASS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__
void cuNearestNeighbor2ndPass(float *C, int *S, float *Q, int NQ, int *checkQInBox, int d, int *neighbor, int *checkOutside);


#endif /* CU_NEAREST_NEIGHBOR_2ND_PASS_H */
