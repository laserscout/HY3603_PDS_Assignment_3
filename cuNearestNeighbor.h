/**********************************************************************
 *
 * cuNearestNeighbor.h -- Find the nearest neighbor kernel
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef CU_NEAREST_NEIGHBOR_H
#define CU_NEAREST_NEUGHBOR_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__
void cuNearestNeighbor(float *C, int *S, float *Q, int NQ, int *checkQInBox, int d, int *neighbor, char *checkOutside);


#endif /* CU_NEAREST_NEIGHBOR_H */
