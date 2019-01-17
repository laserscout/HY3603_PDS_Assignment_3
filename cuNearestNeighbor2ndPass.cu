/**********************************************************************
 *
 * cuNearestNeighbor.cu -- Find the nearest neighbor kernel
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include "cuNearestNeighbor2ndPass.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// https://devblogs.nvidia.com/cuda-pro-tip-flush-denormals-confidence/
#define SOFTENING 1e-9f
#define DIM 3

// S has the points that will look for the nearerst neighbor
// P is a gridded representation of the Q points vector
// d3 is the d value cubed. AKA the number of the grids.

__global__
void cuNearestNeighbor2ndPass(float *C, int *S, float *Q, int NQ, int *checkedQInBox, int d, int *neighbor, char *checkOutside) {

  __shared__ float total_nearestDist[27];
  __shared__ int total_nearestIdx[27];

  int boxIdToCheck;
  int process =  blockIdx.x;
  int stride = gridDim.x;

  int d3 = d*d*d;
  int d2 = d*d;
  float *q, *c;
  float dx, dy, dz, distSqr, dist, nearestDist;
  int boxId, nearestIdx;

  int oneOone[] = {-1, 0, 1};

  for(int idx=process; idx<NQ; idx+=stride) {
    if(checkOutside[idx] == 1) { 

      q = Q+(DIM*idx);
      boxId = checkedQInBox[idx];
      nearestDist = 1;        // This is HUGE!
      nearestIdx = -1;        // Error checking value
      // Calculate the boxIdToCheck of each thread, depending on its Idx
      int div9 = (int)threadIdx.x/9;
      int mod9 = (int)threadIdx.x%9;
      int div3 = (int)mod9/3;
      int mod3 = (int)mod9%3;
      boxIdToCheck = boxId + oneOone[mod3] + d*oneOone[div3] + d2*oneOone[div9]; 

      // printf(">Q[%d] checking in box %d\n",idx,boxIdToCheck);

      if(boxIdToCheck < d3 && boxIdToCheck >=0) {
        for(int S_num=S[boxIdToCheck]; S_num<S[boxIdToCheck+1]; S_num++){
          c = C+(S_num*DIM);
          dx = q[0] - c[0];
          dy = q[1] - c[1];
          dz = q[2] - c[2];
          distSqr = dx*dx + dy*dy + dz*dz;
          dist = sqrtf(distSqr);
          if(dist<nearestDist){
          	nearestDist = dist;
          	nearestIdx = S_num;
          } 
        } // end of for(int S_num=0; S_num<SDim[boxIdToCheck]; S_num++)

      }
      total_nearestDist[threadIdx.x] = nearestDist;
      total_nearestIdx [threadIdx.x] = nearestIdx;

      __syncthreads();
      nearestDist = 1;
      for(int rubix=0;rubix<27;rubix++){
	if(total_nearestDist[rubix]<nearestDist) {
	  nearestDist = total_nearestDist[rubix];
	  nearestIdx = total_nearestIdx[rubix];
	}
      }
      neighbor[idx] = nearestIdx;
    }        
  } // end of  for(int P_num=0; P_num<P_size[i]; P_num++)

}