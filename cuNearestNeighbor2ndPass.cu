/**********************************************************************
 *
 * cuNearestNeighbor.cu -- Find the nearest neighbor kernel
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include "cuNearestNeighbor.h"
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
void cuNearestNeighbor2ndPass(float *C, int *S, float *Q, int NQ, int *checkedQInBox, int d, int *neighbor, int *checkOutside) {

  __shared__ float total_nearestDist;
  __shared__ int total_nearestIdx;

  int boxIdToCheck;
  int proccess =  blockIdx.x;
  int stride = gridDim.x;

  // int d3 = d*d*d;
  int d2 = d*d;
  float invd = 1/(float)d;
  float *q, *c;
  float dx, dy, dz, distSqr, dist, nearestDist, gridX, gridY, gridZ;
  int boxId, temp, nearestIdx;


  for(int idx=proccess; idx<NQ; idx+=stride) {
    if(checkOutside[idx]) {
      q = Q+(DIM*idx);
      boxId = checkedQInBox[idx];
      nearestDist = 1;        //This is HUGE!
      printf("q[%d]:%1.4f, %1.4f, %1.4f | Belongs to %d\n",idx,q[0],q[1],q[2],boxId);

      // Calculate boxIdToCheck through threadIdx.x
      gridZ = (boxId / d2) * invd;
      temp  = boxId % d2;
      gridY = (temp / d) * invd;
      gridX = (temp % d) * invd;

      if(gridZ==0)
        // Exclude Searching Downwards
      else if(gridZ==d-1)
        // Exclude Searching Upwards

      if(gridY==0)
        // Exclude Searching Left
      else if(gridY==d-1)
        // Exclude Searching Right

      if(gridX==0)
        // Exclude Searching Backwards
      else if(gridX==d-1)
        // Exclude Searching Frontwards



      if(boxIdToCheck != -1) {
        for(int S_num=S[boxIdToCheck]; S_num<S[boxIdToCheck+1]; S_num+=3){
          c = C+(S_num);
          dx = q[0] - c[0];
          dy = q[1] - c[1];
          dz = q[2] - c[2];
          distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
          dist = sqrtf(distSqr);
          if(dist<nearestDist){
          	nearestDist = dist;
          	nearestIdx = S_num;
          } 
        } // end of for(int S_num=0; S_num<SDim[boxIdToCheck]; S_num++)
        
        if(nearestDist<total_nearestDist) {
          atomicExch(&total_nearestDist, nearestDist);
          atomicExch(&total_nearestIdx, nearestIdx);
        }
      }
    }      
  } // end of  for(int P_num=0; P_num<P_size[i]; P_num++)
}