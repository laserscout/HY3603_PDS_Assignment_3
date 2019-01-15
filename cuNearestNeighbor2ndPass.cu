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

__constant__ int d_tensorVector0[3];
__constant__ int d_tensorVector1[3];
__constant__ int d_tensorVector2[3];

// S has the points that will look for the nearerst neighbor
// P is a gridded representation of the Q points vector
// d3 is the d value cubed. AKA the number of the grids.

__global__
void cuNearestNeighbor2ndPass(float *C, int *S, float *Q, int NQ, int *checkedQInBox, int d, int *neighbor, char *checkOutside) {

  __shared__ float total_nearestDist;
  __shared__ int total_nearestIdx;

  int boxIdToCheck;
  int proccess =  blockIdx.x;
  int stride = gridDim.x;

  int d3 = d*d*d;
  // int d2 = d*d;
  float *q, *c;
  float dx, dy, dz, distSqr, dist, nearestDist;
  int boxId, nearestIdx;


  for(int idx=proccess; idx<NQ; idx+=stride) {
    if(checkOutside[idx] == 1) { 
    // Probably "== 1" is not needed, but since type variable is char, 
    //it was introduced and may be dumped if it operates without it 

      // Could have been 0 as well
      if(threadIdx.x==13) {
        total_nearestDist = 1.000000;
        total_nearestIdx = -1;
      }
      __syncthreads();

      q = Q+(DIM*idx);
      boxId = checkedQInBox[idx];
      nearestDist = 1;        //This is HUGE!

      // Calculate the boxIdToCheck of each thread, depending on its Idx
      int div9 = (int)threadIdx.x/9;
      int mod9 = (int)threadIdx.x%9;
      int div3 = (int)mod9/3;
      int mod3 = (int)mod9%3;
      boxIdToCheck = boxId + d_tensorVector0[mod3] + d_tensorVector1[div3] + d_tensorVector2[div9]; 

      if(boxIdToCheck < d3 && boxIdToCheck >=0) {
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

      if(threadIdx.x==13)
        neighbor[idx] = total_nearestIdx;  
    }        
  } // end of  for(int P_num=0; P_num<P_size[i]; P_num++)

}