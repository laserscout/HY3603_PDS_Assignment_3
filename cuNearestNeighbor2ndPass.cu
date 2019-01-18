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

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int boxIdToCheck;

  int d3 = d*d*d;
  int d2 = d*d;
  float *q, *c;
  float q_x, q_y, q_z, dx, dy, dz, distSqr, dist, nearestDist;
  int boxId, nearestIdx;

  int oneOone[] = {-1, 0, 1};

  for(int idx=process; idx<NQ; idx+=stride) {
    if(checkOutside[idx] == 1) { 
      
      q = Q+(DIM*idx);
      q_x = q[0];
      q_y = q[1];
      q_z = q[2];

      boxId = checkedQInBox[idx];
      nearestDist = 1;        // This is HUGE!
      nearestIdx = -1;        // Error checking value
      for(int i=0; i<3; i++) {
        for(int i_d=0; i_d<3; i_d++) {
          for(int i_d2=0; i_d2<3; i_d2++) {

            boxIdToCheck = boxId + oneOone[i] + d*oneOone[i_d] + d2*oneOone[i_d2]; 

            // printf(">Q[%d] checking in box %d\n",idx,boxIdToCheck);

            if(boxIdToCheck < d3 && boxIdToCheck >=0) {
              for(int S_num=S[boxIdToCheck]; S_num<S[boxIdToCheck+1]; S_num++){
                c = C+(S_num*DIM);
                dx = q_x - c[0];
                dy = q_y - c[1];
                dz = q_z - c[2];
                distSqr = dx*dx + dy*dy + dz*dz;
                dist = sqrtf(distSqr);
                if(dist<nearestDist){
                	nearestDist = dist;
                	nearestIdx = S_num;
                } 
              } 
            }
          }
        }
      }
      neighbor[idx] = nearestIdx;    
    }
  }// end of for(int S_num=0; S_num<SDim[boxIdToCheck]; S_num++)
}