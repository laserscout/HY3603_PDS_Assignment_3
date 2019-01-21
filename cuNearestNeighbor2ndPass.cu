/**********************************************************************
 *
 * cuNearestNeighbor2ndPass.h -- Kernel 2nd pass for the find 
 *                                  of the nearest neighbor
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
void cuNearestNeighbor2ndPass(float *C, int *S, float *Q, int NQ, int *checkedQInBox, int d, int *neighbor, int *checkOutside) {

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int boxIdToCheck;

  int d3 = d*d*d;
  int d2 = d*d;
  float invd = 1/(float)d;
  float *q, *c;
  float q_x, q_y, q_z, dist, nearestDist;
  int boxId, nearestIdx;
  float sqrDx[3], sqrDy[3], sqrDz[3];
  sqrDx[0]=0; sqrDy[0]=0; sqrDz[0]=0;


  int oneOone[] = {0, -1, 1};

  for(int idx=process; idx<checkOutside[NQ]; idx+=stride) {

    q = Q+(DIM*checkOutside[idx]);
    q_x = q[0];
    q_y = q[1];
    q_z = q[2];

    // printf("%d: %d",idx, checkOutside[idx]);
    boxId = checkedQInBox[checkOutside[idx]];
    nearestDist = 1;        // This is HUGE!
    nearestIdx = -1;        // Error checking value

    float gridZ = (boxId / d2) * invd;
      int temp  = boxId % d2;
      float gridY = (temp / d) * invd;
      float gridX = (temp % d) * invd;

      float dx = q_x - gridX;
      sqrDx[1] = dx*dx;
      sqrDx[2] = (invd-dx)*(invd-dx);
      float dy = q_y - gridY;
      sqrDy[1] = dy*dy;
      sqrDy[2] = (invd-dy)*(invd-dy);
      float dz = q_z - gridZ;
      sqrDz[1] = dz*dz;
      sqrDz[2] = (invd-dz)*(invd-dz);
      // Reminder that sqrD(xyz)[0] = 0
    
    for(int i=0; i<3; i++) {
      for(int i_d=0; i_d<3; i_d++) {
        for(int i_d2=0; i_d2<3; i_d2++) {

          boxIdToCheck = boxId + oneOone[i] + d*oneOone[i_d] + d2*oneOone[i_d2]; 

	  if(nearestDist > (sqrDx[i] + sqrDy[i_d] + sqrDz[i_d2]) &&
	     boxIdToCheck < d3 && boxIdToCheck >=0) {
	    for(int S_num=S[boxIdToCheck]; S_num<S[boxIdToCheck+1]; S_num++){
	      c = C+(S_num*DIM);
	      dx = q_x - c[0];
	      dy = q_y - c[1];
	      dz = q_z - c[2];
	      dist = dx*dx + dy*dy + dz*dz;
	      if(dist<nearestDist){
        	nearestDist = dist;
        	nearestIdx = S_num;
	      } 
	    } 
	  }
          // printf(">Q[%d] checking in box %d\n",idx,boxIdToCheck);
        }
      }
    }
    neighbor[checkOutside[idx]] = nearestIdx; 
  }// end of for(int S_num=0; S_num<SDim[boxIdToCheck]; S_num++)
}