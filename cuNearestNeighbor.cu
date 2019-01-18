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
// d3 is the d value cubed. AKA the number of the grid boxes.

__global__
void cuNearestNeighbor(float *C, int *S, float *Q, int NQ, int *checkQInBox, int d, int *neighbor, char *checkOutside) {

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  // int stride = blockDim.x * gridDim.x;

  // int d3 = d*d*d;
  int d2 = d*d;
  float invd = 1/(float)d;
  float *q, *c;
  float q_x, q_y, q_z;
  float dx, dy, dz, distSqr, dist, nearestDist, gridX, gridY, gridZ;
  int boxId, temp, nearestIdx;

  // for(int idx=process; idx<NQ; idx+=stride) {
    q = Q+(DIM*process);
    q_x = q[0];
    q_y = q[1];
    q_z = q[2];

    boxId = checkQInBox[process];
    nearestDist = 1;        // This is HUGE!
    nearestIdx = -1;        // Error checking value
    // printf("q[%d]:%1.4f, %1.4f, %1.4f | Belongs to %d\n",process,q[0],q[1],q[2],boxId);
    for(int S_num=S[boxId]; S_num<S[boxId+1]; S_num++){
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
    } // end of for(int S_num=0; S_num<SDim[boxId]; S_num++)
    neighbor[process]=nearestIdx;
   
    // These are the XYZ coordinates of the grid box
    gridZ = (boxId / d2) * invd;
    temp  = boxId % d2;
    gridY = (temp / d) * invd;
    gridX = (temp % d) * invd;

    // Now calculate the distances of the point from the 6 faces
    dx = q[0] - gridX;
    dy = q[1] - gridY;
    dz = q[2] - gridZ;
    
    /*
    Here remove checkOutside and perform the 26 checks as in older version 
    (nearestKernel it was I think) then hold the 26 results in a local array 
    and merge this function with 2nd Pass and this way we are good to go 
    with a single pass ! ! ! And reduced checks ! ! !
    */
    if( (dx)<nearestDist || (invd-dx)<nearestDist ||
      	(dy)<nearestDist || (invd-dy)<nearestDist ||
      	(dz)<nearestDist || (invd-dz)<nearestDist  )
      checkOutside[process]=1;
    else
      checkOutside[process]=0;      
      
  // } // end of  for(int P_num=0; P_num<P_size[i]; P_num++)
}
