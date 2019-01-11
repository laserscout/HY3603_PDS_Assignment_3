/**********************************************************************
 *
 * cuNearestNeighbor.cu -- Find the nearest neighbor kernel
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include "cuNearestNeighbor.cu"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// https://devblogs.nvidia.com/cuda-pro-tip-flush-denormals-confidence/
#define SOFTENING 1e-9f


// S has the points that will look for the nearerst neighbor
// P is a gridded representation of the Q points vector
// d3 is the d value cubed. AKA the number of the grids.

__global__
void cuNearestNeighbor(float **S, int * S_size, float ***P, int *P_size, int d) {
  int proccess = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int d3 = d*d*d;
  int d2 = d*d;

  float *q, *c, dx, dy, dz, distSqr, dist, nearestDist;
  int nearestIdx, ;
  

  for(int gridId=proccess; gridId<d3; gridId+=stride) {
    // There are the XYZ coordinates of the grid
    gridZ = gridId / d2;
    temp  = gridId % d2;
    gridY = temp / d;
    gridX = temp % d;
    
    for(int P_num=0; P_num<P_size[i]; P_num++) {
      q = P[gridId][P_num];   //the point that we will be calculating q[0,1,2]
      nearestDist = 1;        //This is HUGE!
      for(int S_num=0; S_num<S_size[gridId]; S_num++){
	c = *S[gridId][3*S_num];  //check distance of c anf q;
	dx = q[0] - c[0];
	dy = q[1] - c[1];
	dz = q[2] - c[2];
	distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
	dist = cbrtf(distSqr);
	if(dist<nearestDist){
	  nearestDist = dist;
	  nearestIdx = S_num;
	}
      }

      // Now calculate the distance of the point from:
      // the 8 verteces of the grid cube [0]
      // the 12 edges [1]
      // and the 6 faces [2]

      for(int zi=0; zi<2; zi++){
	for(int yi=0; yi<2; yi++){
	  for(int xi=0; xi<2; xi++){
	    dx = q[0] - (gridX + xi*d);
	    dy = q[1] - (gridY + yi*d);
	    dz = q[2] - (gridZ + zi*d);
	    distSqr = dx*dx + dy*dy + dz*dz;
	    boundaryDist[0][xi+2*yi+4*zi] = cbrtf(distSqr);
	    distSqr = dx*dx + dy*dy;
	    boundaryDist[1][
	  }
	}
      }
    }
  }
}

S[boxid][i][0,1,2]

// secont way

S_acutal[boxid][3*i+"0, 1, 2"]

S[ 3*( boxid*S_size[boxid] + i ) + "0,1,2"]