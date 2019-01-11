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
#define x 0
#define y 1
#define z 2


// S has the points that will look for the nearerst neighbor
// P is a gridded representation of the Q points vector
// d3 is the d value cubed. AKA the number of the grids.

__global__
void cuNearestNeighbor(float **S, int * S_size, float ***P, int *P_size,
		       int d ,float **boundaryDist) {

  int proccess = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int d3 = d*d*d;
  int d2 = d*d;
  float *q, *c, dx, dy, dz, distSqr, dist, nearestDist;
  float sqrDx[3], sqrDy[3], sqrDz[3];
  sqrDx[0]=0; sqrDy=0; sqrDz=0;
  int nearestIdx;
  
  for(int gridId=proccess; gridId<d3; gridId+=stride) {
    for(int P_num=0; P_num<P_size[i]; P_num++) {
      q = P[gridId][P_num];   //the point that we will be calculating q[0,1,2]
      nearestDist = 1;        //This is HUGE!
      for(int S_num=0; S_num<S_size[gridId]; S_num++){
	c = *S[gridId][3*S_num];  //check distance of c anf q;
	dx = q[x] - c[x];
	dy = q[y] - c[y];
	dz = q[z] - c[z];
	distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
	dist = cbrtf(distSqr);
	if(dist<nearestDist){
	  nearestDist = dist;
	  nearestIdx = S_num;
	}
      } // end of for(int S_num=0; S_num<S_size[gridId]; S_num++

      // These are the XYZ coordinates of the grid
      gridZ = gridId / d2;
      temp  = gridId % d2;
      gridY = temp / d;
      gridX = temp % d;

      // Now calculate the distance of the point from:
      // the 8 verteces of the grid cube
      // the 12 edges
      // and the 6 faces

      dx       = q[x] - gridX;
      sqrDx[1] = dx*dx;
      sqrDx[2] = (d-dx)*(d-dx);
      dy       = q[y] - gridY;
      sqrDy[1] = dy*dy;
      sqrDy[2] = (d-dy)*(d-dy);
      dz       = q[z] - gridZ;
      sqrDz[1] = dz*dz;
      sqrDz[2] = (d-dz)*(d-dz);
      // Reminder that sqrD(xyz)[0] = 0
      
      for(int zi=0; zi<3; zi++){
	for(int yi=0; yi<3; yi++){
	  for(int xqi=0; xi<3; xi++){
	    distSqr = sqrDx[xi] + sqrDy[yi] + sqrDz[zi];
	    boundaryDist[P_num][xi+3*yi+9*zi] = cbrtf(distSqr);
	  }
	}
      }
      
    } // end of  for(int P_num=0; P_num<P_size[i]; P_num++)
  } // end of for(int gridId=proccess; gridId<d3; gridId+=stride)
}

#undef x
#undef y
#undef z

// if(dx<nearestDist) { // It's near the left side x face
//   //code
//   if(cbrtf(dx*dx+dy*dy)<nearestDist) { // left x, bottom y edge
//     //code
//   }
//   else if(cbrtf(dx*dx+
// 		}
// 	  else if(d-dx<nearestDist) { // It's near the right side x face
// 	  }
      
// 	  if(dy<nearestDist) { // Left y face
// 	  }
// 	  else if(d-dy<nearestDist) { // Right y face
// 	  }
      
// 	  if(dz<nearestDist) { // Left z face
// 	  }
// 	  else if(d-dz<nearestDist) { // Right z face
// 	  }

// S[boxid][i][0,1,2]

// // secont way

// S_acutal[boxid][3*i+"0, 1, 2"]

// S[ 3*( boxid*S_size[boxid] + i ) + "0,1,2"]