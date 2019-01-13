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
void cuNearestNeighbor(float **S, int *SDim, float *Q, int *checkQInBox, int d, float **neighbor) {

  int proccess = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int d3 = d*d*d;
  int d2 = d*d;
  float *q, *c, *nearestPoint;
  float dx, dy, dz, distSqr, dist, nearestDist;
  float sqrDx[3], sqrDy[3], sqrDz[3];
  sqrDx[0]=0; sqrDy[0]=0; sqrDz[0]=0;
  int boxId, gridX, gridY, gridZ, temp;


  for(int idx=proccess; idx<d3; idx+=stride) {
    q = Q+(DIM*idx);
    boxId = checkQInBox[idx];
    nearestDist = 1;        //This is HUGE!
    for(int S_num=0; S_num<SDim[boxId]; S_num++){
      c  = S[boxId] + (DIM * S_num);  //check distance of c anf q;
      dx = q[0] - c[0];
      dy = q[1] - c[1];
      dz = q[2] - c[2];
      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      dist = sqrtf(distSqr);
      if(dist<nearestDist){
	nearestDist = dist;
	nearestPoint = c;
      } // !!Try two nops here as an else???
    } // end of for(int S_num=0; S_num<SDim[boxId]; S_num++)
    neighbor[idx]=nearestPoint;

    // These are the XYZ coordinates of the grid
    gridZ = boxId / d2;
    temp  = boxId % d2;
    gridY = temp / d;
    gridX = temp % d;

    // Now calculate the distance of the point from:
    // the 8 verteces of the grid cube
    // the 12 edges
    // and the 6 faces

    dx       = q[0] - gridX;
    sqrDx[1] = dx*dx;
    sqrDx[2] = (d-dx)*(d-dx);
    dy       = q[1] - gridY;
    sqrDy[1] = dy*dy;
    sqrDy[2] = (d-dy)*(d-dy);
    dz       = q[2] - gridZ;
    sqrDz[1] = dz*dz;
    sqrDz[2] = (d-dz)*(d-dz);
    // Reminder that sqrD(xyz)[0] = 0
      
    for(int zi=0; zi<3; zi++){
      for(int yi=0; yi<3; yi++){
	for(int xi=0; xi<3; xi++){
	  distSqr = sqrDx[xi] + sqrDy[yi] + sqrDz[zi];
	  //cbrtf(distSqr);
	}
      }
    }
      
  } // end of  for(int P_num=0; P_num<P_size[i]; P_num++)
}

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