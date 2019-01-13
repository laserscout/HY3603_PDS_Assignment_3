/**********************************************************************
 *
 * hashing3D.cu -- Hashing the 3D space into d*d*d boxes
 * 
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include "hashing3D.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DIM 3

__global__
void cuFindBelongsToBox (float *v, int N, int d, int *belongsToBox, int *boxDim){

  int proccess = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int d2 = d * d;
  int d3 = d * d * d;
  int boxId;

  float x, y, z;

  for(boxId=proccess; boxId<d3; boxId+=stride)
    boxDim[boxId]=0;

  for(int n=proccess; n<N; n+=stride){
      x = v[n*DIM];
      y = v[n*DIM+1];
      z = v[n*DIM+2];
      boxId = (int)(x*d) + (int)(y*d2) + (int)(z*d3)-1; //THIS IS WRONG
      printf("%d:%d\n",n,boxId);
      belongsToBox[n] = boxId;
      // boxDim[boxId]++;
    }
}

int hashing3D(float *v, float *d_v, size_t vSize, int N, int d, float ***vParts, float ***d_vParts,
              int **partsDim, int **d_partsDim, size_t numberOfBlocks, size_t threadsPerBlock)
{

  int d3 = d*d*d;
  int position, tempBelongs;
  int *boxDim, *d_boxDim, *belongsToBox, *d_belongsToBox;
  float tempX, tempY, tempZ;
  float **box, **d_box;
  
  size_t boxSize          = d3*sizeof(float *);
  size_t boxDimSize       = d3*sizeof(int);
  size_t belongsToBoxSize =  N*sizeof(int);

  CUDA_CALL(cudaMalloc(&d_boxDim, boxDimSize));
  CUDA_CALL(cudaMalloc(&d_box, boxSize));
  CUDA_CALL(cudaMalloc(&d_belongsToBox, belongsToBoxSize));

  boxDim = (int *)malloc(boxDimSize);
  if(boxDim == NULL) {
    printf("Error allocating boxDim\n");
    exit(1);
  }
  box = (float **)malloc(boxSize);
  if(box == NULL) {
    printf("Error allocating box\n");
    exit(1);
  }
  belongsToBox = (int *)malloc(belongsToBoxSize);
  if(belongsToBox == NULL) {
    printf("Error allocating belongsToBox\n");
    exit(1);
  }

  cuFindBelongsToBox<<<threadsPerBlock, numberOfBlocks>>>
    (d_v, N, d, d_belongsToBox, d_boxDim);
  CUDA_CALL(cudaDeviceSynchronize());

  // CUDA_CALL(cudaMemcpy(v, d_v, vSize, cudaMemcpyDevicyToHost));
  CUDA_CALL(cudaMemcpy(belongsToBox, d_belongsToBox, belongsToBoxSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(boxDim, d_boxDim, boxDimSize, cudaMemcpyDeviceToHost));

  position = 0;
  for(int boxId=0; boxId<d3; boxId++){
    box[boxId] = v + DIM*position;
    boxDim[boxId] = 0;
    for(int n=position; n<N; n++){
      if(belongsToBox[n] == boxId) {
	tempX             = v[DIM*position];
	tempY             = v[DIM*position+1];
	tempZ             = v[DIM*position+2];
	v[DIM*position]   = v[DIM*n];
	v[DIM*position+1] = v[DIM*n+1];
	v[DIM*position+2] = v[DIM*n+2];
	v[DIM*n]          = tempX;
	v[DIM*n+1]        = tempY;
	v[DIM*n+2]        = tempZ;
	tempBelongs = belongsToBox[position];
	belongsToBox[position] = belongsToBox[n];
	belongsToBox[n] = tempBelongs;
	position++;
	boxDim[boxId]++;
      }
    }
  }

  CUDA_CALL(cudaMemcpy(d_v, v, vSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_box, box, boxSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_boxDim, boxDim, boxDimSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaFree(d_belongsToBox));
  free(belongsToBox);

  *vParts = box;
  *d_vParts = d_box;
  *partsDim = boxDim;
  *d_partsDim = d_boxDim;
    
  return 0;
}