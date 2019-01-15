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
void cuFindBelongsToBox (float *v, int N, int d, int *belongsToBox){

  int proccess = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int d2 = d * d;
  // int d3 = d * d * d;
  int boxId;

  float x, y, z;

  for(int n=proccess; n<N; n+=stride){
      x = v[n*DIM];
      y = v[n*DIM+1];
      z = v[n*DIM+2];
      boxId = (int)(x*d) +(int)(y*d)*d +(int)(z*d)*d2;
      // printf("%d:%d = %d + %d + %d\n",n,boxId,(int)(x*d),(int)(y*d)*d,(int)(z*d)*d*d); 
      belongsToBox[n] = boxId;
    }
}

int hashing3D(float *v, float *d_v, size_t vSize, int N, int d, int **vPartsStart, int **d_vPartsStart,
              int **vBelongsToBox, int **d_vBelongsToBox, size_t numberOfBlocks, size_t threadsPerBlock)
{

  int d3 = d*d*d;
  int position, tempBelongs;
  int *belongsToBox, *d_belongsToBox, *boxStart, *d_boxStart;
  float tempX, tempY, tempZ;
  cudaError_t err;
  
  size_t boxStartSize     = (d3+1)*sizeof(int);
  size_t belongsToBoxSize =  N*sizeof(int);

  CUDA_CALL(cudaMalloc(&d_boxStart, boxStartSize));
  CUDA_CALL(cudaMalloc(&d_belongsToBox, belongsToBoxSize));

  boxStart = (int *)malloc(boxStartSize);
  if(boxStart == NULL) {
    printf("Error allocating boxStart\n");
    exit(1);
  }
  belongsToBox = (int *)malloc(belongsToBoxSize);
  if(belongsToBox == NULL) {
    printf("Error allocating belongsToBox\n");
    exit(1);
  }
  
  // printf("tr:%zu, bl:%zu\n",threadsPerBlock, numberOfBlocks);

  cuFindBelongsToBox<<<threadsPerBlock, numberOfBlocks>>>
    (d_v, N, d, d_belongsToBox);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  CUDA_CALL(cudaDeviceSynchronize());

  // CUDA_CALL(cudaMemcpy(v, d_v, vSize, cudaMemcpyDevicyToHost));
  CUDA_CALL(cudaMemcpy(belongsToBox, d_belongsToBox, belongsToBoxSize, cudaMemcpyDeviceToHost));

  position = 0;
  for(int boxId=0; boxId<d3; boxId++){
    boxStart[boxId] = position;
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
      }
    }
  }
  boxStart[d3]=N;

  CUDA_CALL(cudaMemcpy(d_v, v, vSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_boxStart, boxStart, boxStartSize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_belongsToBox, belongsToBox, belongsToBoxSize, cudaMemcpyHostToDevice));

  *vPartsStart = boxStart;
  *d_vPartsStart = d_boxStart;
  *vBelongsToBox = belongsToBox;
  *d_vBelongsToBox = d_belongsToBox;
    
  return 0;
}

int hashing3D(float *v, float *d_v, size_t vSize, int N, int d, int **vPartsStart, int **d_vPartsStart,
              size_t numberOfBlocks, size_t threadsPerBlock) {

  int *belongsToBox, *d_belongsToBox;
  int ret = hashing3D(v,d_v,vSize,N,d,vPartsStart,d_vPartsStart,&belongsToBox,
		      &d_belongsToBox,numberOfBlocks,threadsPerBlock);
  CUDA_CALL(cudaFree(d_belongsToBox));
  free(belongsToBox);
  return ret;
}