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

void prefixSum(int *array, int size) {
	for(int i=1; i<size; i++) 
		array[i] += array[i-1];
}

__global__
void cuInitZero(int *d_boxSize, int n) {

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i=process; i<n; i+=stride){
    d_boxSize[i]=0;
  }
}

__global__
void cuFindBelongsToBox (float *v, int N, int d, int *belongsToBox, int *positionWithinBox, int *boxSize){

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int d2 = d * d;
//   int d3 = d * d * d;
  float x, y, z;

  for(int n=process; n<N; n+=stride){
      x = v[n*DIM];
      y = v[n*DIM+1];
      z = v[n*DIM+2];
      belongsToBox[n] = (int)(x*d) +(int)(y*d)*d +(int)(z*d)*d2; //= boxId;
      positionWithinBox[n] = atomicAdd(&boxSize[belongsToBox[n]], 1);
      // printf("> v[%d]: belongsToBox %d, positionWithinBox=%d, boxSize=%d\n",n,belongsToBox[n],positionWithinBox[n],boxSize[belongsToBox[n]]);
    }
}

__global__
void cuRearrangeV (float *v, float *newV, int N, int d, int *belongsToBox, int *newBelongsToBox, int *positionWithinBox, int *boxStart){

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int position;

  for(int n=process; n<N; n+=stride){
    position   = boxStart[belongsToBox[n]] + positionWithinBox[n];
    newV[DIM*position]   = v[DIM*n];
    newV[DIM*position+1] = v[DIM*n+1];
    newV[DIM*position+2] = v[DIM*n+2];
    newBelongsToBox[position] = belongsToBox[n];
    }
}

int hashing3D(float *v, float **d_v, size_t vSize, int N, int d, int **vPartsStart, int **d_vPartsStart,
              int **vBelongsToBox, int **d_vBelongsToBox, size_t numberOfBlocks, size_t threadsPerBlock)
{

  int d3 = d*d*d;
  int *belongsToBox, *d_newBelongsToBox, *d_belongsToBox, *d_positionWithinBox, *boxStart, *d_boxStart;
  cudaError_t err;
  float *d_newV;

  size_t belongsToBoxSize = N*sizeof(int);
  size_t boxStartSize     = (d3+1)*sizeof(int);

  CUDA_CALL(cudaMalloc(&d_belongsToBox, belongsToBoxSize));
  CUDA_CALL(cudaMalloc(&d_newBelongsToBox, belongsToBoxSize));
  CUDA_CALL(cudaMalloc(&d_positionWithinBox, belongsToBoxSize));
  CUDA_CALL(cudaMalloc(&d_boxStart, boxStartSize));
  CUDA_CALL(cudaMalloc(&d_newV, vSize));

  belongsToBox = (int *)malloc(belongsToBoxSize);
  if(belongsToBox == NULL) {
    printf("Error allocating belongsToBox\n");
    exit(1);
  }
  boxStart = (int *)malloc(boxStartSize);
  if(boxStart == NULL) {
    printf("Error allocating boxStart\n");
    exit(1);
  }  
  // printf("tr:%zu, bl:%zu\n",threadsPerBlock, numberOfBlocks);
  cuInitZero<<<threadsPerBlock, numberOfBlocks>>>(d_boxStart, d3+1);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }

  CUDA_CALL(cudaDeviceSynchronize());

  cuFindBelongsToBox<<<threadsPerBlock, numberOfBlocks>>>
    (*d_v, N, d, d_belongsToBox, d_positionWithinBox, d_boxStart+1);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }

  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(boxStart, d_boxStart, boxStartSize, cudaMemcpyDeviceToHost));

  prefixSum(boxStart,d3);
  
/*  for(int i=0;i<d3;i++)
    printf("%d: %d, ",i, boxStart[i]);
  printf("\n");
*/
  CUDA_CALL(cudaMemcpy(d_boxStart, boxStart, boxStartSize, cudaMemcpyHostToDevice));

  cuRearrangeV<<<threadsPerBlock, numberOfBlocks>>>
    (*d_v, d_newV, N, d, d_belongsToBox, d_newBelongsToBox, d_positionWithinBox, d_boxStart);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaFree(*d_v));
  CUDA_CALL(cudaFree(d_belongsToBox));
  *d_v = d_newV;

  *vPartsStart = boxStart;
  *d_vPartsStart = d_boxStart;
  *vBelongsToBox = belongsToBox;
  *d_vBelongsToBox = d_newBelongsToBox;

  CUDA_CALL(cudaFree(d_positionWithinBox));
    
  return 0;
}

int hashing3D(float *v, float **d_v, size_t vSize, int N, int d, int **vPartsStart, int **d_vPartsStart,
              size_t numberOfBlocks, size_t threadsPerBlock) {

  int *belongsToBox, *d_belongsToBox;
  int ret = hashing3D(v,d_v,vSize,N,d,vPartsStart,d_vPartsStart,&belongsToBox,
		      &d_belongsToBox,numberOfBlocks,threadsPerBlock);
  CUDA_CALL(cudaFree(d_belongsToBox));
  free(belongsToBox);
  return ret;
}