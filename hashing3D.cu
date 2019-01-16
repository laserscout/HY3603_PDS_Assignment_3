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
void cuFindBelongsToBox (float *v, int N, int d, int *belongsToBox, int *positionWithinBox, int *boxSize){

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int d2 = d * d;
  int d3 = d * d * d;
  float x, y, z;

  for(int n=process; n<d3; n+=stride)
  	  boxSize[n]=0;

  __syncthreads();

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
void cuPrefixSum (int *array, int size){
// CUDA implementation of the Exclusive Prefix Sum Algorithm for array boxStart
// Reference: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf

  extern __shared__ int temp[];

  int process = threadIdx.x;
  int offset = 1;

  // AS IS, IT WON'T WORK FOR d > 3
  if(size>1024) printf("size>1024: PROBLEM\n");

if (process<size/2) {
	  temp[2*process]   = array[2*process];
	  temp[2*process+1] = array[2*process+1];

	  for(int n = size>>1; n > 0; n >>= 1)
	  {
	  	__syncthreads();

	  	if(process < n)
	  	{
	  		int ai = offset*(2*process+1)-1;
	  		int bi = offset*(2*process+2)-1;

	  		temp[bi] += temp[ai];
	  	}

	  	offset *= 2;
	  }

	  if(process == 0) {
	  	array[size] = temp[size-1];
	    temp[size-1] = 0;
	  }

	  for(int n = 1; n < size; n *= 2)
	  {
	  	offset >>= 1;
	  	__syncthreads();

	  	if(process < n)
	  	{
	  		int ai = offset*(2*process+1)-1;
	  		int bi = offset*(2*process+2)-1;

	  		float t   = temp[ai];
	  		temp[ai]  = temp[bi];
	  		temp[bi] += t;
	  	}
	  }

	  __syncthreads();

	  array[2*process] = temp[2*process];
	  array[2*process+1] = temp[2*process+1];

	  //printf("boxStart[%d]=%d\n",2*process,array[2*process]);
	  //printf("boxStart[%d]=%d\n",2*process+1,array[2*process+1]);
	}
}

__global__
void cuRearrangeV (float *v, int N, int d, int *belongsToBox, int *positionWithinBox, int *boxStart){

  int process = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  int position;

  for(int n=process; n<N; n+=stride){
    position   = boxStart[belongsToBox[n]] + positionWithinBox[n];
    v[DIM*n]   = atomicExch( &v[DIM*position], v[DIM*n]);
    v[DIM*n+1] = atomicExch( &v[DIM*position+1], v[DIM*n+1]);
    v[DIM*n+2] = atomicExch( &v[DIM*position+2], v[DIM*n+2]);
    belongsToBox[n] = atomicExch( &belongsToBox[position], belongsToBox[n]);
    }
}


int hashing3D(float *v, float *d_v, size_t vSize, int N, int d, int **vPartsStart, int **d_vPartsStart,
              int **vBelongsToBox, int **d_vBelongsToBox, size_t numberOfBlocks, size_t threadsPerBlock)
{

  int d3 = d*d*d;
  int *belongsToBox, *d_belongsToBox, *d_positionWithinBox, *boxStart, *d_boxStart;
  cudaError_t err;
  
  size_t belongsToBoxSize = N*sizeof(int);
  size_t boxStartSize     = (d3+1)*sizeof(int);

  CUDA_CALL(cudaMalloc(&d_belongsToBox, belongsToBoxSize));
  CUDA_CALL(cudaMalloc(&d_positionWithinBox, belongsToBoxSize));
  CUDA_CALL(cudaMalloc(&d_boxStart, boxStartSize));

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

  cuFindBelongsToBox<<<threadsPerBlock, numberOfBlocks>>>
    (d_v, N, d, d_belongsToBox, d_positionWithinBox, d_boxStart);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }

  CUDA_CALL(cudaDeviceSynchronize());

  size_t maxNumOfThreads = 1024;
  
  // printf(" ==== Prefix Sum ==== \n");

  // As is, IT WON'T WORK FOR d > 3
  // We will have to try an implementation with more than 1 blocks, 
  // in order to yeild maximum performonce 
  cuPrefixSum<<<1, maxNumOfThreads, boxStartSize>>>
    (d_boxStart, d3);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }

  CUDA_CALL(cudaDeviceSynchronize());

  cuRearrangeV<<<threadsPerBlock, numberOfBlocks>>>
    (d_v, N, d, d_belongsToBox, d_positionWithinBox, d_boxStart);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error \"%s\" at %s:%d\n", cudaGetErrorString(err),
             __FILE__,__LINE__);
      return EXIT_FAILURE;
  }
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(v, d_v, vSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(boxStart, d_boxStart, boxStartSize, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(belongsToBox, d_belongsToBox, belongsToBoxSize, cudaMemcpyDeviceToHost));

  *vPartsStart = boxStart;
  *d_vPartsStart = d_boxStart;
  *vBelongsToBox = belongsToBox;
  *d_vBelongsToBox = d_belongsToBox;

  CUDA_CALL(cudaFree(d_positionWithinBox));
    
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