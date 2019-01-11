/**********************************************************************
 *
 * hashing3D.cu -- Hashing the 3D space into d*d*d boxes
 * 
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DIM 3

__global__
void hashingKernel(float *v, int N, int d, float ***boxes, int *boxesSizes)
{
	int proccess = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int d2 = d * d;
	int d3 = d * d * d;

	int boxIdx;
	float x, y, z;

	for (int i = process; i < N; i += stride)
	{
		x = v[i*DIM];
		y = v[i*DIM+1];
		z = v[i*DIM+2];

		boxIdx = (int)(x*d) + (int)(y*d2) + (int)(z*d3);
		boxes[boxIdx][boxesSizes[boxIdx]++] = &v[i*DIM];
	}
}


__global__
void InitBoxesSizes(int *boxesSizes, int N)
{
	int proccess = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i = process; i < N; i += stride)
		boxesSizes[i] = 0;
		
}


void hashing3D(float *v, int N, int d, float ***boxes, int *boxesSizes)
{
	int d3 = d * d * d;

	float ***boxes_temp, **ptr;
	int *boxesSizes_temp;
	size_t size;

	CUDA_CALL(cudaMallocManged(&boxesSizes_temp, d3 * sizeof(int)));

	CUDA_CALL(InitBoxesSizes<<<numberOfBlocks, threadsPerBlock>>>(boxesSizes_temp, d3));

	size = d3 * sizeof(float **) + N * sizeof(float*);

	CUDA_CALL(cudaMallocManged(&boxes_temp, size));

	ptr =(float **)boxes_temp + d3;
	for(int i=0;i<d3;i++)
		boxes_temp[i] = ptr + i*N/d3;
	// https://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/

	// Making sure that InitBoxesSizes is done initializing boxesSizes_temp before proceeding to hashingKernel launch
	cudaDeviceSynchronize();

	CUDA_CALL(hashingKernel<<<numberOfBlocks, threadsPerBlock>>>(v, N, d, boxes_temp, boxesSizes_temp));

	// Access of boxes array: boxes[d3][N]; actually it's boxes[d3][boxesSizes[d3]] with boxesSizes[d3] having a sum of N points
	boxes = boxes_temp;

	boxesSizes = boxesSizes_temp;
}
