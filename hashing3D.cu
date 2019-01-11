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
void hashingKernel(float *v, int N, int d, int boxIdx, float **pointersToBoxPoints, int size_boxPoints)
{
	int proccess = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int d2 = d * d;
	int d3 = d * d * d;

	float x, y, z;

	for (int i = process; i < N; i += stride)
	{
		x = v[i*DIM];
		y = v[i*DIM+1];
		z = v[i*DIM+2];

		if(boxIdx == (int)(x*d) + (int)(y*d2) + (int)(z*d3))
			pointersToBoxPoints[size_boxPoints++] = &v[i*DIM];
	}
}


void hashing3D(float *v, int N, int d, float ***boxes, int *boxesSizes)
{
	int d3 = d * d * d;

	float ***boxes_temp, **ptr, **pointersToBoxPoints;
	int *boxesSizes_temp;
	size_t size, size_boxPoints, count;

	CUDA_CALL(cudaMallocManged(&boxesSizes_temp, d3 * sizeof(int)));

	size = d3 * sizeof(float **) + N * sizeof(float*);
	// https://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/

	CUDA_CALL(cudaMallocManged(&boxes_temp, size));

	size_boxPoints = N * sizeof(float*);

	CUDA_CALL(cudaMallocManged(&pointersToBoxPoints, size_boxPoints)); 

	// Making sure that InitBoxesSizes is done initializing boxesSizes_temp before proceeding to hashingKernel launch
	cudaDeviceSynchronize();

	count = 0;
	ptr =(float **)boxes_temp + d3;
	for(int i=0;i<d3;i++) {
		size_boxPoints = 0;
		CUDA_CALL(hashingKernel<<<numberOfBlocks, threadsPerBlock>>>(v, N, d, i, pointersToBoxPoints, size_boxPoints));
		boxesSizes_temp[i] = size_boxPoints;
		boxes_temp[i] = ptr + count;
		count += size_boxPoints;
		for(int boxPoints = 0; boxPoints < size_boxPoints; boxPoints++)
			boxes_temp[i][boxPoints] = &pointersToBoxPoints[boxPoints];
	}

	// Access of boxes array: boxes[d3][N]; actually it's boxes[d3][boxesSizes[d3]] with boxesSizes[d3] having a sum of N points
	boxes = boxes_temp;

	boxesSizes = boxesSizes_temp;
}
