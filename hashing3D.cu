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
void hashingKernel(float *v, int N, int d, int boxIdx, float **pointersToBoxPoints, int size_boxPoints)
{
	int proccess = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int d2 = d * d;
	int d3 = d * d * d;

	float x, y, z;

	for (int i = proccess; i < N; i += stride)
	{
		x = v[i*DIM];
		y = v[i*DIM+1];
		z = v[i*DIM+2];

		if(boxIdx == (int)(x*d) + (int)(y*d2) + (int)(z*d3))
			pointersToBoxPoints[size_boxPoints++] = &v[i*DIM];
	}
}


int hashing3D(float *v, int N, int d, float ****boxes, int **boxesSizes,
              size_t numberOfBlocks, size_t threadsPerBlock)
{
	int d3 = d * d * d;

	float ***boxes_temp, **ptr, **pointersToBoxPoints;
	int *boxesSizes_temp;
	size_t size, size_boxPoints, count;

	CUDA_CALL(cudaMalloc(&boxesSizes_temp, d3 * sizeof(int)) );

	size = d3 * sizeof(float **) + N * sizeof(float*);
	// https://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/

	CUDA_CALL(cudaMalloc(&boxes_temp, size));

	// Alocating memory for float** to hold the pointers pointing the box points of boxIdx, as found by the hashingKernel
	size_boxPoints = N * sizeof(float**);

	CUDA_CALL(cudaMalloc(&pointersToBoxPoints, size_boxPoints)); 

	// counter to hold the sum of points mapped to all boxIdxs < curent i
	count = 0;

	// Reserving the first d3 positions of boxes_temp for the boxes_temp[i] double pointers
	ptr =(float **)boxes_temp + d3;
	
	for(int i=0;i<d3;i++) {

		size_boxPoints = 0;
		
		hashingKernel<<<numberOfBlocks, threadsPerBlock>>>
            (v, N, d, i,pointersToBoxPoints, size_boxPoints);
        
		cudaDeviceSynchronize();
		// size_boxPoints is the number of points that are mapped to boxIdx = i, so
		boxesSizes_temp[i] = size_boxPoints;
		
		// boxes_temp[i] = boxes_temp + sum of points mapped to all boxIdxs < i
		boxes_temp[i] = ptr + count;
		
		// Post incrementing the counter count, so that it gives the correct offset for the next iteration
		count += size_boxPoints;
		
		for(int boxPoints = 0; boxPoints < size_boxPoints; boxPoints++)
			// Placing each pointer of each box's point into the appropriate positions of boxes_temp
			boxes_temp[i][boxPoints] = pointersToBoxPoints[boxPoints];
	}

	// Access of boxes array: boxes[d3][N][0,1,2]; actually it's boxes[d3][boxesSizes[d3]][0,1,2] with boxesSizes[d3] having a sum of N points
	*boxes = boxes_temp;
	*boxesSizes = boxesSizes_temp;
    
    return 0;
}

