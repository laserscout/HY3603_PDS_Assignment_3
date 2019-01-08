/**********************************************************************
 *
 * main.cu -- main function for the NN in cuda
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis < @auth.gr>
 *
 **********************************************************************/

#include <stdio.h>
#include "cuNearestNeighborHelper.h"


int main () {

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  int deviceId;
  cudaDeviceProp props;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);
  
  threadsPerBlock = 8*props.warpSize;
  numberOfBlocks  = 50*props.multiProcessorCount;

  return 0;
}