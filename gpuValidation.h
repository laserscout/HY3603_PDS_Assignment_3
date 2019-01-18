/**********************************************************************
 *
 * gpuValidation.h -- gpuValidation function for the NNs of Q as 
 * 					  calculated via CUDA
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/
#ifndef GPUVALIDATION_H
#define GPUVALIDATION_H

int gpuValidation(float *Q, int NQ, float *C, int NC, int *results, char verboseFlag, size_t numberOfBlocks, size_t threadsPerBlock);

#endif /* GPUVALIDATION_H */
