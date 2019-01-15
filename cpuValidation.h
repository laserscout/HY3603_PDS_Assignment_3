/**********************************************************************
 *
 * cpuValidation.h -- cpuValidation function for the NNs of Q as 
 * 					  calculated via CUDA
 *
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/
#ifndef CPUVALIDATION_H
#define CPUVALIDATION_H

int cpuValidation(float *Q, int NQ, float *C, int NC, int *results, char verboseFlag);

#endif /* CPUVALIDATION_H */
