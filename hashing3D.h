/**********************************************************************
 *
 * hashing3D.h --  Hashing the 3D space into d*d*d boxes
 * 
 * Frank Blanning <frankgou@auth.gr>
 * John Flionis <iflionis@auth.gr>
 *
 **********************************************************************/

#ifndef HASHING3D_H
#define HASHING3D_H

int hashing3D(float *v, float *d_v, size_t vSize, int N, int d, int **vPartsStart, int **d_vPartsStart,
              size_t numberOfBlocks, size_t threadsPerBlock);

int hashing3D(float *v, float *d_v, size_t vSize, int N, int d, int **vPartsStart, int **d_vPartsStart,
              int **vBelongsToBox, int **d_vBelongsToBox, size_t numberOfBlocks, size_t threadsPerBlock);

#endif /* HASHING3D_H */

#ifndef CUDA_CALL
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error \"%s\" at %s:%d\n", \
    cudaGetErrorString(cudaGetLastError()), \
    __FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#endif

