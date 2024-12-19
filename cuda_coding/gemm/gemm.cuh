#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

#define CEIL(a, b) ((a + (b - 1)) / (b))
#define FLOAT4(a) (*(float4*)(&(a)))
#define cuda_check(err) _cuda_check(err, __FILE__, __LINE__)

void _cuda_check(cudaError_t error, const char* file, int line){
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;    
}

