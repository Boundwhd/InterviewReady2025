#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CEIL(a,b) ((a) + (b-1)) / (b)
#define cuda_check(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        printf("Cuda ERROR at file %s(line %d) : \n%s\n", file, line, cudaGetErrorString(err));
    }
    return;
}

__global__ void sgemv_k64(float* A, float* x, float*y, const int M, const int K){
    int laneID = threadIdx.x % warpSize;    //线程的索引    0 - 31
    int warpID = threadIdx.x / warpSize;    //线程束索引    0 - 1
    int row = blockIdx.x;

    if (row < M){
        float res = 0.0f;
        int iteration = CEIL(K, (2 * warpSize));

        for (int i = 0; i < iteration; i++){
            int col = i * 2 * warpSize + warpID * warpSize + laneID;
            if (col < K) {
                res += A[row * K + col] * x[col];
            }
        }   
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            res += __shfl_down_sync(0xFFFFFFFF, res, offset);
        }
        if (laneID == 0) {
            atomicAdd(&y[row], res);
        }
    }
}

int main(){
    size_t M = 10;
    size_t K = 1280;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_x = sizeof(float) * K;
    size_t bytes_y = sizeof(float) * M;
    float* h_A  = (float*)malloc(bytes_A);
    float* h_x  = (float*)malloc(bytes_x);
    float* h_y  = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    cuda_check(cudaMalloc(&d_A, bytes_A));
    cuda_check(cudaMalloc(&d_x, bytes_x));
    cuda_check(cudaMalloc(&d_y, bytes_y));

    for (int i = 0; i < M * K; i++){
        h_A[i] = float(i / K) + 1;
    }
    for (int i = 0; i < K; i++){
        h_x[i] = 1.0f;
    }
    memset(h_y, 0, bytes_y);

    cuda_check(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));

    int block_size = 64;
    int grid_size = M;
    sgemv_k64<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);

    cuda_check(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    for (int i = 0;i < M; i ++){
        printf("ans = %f\n", h_y[i]);
    }

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(h_A);
    cudaFree(h_x);
    cudaFree(h_y);
    return 0;
}