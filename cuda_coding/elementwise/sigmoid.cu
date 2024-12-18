#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define cuda_check(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return;
}

typedef struct 
{
    float a, b, c, d, e, f, g, h;
} float8;


__global__ void Float8Sigmoid(const int N, float* in, float* out){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ( N + 7 )/ 8) return;
    printf("kk%d\n", idx);
    float8 temp_in = *(float8*)(&in[idx * 8]);
    float8 temp_out;
    
    temp_out.a = 1.0f / (1.0f + exp(-temp_in.a));
    temp_out.b = 1.0f / (1.0f + exp(-temp_in.b));
    temp_out.c = 1.0f / (1.0f + exp(-temp_in.c));
    temp_out.d = 1.0f / (1.0f + exp(-temp_in.d));
    temp_out.e = 1.0f / (1.0f + exp(-temp_in.e));
    temp_out.f = 1.0f / (1.0f + exp(-temp_in.f));
    temp_out.g = 1.0f / (1.0f + exp(-temp_in.g));
    temp_out.h = 1.0f / (1.0f + exp(-temp_in.h));

    *(float8*)(&out[idx * 8]) = temp_out;
}

int main(){
    constexpr int N = 7;
    float* in_h = (float*)malloc(N * sizeof(float));
    float* out_h = (float*)malloc(N * sizeof(float));

    for(int i = 0;i < N;i++){
        in_h[i] = i;
    }

    float* in_d = nullptr;
    float* out_d = nullptr; 
    cuda_check(cudaMalloc((void**)&in_d, N * sizeof(float)));
    cuda_check(cudaMalloc((void**)&out_d, N * sizeof(float)));
    cuda_check(cudaMemcpy(in_d, in_h, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 1024;
    int grid_size = (N + 8 * block_size - 1) / (8 * block_size);

    Float8Sigmoid<<<block_size, grid_size>>>(N, in_d, out_d);

    cuda_check(cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0;i < N;i++){
        printf("in:%f, out%f\n", in_h[i], out_h[i]);
    }
    return 0;
}
