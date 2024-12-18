#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT4(a) *(float4*)(&(a))
#define CEIL(a,b) ((a+b-1)/(b))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line){
    if(error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
}

__global__ void elementwise_add_float4(float* a, float* b, float* c, int N){
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N) return;

    float4 tem_a = FLOAT4(a[idx]);
    float4 tem_b = FLOAT4(b[idx]);
    float4 tem_c;
    tem_c.x = tem_a.x + tem_b.x;
    tem_c.y = tem_a.y + tem_b.y;
    tem_c.z = tem_a.z + tem_b.z;
    tem_c.w = tem_a.w + tem_b.w;
    FLOAT4(c[idx]) = tem_c;
}

int main(){
    constexpr int N = 65536;
    float* a_h = (float*)malloc(N * sizeof(float));     //cpu上分配内存
    float* b_h = (float*)malloc(N * sizeof(float));
    float* c_h = (float*)malloc(N * sizeof(float));
    for (int i = 0;i < N;i++){
        a_h[i] = i;
        b_h[i] = N - i - 1;
    }

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;
    cudaCheck(cudaMalloc((void**)&a_d, N * sizeof(float)));     //GPU上分配内存
    cudaCheck(cudaMalloc((void**)&b_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&c_d, N * sizeof(float)));
    cudaCheck(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));     //CPU数据搬运到GPU上
    cudaCheck(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));     

    int block_size = 65536;
    int grid_size = 16;

    // 创建事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    elementwise_add_float4<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    cudaEventRecord(stop, 0);

    // 等待核函数执行完毕
    cudaEventSynchronize(stop);

    // 计算时间差（毫秒）
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %f ms\n", milliseconds);

    cudaCheck(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    // printf("a_h:\n");
    // for (int i = 0; i < N; i++ ) {
    //     if (i == N-1) printf("%f\n", a_h[i]);
    //     else printf("%f ", a_h[i]);
    // }
    // printf("b_h:\n");
    // for (int i = 0; i < N; i++ ) {
    //     if (i == N-1) printf("%f\n", b_h[i]);
    //     else printf("%f ", b_h[i]);
    // }
    // printf("c_h:\n");
    // for (int i = 0; i < N; i++ ) {
    //     if (i == N-1) printf("%f\n", c_h[i]);
    //     else printf("%f ", c_h[i]);
    // }
    return 0;
}

