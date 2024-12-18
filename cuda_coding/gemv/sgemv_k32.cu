#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CEIL(a,b) ((a) + ((b) - 1)) / (b) //除法，向上取整
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line){
    if(error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
}

__global__ void sgemv_k32(float* A, float* x, float* y, int M, int K){
    int laneID = threadIdx.x % warpSize;    //一个块中选择哪一个线程
    int row = blockIdx.x;                   //选择行
    if(row < M){
        float res = 0.0f;                       //每个线程有一个res，res最大有32个，等于线程的数量，每个线程不一定只计算一次，取决于K
        int kIteration = CEIL(K, warpSize);         //如果一行不止32个数字，则必定有线程要计算多个。

        #pragma unroll
        for(int i = 0; i < kIteration;i++){
            int col = i * warpSize + laneID;
            res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
        }

        for(int offset = warpSize >> 1;offset > 0;offset >>= 1){
            res += __shfl_down_sync(0xFFFFFFFF, res, offset);
        }

        if(laneID == 0) y[row] = res;
    }
}

int main(){
    size_t M = 1024;        //矩阵行
    size_t K = 64;          //矩阵列

    size_t bytes_A = sizeof(float) * M * K;     //矩阵内存大小
    size_t bytes_x = sizeof(float) * K;         //输入向量内存大小
    size_t bytes_y = sizeof(float) * M;         //输出向量内存大小

    float* h_A = (float*)malloc(bytes_A);
    float* h_x = (float*)malloc(bytes_x);
    float* h_y = (float*)malloc(bytes_y);

    float* d_A = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaCheck(cudaMalloc(&d_A, bytes_A));
    cudaCheck(cudaMalloc(&d_x, bytes_x));
    cudaCheck(cudaMalloc(&d_y, bytes_y));

    double duration[2] = {0, 0};
    double GFLOPS[2] = {0, 0};
    double GFLOPs = 2.0 * M * 1 * K;

    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        h_A[i] = (i / K) + 1.0f; // 按行填充1, 2, 3等值
    }

    // 生成x的数据
    for( int i = 0; i < K; i++ ) {
        h_x[i] = 1.0f;
    }

    memset(h_y, 0, M * sizeof(float));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float msecTotal = 0;
    int iter = 10000;

    cudaCheck(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));

    cudaCheck(cudaEventRecord(start));

    int block_size = 32;
    int grid_size = M;
    for(int run = 0;run < iter; run++){
        sgemv_k32<<<grid_size, block_size>>>(d_A, d_x, d_y, M, K);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&msecTotal, start, stop));
    cudaCheck(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    duration[0] = msecTotal / iter;
    GFLOPS[0] = (GFLOPs * 1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);
    
    for (int i = 0; i < 3; i++){
        printf("ans = %f\n", h_y[i]);
    }


    return 0;
}
