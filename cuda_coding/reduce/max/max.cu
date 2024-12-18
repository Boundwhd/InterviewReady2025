#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include <random>
#include "./include/utils.cuh"

float max_cpu(float* input, float* output, int N) {
    *output = *(std::max_element(input, input + N));
    return *output;
}
/*
同一时刻很多线程（每个block中的warp[0] lane[0]都在往output写，所以为了防止线程冲突，需要使用原子交换，看看这一时刻是不是没有变化）
*/
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;  // address转为int指针
    int old = *address_as_i;  // address中的旧值，用int解码
    int assumed;
    do {
        assumed = old;  // assumed存储旧值
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

//保证线程的总个数要>=元素数量

__global__ void max(float* input, float* output, int N) {
    __shared__ float s_mem[32];     //共享内存，允许一个block上的线程共享;开了32说明最多有32个warp，32*32 = 1024;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;        //线程标识
    int warpID = threadIdx.x / warpSize;                    //基于一个block上的线程束标识
    int laneID = threadIdx.x % warpSize;                    //基于一个block上的线程束上的线程标识

    // 求max
    float val = (idx < N) ? input[idx] : (-FLT_MAX);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val = fmax(val, __shfl_down_sync(0XFFFFFFFF, val, offset));
    }
    if (laneID == 0) s_mem[warpID] = val;
    __syncthreads();
    
    if(warpID == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneID < warpNum) ? s_mem[laneID] : (-FLT_MAX);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (laneID == 0) atomicMax(output, val);
    }
}


int main(){
    size_t N = 40960;
    int repeat_times = 10000;

    float* input = (float*)malloc(sizeof(float) * N);
    for(int i = 0; i < N - 1; i++){
        input[i] = 1;
    }
    input[N-1] = 2;
    float* output_ref = (float*)malloc(1 * sizeof(float));
    float ans = 0;
    float total_time = TIME_RECORD(repeat_times, ([&]{ans = max_cpu(input, output_ref, N);}));
    printf("[max_cpu]: total_time_h = %f ms\n max_value = %f\n", total_time, ans);

    float* output = (float*)malloc(sizeof(float) * 1);
    output[0] = 0;

    float* in_d = nullptr;
    float* out_d = nullptr;
    cudaCheck(cudaMalloc(&in_d, sizeof(float) * N));
    cudaCheck(cudaMalloc(&out_d, sizeof(float) * 1));

    cudaCheck(cudaMemcpy(in_d, input, sizeof(float) * N, cudaMemcpyHostToDevice));
    // cudaCheck(cudaMemcpy(out_d, output, sizeof(float) * 1, cudaMemcpyHostToDevice));

    int block_size = 128;
    int grid_size = 320;

    float time2 = TIME_RECORD(repeat_times, ([&]{max<<<grid_size, block_size>>>(in_d, out_d, N);}));
    printf("[max_gpu]: total_time_h = %f ms\n", time2);
    cudaMemcpy(output, out_d, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("output = %f\n", *output);
    return 0;
}