#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define CEIL(a,b) (((a) + (b) - 1) / (b))
#define cudacheck(err) _cudacheck(err, __FILE__, __LINE__);

void _cudacheck(cudaError_t err, const char* file, int line){
    if(err != cudaSuccess){
        printf("error at file%s, line%d\n info:%s\n", file, line, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void setToNegativeMax(float* d_value){   //把一个地址的内容改成最小值
    *d_value = -FLT_MAX;
}    

__device__ static float atomicMax(float* address, float val){
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assume;
    do {
        assume = old;
        old = atomicCAS(address_as_i, assume, __float_as_int(fmax(val, __int_as_float(assume))));
    } while(assume != old);
    return __int_as_float(old);
}

//求长为N的数组上的最大值
__global__ void max_kernel(float* input, float* output, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // 求M(max)
    float val = (idx < N) ? input[idx] : (-FLT_MAX);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();

	if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : (-FLT_MAX);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (laneId == 0) atomicMax(output, val);
    }
}

__global__ void sum_kernel(float* input, float* sum, float* max_val, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;    //获得每个线程的索引    这里的线程可能不是在一个block上的
    int warpId = threadIdx.x / warpSize;                //获取每个block上，每个线程所在的warp
    int laneId = threadIdx.x % warpSize;                //每个block上，每个warp上，的线程索引0-31

    float val = (idx < N) ? exp(input[idx] - *max_val) : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (laneId == 0) s_mem[warpId] = val;
    __syncthreads();    //一个block上的所有warp的和存放在了block的共享内存中

    if (warpId == 0) {      //用每个block的第一个线程束再进行规约操作
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if(laneId == 0) atomicAdd(sum, val);
    }
}

__global__ void softmax(float* input, float* output, float* sum, float* max_val, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) output[idx] = exp(input[idx] - *max_val) / (*sum);
}

int main(){
    size_t N = 16;

    float* in_h = (float*)malloc(sizeof(float) * N);
    float* out_h = (float*)malloc(sizeof(float) * N);

    for(int i = 0;i < N;i++){
        in_h[i] = (float)(i);
    }
    
    memset(out_h, 0, sizeof(float) * N);

    float* max_d = nullptr;
    float* sum_d = nullptr;
    float* in_d  = nullptr;
    float* out_d = nullptr;

    cudacheck(cudaMalloc(&max_d, sizeof(float)));
    cudacheck(cudaMalloc(&sum_d, sizeof(float)));
    cudacheck(cudaMalloc(&in_d, sizeof(float) * N));
    cudacheck(cudaMalloc(&out_d, sizeof(float) * N));

    cudacheck(cudaMemcpy(in_d, in_h, sizeof(float) * N, cudaMemcpyHostToDevice));
    cudacheck(cudaMemcpy(out_d, out_h, sizeof(float) * N, cudaMemcpyHostToDevice));

    setToNegativeMax<<<1,1>>>(max_d);
    cudacheck(cudaMemset(sum_d, 0, sizeof(float)));

    int block_size = 128;
    int grid_size = CEIL(N, block_size);

    max_kernel<<<grid_size, block_size>>>(in_d, max_d, N);
    sum_kernel<<<grid_size, block_size>>>(in_d, sum_d, max_d, N);
    softmax<<<grid_size, block_size>>>(in_d, out_d, sum_d, max_d, N);
    
    cudacheck(cudaMemcpy(out_h, out_d, sizeof(float) * N, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    for (int i = 0;i < N; i++){
        printf("ans = %f\n", out_h[i]);
    }
    
    return 0;
}