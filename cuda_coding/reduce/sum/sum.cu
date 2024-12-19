#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

#define CEIL(a,b) ((a) + ((b) - 1)) / (b)
#define cudaCheck(err) _cuda_check(err, __FILE__, __LINE__)
#define FLOAT4(value) (*(float4*)(&(value)))

void _cuda_check(cudaError_t error, const char* file, int line){
    if(error != cudaSuccess){
        printf("erroe at file %s, line %d\n message:%s\n", file, line, cudaGetErrorString(error));
        exit(0);
    }
    return;
}

__global__ void sum_kernel(float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpID = threadIdx.x / warpSize;
    int laneID = threadIdx.x % warpSize;

    __shared__ float s_mem[32];

    float val = (idx < N) ? input[idx] : 0.0f;      //每个线程自己的寄存器
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if(laneID == 0) s_mem[warpID] = val;
    __syncthreads();


    if(warpID == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneID < warpNum) ? s_mem[laneID] : 0.0f;

        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (laneID == 0) atomicAdd(output, val);
    }
}

__global__ void device_reduce_v5(float* d_x, float* d_y, const int N) {
	__shared__ float s_y[32];
	int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;  // 这里要乘以4
	int warpId = threadIdx.x / warpSize;   // 当前线程位于第几个warp
	int laneId = threadIdx.x % warpSize;   // 当前线程是warp中的第几个线程
	float val = 0.0f;
	if (idx < N) {
		float4 tmp_x = FLOAT4(d_x[idx]);
		val += tmp_x.x;
		val += tmp_x.y;
		val += tmp_x.z;
		val += tmp_x.w;
	}
    printf("val-%d = %f\n",idx, val);
	#pragma unroll
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	}

	if (laneId == 0) s_y[warpId] = val;
	__syncthreads();

	if (warpId == 0) {
		int warpNum = blockDim.x / warpSize;
		val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
		for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
			val += __shfl_down_sync(0xFFFFFFFF, val, offset);
		}
		if (laneId == 0) atomicAdd(d_y, val);
	}
}

int main() {
    size_t N = 16;

    float *in_h = (float*)malloc(sizeof(float) * N);
    float *out_h = (float*)malloc(sizeof(float) * 1);

    for(int i = 0; i < N; i++){
        in_h[i] = i + 1;
    }
    memset(out_h, 0, sizeof(float) * 1);

    float* in_d = nullptr;
    float* out_d = nullptr;

    cudaCheck(cudaMalloc(&in_d, sizeof(float) * N));
    cudaCheck(cudaMalloc(&out_d, sizeof(float) * 1));

    cudaCheck(cudaMemcpy(in_d, in_h, sizeof(float) * N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(out_d, out_h, sizeof(float) * 1, cudaMemcpyHostToDevice));
    int block_size = 32;
    int grid_size = CEIL(CEIL(N, block_size), 4);
    device_reduce_v5<<<grid_size, block_size>>>(in_d, out_d, N);
    cudaCheck(cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return 0;
}