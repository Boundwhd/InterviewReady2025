#include "transpose.cuh"

/*朴素实现*/
__global__ void transpose_v0(const float* input, float* output, int M, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        output[col * M + row] = input[row * N + col];
    }
}
/*写入合并，读取不连续*/
__global__ void transpose_v1(const float* input, float* output, int M, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M){
        output[row * M + col] = input[col * N + row];
    }
}

int main(){
    size_t M = 2;
    size_t N = 4;

    dim3 block_size(32, 32);
    dim3 grid_size(CEIL(N, block_size.x), CEIL(M, block_size.y));


    float* A = (float*)malloc(sizeof(float) * M * N);
    float* C = (float*)malloc(sizeof(float) * M * N);

    for(int i = 0; i < M * N; i++){
        A[i] = 1.0f + i;
    }
    memset(C, 0, sizeof(float) * M * N);
    
    float* A_d = nullptr;
    float* C_d = nullptr;

    cudaMalloc(&A_d, sizeof(float) * M * N);
    cudaMalloc(&C_d, sizeof(float) * M * N);

    cudaMemcpy(A_d, A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    transpose_v0<<<grid_size, block_size>>>(A_d, C_d, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++){
            printf("%f\t", C[i * M + j]);
        }
        printf("\n");
    }
    return 0;
}