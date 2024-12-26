#include "gemm.cuh"

template<const int BLOCKSIZE>
__global__ void gemm_my(float* A, float* B, float* C, const int M, const int K, const int N) {
    const int BM = BLOCKSIZE;
    const int BN = BLOCKSIZE;
    const int BK = BLOCKSIZE;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    __shared__ float mem_A[BM * BK];
    __shared__ float mem_B[BK * BN];

    A = &A[by * K * BM];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int j = 0; j < K; j += BK){
        
        mem_A[ty * BK + tx] = A[ty * K + tx];
        mem_B[ty * BN + tx] = B[ty * N + tx];

        __syncthreads();
        A += BK;
        B += N * BK;

        for(int i = 0; i < BK; i++) {
            tmp += mem_A[ty * BK + i] * mem_B[i * BN + tx];
        }

        __syncthreads();
    }

    C[ty * N + tx] = tmp;
    
}

int main(){
    size_t M = 4;
    size_t K = 4;
    size_t N = 4;

    dim3 block_size(4);
    dim3 grid_size(CEIL(N, 2), CEIL(M, 2));


    float* A = (float*)malloc(sizeof(float) * M * K);
    float* B = (float*)malloc(sizeof(float) * N * K);
    float* C = (float*)malloc(sizeof(float) * M * N);

    for(int i = 0; i < M * K; i++){
        A[i] = 3.0f;
    }
    for(int i = 0; i < N * K; i++){
        B[i] = 3.0f;
    }
    memset(C, 0, sizeof(float) * M * N);
    
    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;

    cudaMalloc(&A_d, sizeof(float) * M * K);
    cudaMalloc(&B_d, sizeof(float) * N * K);
    cudaMalloc(&C_d, sizeof(float) * M * N);

    cudaMemcpy(A_d, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(float) * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    gemm_my<2><<<grid_size, block_size>>>(A_d, B_d, C_d, M, K, N);
    cudaMemcpy(C, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++){
            printf("%f\t", C[i * N + j]);
        }
        printf("\n");
    }
    return 0;
}