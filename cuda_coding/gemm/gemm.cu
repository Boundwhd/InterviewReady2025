#include "gemm.cuh"
/*
每个线程分配一个像素去计算一个像素，一个线程需要负责A的一行，B的一列，然后累加乘
最基础Naive实现
1. **访存比低**：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2；
2. **访存延迟高**：访问**全局内存**，**延迟高**，需要几百个时钟周期 (cycle)
3. **较低的访存比无法有效隐藏访存延迟**
4. 访存量：矩阵C的每个元素计算需要访问2K个单精度浮点数，完成全部计算需要 $2\times K\times M\times N$
5. 相同位置元素被重复读取（C中同一行元素计算共享A中同一行元素，C中同一列元素计算共享B中同一列元素）
*/
__global__ void gemm_v1(float* A, float*B, float*C, int M, int N, int K){
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y; 

    if(id_x < N && id_y < M){
        float tep = 0;
        for(int i = 0; i < K; i++) {
            tep += A[id_y * K + i] * B[i * N + id_x];
        }
        C[id_y * N + id_x] = tep;
    }
}


template<const int BLOCK_SIZE>
__global__ void gemm_v2(float* A, float*B, float*C, int M, int N, int K){
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    //blockID and threadId
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;
}

int main(){
    size_t M = 8;
    size_t K = 128;
    size_t N = 16;

    dim3 block_size(32, 32);
    dim3 grid_size(CEIL(N, block_size.x), CEIL(M, block_size.y));


    float* A = (float*)malloc(sizeof(float) * M * K);
    float* B = (float*)malloc(sizeof(float) * N * K);
    float* C = (float*)malloc(sizeof(float) * M * N);

    for(int i = 0; i < M * K; i++){
        A[i] = 1.0f;
    }
    for(int i = 0; i < N * K; i++){
        B[i] = 2.0f;
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

    gemm_v1<<<grid_size, block_size>>>(A_d, B_d, C_d, M, N, K);
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