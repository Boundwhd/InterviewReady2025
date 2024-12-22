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
__global__ void sgemm_v2(int M, int N, int K, float *A, float *B, float *C) {
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    // blockId and threadId
    int bx = blockIdx.x;
    int by = blockIdx.y;    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // 申请共享内存空间
    // NVIDIA GeForce GTX 1050's sharedMemPerBlock is 48KB = 48*1024B = 49152B(0xc000)
    // 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <= 48*1024/4 = 12288
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {  // 窗口滑动
        // 缓存A_tile和B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();  // 同步同一个线程块(block)中的线程，执行到同一个点
        // 移动A,B指针到下一个矩阵块
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = tmp;
}

int main(){
    size_t M = 2;
    size_t K = 4;
    size_t N = 4;

    dim3 block_size(4);
    dim3 grid_size(CEIL(N, 2), CEIL(M, 2));


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

    sgemm_v2<2><<<grid_size, block_size>>>(M, N, K, A_d, B_d, C_d);
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