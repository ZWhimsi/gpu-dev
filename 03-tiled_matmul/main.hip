#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <vector> 
#include <cmath>
#include <chrono>
#include "../error_handling.h"

// --- KERNEL  Tiled GEMM  ---
__global__ void tiledGEMM(float *A, float *B, float *C, int M, int N, int K) {
    
    #define TILE_WIDTH 16
    
    // Standardisation : x = col, y = row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float  tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float  tile_B[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float sum = 0.0f; // Registre rapide

    // Boucle sur les tuiles
    for (int p=0; p<(K+TILE_WIDTH-1)/TILE_WIDTH; p++){
        
        // Chargement A (Sécurisé)
        int rowA = row;
        int colA = p*TILE_WIDTH + tx;
        if(rowA < M && colA < K){
            tile_A[ty][tx] = A[rowA*K + colA];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        // Chargement B (Sécurisé & Row-Major)
        int rowB = p*TILE_WIDTH + ty;
        int colB = col;
        if(rowB < K && colB < N){
            tile_B[ty][tx] = B[rowB*N + colB];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
            
        __syncthreads();
            
        // Calcul (Produit matriciel sur la tuile)
        for (int k = 0; k < TILE_WIDTH; k++) { 
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        __syncthreads();
    }
        
    // Écriture finale
    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

// --- CPU IMPLEMENTATION (Baseline) ---
void cpuGEMM(float *A, float *B, float *C, int M, int N, int K) {
    // Note: Pour de très grosses matrices, on peut paralléliser ceci avec OpenMP (#pragma omp parallel for)
    // Mais ici on veut voir la performance "single core" brute vs GPU.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// --- HOST WRAPPER ---
void GEMHost(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    // 1. Allocation Device
    CHECK(hipMalloc(&d_A, sizeA)); 
    CHECK(hipMalloc(&d_B, sizeB));
    CHECK(hipMalloc(&d_C, sizeC));

    // 2. Copie H2D
    CHECK(hipMemcpy(d_A, h_A, sizeA, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, h_B, sizeB, hipMemcpyHostToDevice));

    // Configuration Grille
    dim3 dimBlock(16, 16); 
    dim3 dimGrid((N + 15) / 16, (M + 15) / 16);

    // Events pour le timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // ---------------------------------------------------------
    // BENCHMARK 1: CUSTOM TILED GEMM
    // ---------------------------------------------------------
    
    // Warmup (Pour éviter de mesurer l'overhead du premier lancement)
    tiledGEMM<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK(hipDeviceSynchronize());

    std::cout << "Running Custom Tiled GEMM..." << std::endl;
    hipEventRecord(start);
    tiledGEMM<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    double gflops_custom = (2.0 * M * N * K) / (milliseconds * 1e-3 * 1e9);
    std::cout << "Custom GPU Time: " << milliseconds / 1000.0 << " s | Performance: " << gflops_custom << " GFLOPS" << std::endl;
    
    // Vérification erreurs Kernel
    CHECK(hipGetLastError());

    // Validation Custom
    CHECK(hipMemcpy(h_C, d_C, sizeC, hipMemcpyDeviceToHost));
    std::cout << "Validation Custom [0][0] : " << h_C[0];
    if (std::abs(h_C[0] - (float)K) < 0.01f) std::cout << " -> SUCCESS" << std::endl;
    else std::cout << " -> ECHEC (Attendu: " << K << ")" << std::endl;

    // ---------------------------------------------------------
    // BENCHMARK 2: ROCBLAS (La référence)
    // ---------------------------------------------------------
    std::cout << "Running rocBLAS..." << std::endl;

    rocblas_handle handle;
    rocblas_create_handle(&handle);
    float alpha = 1.0f, beta = 0.0f;
    // --- WARMUP ROCBLAS ---
    // On fait un appel à vide pour charger les librairies
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    hipDeviceSynchronize(); // On attend que le warmup soit fini

    // MAINTENANT on peut mesurer
    std::cout << "Running rocBLAS..." << std::endl;
    hipEventRecord(start);
    // Reset de C pour être sûr
    hipMemset(d_C, 0, sizeC);

    hipEventRecord(start);
    // rocBLAS est column-major, mais avec des matrices de 1.0f et carrées, 
    // le résultat numérique est invariant par transposition.
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  N, M, K, // Attention : rocBLAS attend (m, n, k)
                  &alpha,
                  d_B, N,  // B
                  d_A, K,  // A
                  &beta,
                  d_C, N); // C
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&milliseconds, start, stop);
    double gflops_rocblas = (2.0 * M * N * K) / (milliseconds * 1e-3 * 1e9);
    std::cout << "rocBLAS Time:    " << milliseconds / 1000.0 << " s | Performance: " << gflops_rocblas << " GFLOPS" << std::endl;

    // Comparaison statistique
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Efficiency vs rocBLAS: " << (gflops_custom / gflops_rocblas) * 100.0 << "%" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Nettoyage
    rocblas_destroy_handle(handle);
    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));
    hipEventDestroy(start);
    hipEventDestroy(stop);
}

// --- MAIN ---
int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024; 

    std::cout << "Benchmark Matrix: " << M << "x" << N << "x" << K << std::endl;
    
    // Initialisation vecteurs (Host)
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_host(M * N, 0.0f); 

    // 1. Benchmark CPU (Baseline)
    std::cout << "Running CPU GEMM... (Patience !)" << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuGEMM(h_A.data(), h_B.data(), h_C_host.data(), M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    double gflops_cpu = (2.0 * M * N * K) / (diff_cpu.count() * 1e9);
    std::cout << "CPU Time:        " << diff_cpu.count() << " s | Performance: " << gflops_cpu << " GFLOPS" << std::endl;
    std::cout << "Check CPU [0]: " << h_C_host[0] << std::endl;
    // 2. Benchmark GPU (Custom + rocBLAS)
    GEMHost(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    return 0; 
}