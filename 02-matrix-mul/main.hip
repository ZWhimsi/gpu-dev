#include <hip/hip_runtime.h>
#include <iostream>
#include <vector> 
#include "../error_handling.h"

// --- KERNEL ---
__global__ void naiveGEMKernel(float *A, float *B, float *C, int M, int N, int K) {
    
    // Standardisation : x = col, y = row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f; // Registre rapide
        
        for (int k = 0; k < K; k++) { // k doit être initialisé à 0
            // A[row][k] * B[k][col]
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Écriture UNIQUE en VRAM à la fin
        C[row * N + col] = sum;
    }
}

// --- HOST ---
void GEMHost(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocation
    CHECK(hipMalloc(&d_A, sizeA)); // Pas besoin de cast (void**) en C++ moderne, mais accepté
    CHECK(hipMalloc(&d_B, sizeB));
    CHECK(hipMalloc(&d_C, sizeC));

    // Copie Aller
    CHECK(hipMemcpy(d_A, h_A, sizeA, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, h_B, sizeB, hipMemcpyHostToDevice));

    // Définition de la Grille
    dim3 threadsPerBlock(16, 16); // Constructeur correct

    // Rappel y = row; x = col !!
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    // Lancement
    naiveGEMKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    // Copie Retour
    CHECK(hipMemcpy(h_C, d_C, sizeC, hipMemcpyDeviceToHost));

    // Validation (Pour N=1024, somme de 1.0 = 1024.0)
    std::cout << "Resultat case [0][0] : " << h_C[0] << std::endl;
    if (std::abs(h_C[0] - (float)K) < 0.01f) { // Comparaison float propre
        std::cout << "SUCCESS ! Logique validée." << std::endl;
    } else {
        std::cout << "ECHEC. Attendu : " << K << ", Reçu : " << h_C[0] << std::endl;
    }

    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));
}

// --- MAIN ---
int main() {
    int M = 256;
    int N = 512;
    int K = 1024; // Point-virgule manquant corrigé

    // Initialisation vecteurs
    std::vector<float> h_A(M * K, 1.0f); // Remplis directement à 1.0f
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N);

    // Appel
    GEMHost(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    return 0; // Point-virgule corrigé
}