#include <hip/hip_runtime.h>
#include <iostream>
#include <vector> 
#include "../error_handling.h"

// --- KERNEL ---
__global__ void naiveGEMKernel(float *A, float *B, float *C, int N) {
    // Standardisation : x = col, y = row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f; // Registre rapide
        
        for (int k = 0; k < N; k++) { // k doit être initialisé à 0
            // A[row][k] * B[k][col]
            sum += A[row * N + k] * B[k * N + col];
        }
        
        // Écriture UNIQUE en VRAM à la fin
        C[row * N + col] = sum;
    }
}

// --- HOST ---
void GEMHost(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocation
    CHECK(hipMalloc(&d_A, size)); // Pas besoin de cast (void**) en C++ moderne, mais accepté
    CHECK(hipMalloc(&d_B, size));
    CHECK(hipMalloc(&d_C, size));

    // Copie Aller
    CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    // Définition de la Grille
    dim3 threadsPerBlock(16, 16); // Constructeur correct
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Lancement
    naiveGEMKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    // Copie Retour
    CHECK(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));

    // Validation (Pour N=1024, somme de 1.0 = 1024.0)
    std::cout << "Resultat case [0][0] : " << h_C[0] << std::endl;
    if (std::abs(h_C[0] - (float)N) < 0.01f) { // Comparaison float propre
        std::cout << "SUCCESS ! Logique validée." << std::endl;
    } else {
        std::cout << "ECHEC. Attendu : " << N << ", Reçu : " << h_C[0] << std::endl;
    }

    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));
}

// --- MAIN ---
int main() {
    int N = 1024; // Point-virgule manquant corrigé

    // Initialisation vecteurs
    std::vector<float> h_A(N * N, 1.0f); // Remplis directement à 1.0f
    std::vector<float> h_B(N * N, 1.0f);
    std::vector<float> h_C(N * N);

    // Appel
    GEMHost(h_A.data(), h_B.data(), h_C.data(), N);

    return 0; // Point-virgule corrigé
}