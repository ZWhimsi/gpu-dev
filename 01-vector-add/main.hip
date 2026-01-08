#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include "../error_handling.h"



//Create Kernel
__global__ 
void vecAddKernel(float *a, float *b, float *c, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){
        c[i] = a[i] + b[i];
    }
}


//Host Code

void vecAdd(float* a_h, float* b_h, float* c_h, int n){
    size_t size = n*sizeof(float);
    float *a_d, *b_d, *c_d;

    CHECK(hipMalloc((void **) &a_d,size));
    CHECK(hipMalloc((void **) &b_d,size));
    CHECK(hipMalloc((void **) &c_d,size));

    CHECK(hipMemcpyHtoD(a_d, a_h, size));
    CHECK(hipMemcpyHtoD(b_d, b_h, size));

    vecAddKernel<<<dim3(int(ceil(n/256.0))),dim3(256)>>>(a_d,b_d,c_d,n);
    CHECK(hipGetLastError()); // We need to do this to check the kernel correctly runned cause we cant add the CHECk on the kernel call
    CHECK(hipDeviceSynchronize()); // Same here

    CHECK(hipMemcpyDtoH(c_h,c_d,size));
    CHECK(hipFree(a_d));
    CHECK(hipFree(b_d));
    CHECK(hipFree(c_d));
}


//Main function

int main(){
    int n = 1000000;
    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> h_C(n);
    
    std::generate(h_A.begin(),h_A.end(),[](){return static_cast<float>(rand())/RAND_MAX;});
    std::generate(h_B.begin(),h_B.end(),[](){return static_cast<float>(rand())/RAND_MAX;});


    // Verification of device
    int count;
    hipError_t err = hipGetDeviceCount(&count);
    if(err != hipSuccess || count == 0){
        std::cout << "No device found or error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    int deviceId;
    err = hipGetDevice(&deviceId);
    if(err != hipSuccess){
        std::cout << "Error getting device: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, deviceId);
    if(err == hipSuccess){
        std::cout << "Device count: " << count << std::endl;
        std::cout << "Current device ID: " << deviceId << std::endl;
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    }
    vecAdd(h_A.data(),h_B.data(),h_C.data(),n);
    std::cout << '\n'<< "Verification : " << h_A[n-1] << " + " << h_B[n-1] << " = " << h_C[n-1] << std::endl;
    return 0;

}