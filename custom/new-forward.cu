#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_WIDTH 16
// #define CONV_DEBUG

__global__ void conv_forward_kernel(
    float *y, const float *x, const float *k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int W_num = ceil(W_out / (BLOCK_WIDTH * 1.0));
        // H_num = ceil(H_out / (BLOCK_WIDTH * 1.0));
    int b = blockIdx.x, m = blockIdx.y;
    int h = (blockIdx.z / W_num) * BLOCK_WIDTH + threadIdx.y,
        w = (blockIdx.z % W_num) * BLOCK_WIDTH + threadIdx.x;
    // int h = threadIdx.y, w = threadIdx.x;

    int c, p, q;
    float res = 0.0f;
    if (w >= W_out || h >= H_out) return;
    for (c = 0; c < C; ++c) {
        for (p = 0; p < K; ++p) {
            for (q = 0; q < K; ++q) {
                res += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
            }
        }
    }
    y4d(b, m, h, w) = res;

#undef y4d
#undef x4d
#undef k4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_y, const float *host_x, const float *host_k, 
    float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int y_size = B*M*H_out*W_out*sizeof(float), 
        x_size = B*C*H*W*sizeof(float), 
        k_size = M*C*K*K*sizeof(float);
    cudaMalloc((void**) device_y_ptr, y_size);
    cudaMalloc((void**) device_x_ptr, x_size);
    cudaMalloc((void**) device_k_ptr, k_size);

    cudaMemcpy(*device_y_ptr, host_y, y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_x_ptr, host_x, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, k_size, cudaMemcpyHostToDevice);
    
#ifdef CONV_DEBUG
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error@conv_forward_gpu_prolog: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
#endif
}


__host__ void GPUInterface::conv_forward_gpu(
    float *device_y, const float *device_x, const float *device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#ifdef CONV_DEBUG
    std::cout << "Output image: " << H_out << " x " << W_out << std::endl;
    // print dimension information
    std::cout << "Grid Dimension: " << B << " x " << M << " x " << W_num * H_num << std::endl;
    std::cout << "Block Dimension: " << BLOCK_WIDTH << " x " << BLOCK_WIDTH << " x " << 1 << std::endl;
#endif

    int W_num = ceil(W_out / (BLOCK_WIDTH * 1.0)),
        H_num = ceil(H_out / (BLOCK_WIDTH * 1.0));

    dim3 dimGrid(B, M, W_num * H_num);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    cudaDeviceSynchronize();

#ifdef CONV_DEBUG
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error@conv_forward_gpu: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
#endif
}


__host__ void GPUInterface::conv_forward_gpu_epilog(
    float *host_y, float *device_y, float *device_x, float *device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int y_size = B*M*H_out*W_out*sizeof(float), 
        x_size = B*C*H*W*sizeof(float), 
        k_size = M*C*K*K*sizeof(float);
    cudaMemcpy(host_y, device_y, y_size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
