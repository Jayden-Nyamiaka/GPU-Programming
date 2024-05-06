/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data, int padded_length) {
    /* DONE: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal 
    (i.e. divide by padded_length) (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.
    */

    // Compute the current thread index
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute point-wise multiplication with padded length scaling
    while (tid < padded_length) {
        float real = 
            raw_data[tid].x * impulse_v[tid].x - raw_data[tid].y * impulse_v[tid].y;
        float imag = 
            raw_data[tid].x * impulse_v[tid].y + raw_data[tid].y * impulse_v[tid].x;
        out_data[tid].x = real / padded_length;
        out_data[tid].y = imag / padded_length;
        // Move to next set of blocks
        tid += blockDim.x * gridDim.x;
    }
    
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    /* DONE 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)
    */
    // Utilizes dynamic shmem to increase speed and GPU utilization 
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x*2 + threadIdx.x;

    // First step of reduction
    // Each thread reads (and compares) 2 values from global to shmem
    // Since both signals are real, the resulting value must be real s.t.
    // we ignore imaginary part and extract real (done here via out_data[#].x)
    while (i < padded_length) {
        s_data[tid] = fabsf(out_data[i].x);
        if (i + blockDim.x < padded_length) 
            s_data[tid] = fmaxf(s_data[tid], fabsf(out_data[i + blockDim.x].x));
        __syncthreads();

        // Implements binary tree parallel reduction using sequential addressing
        for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
            if (tid < s) {
                s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
            }
            __syncthreads();
        }

        // Unrolls the last few computations of the loop
        if(tid < 32) 
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + 32]); __syncthreads();
        if(tid < 16) 
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + 16]); __syncthreads();
        if(tid <  8) 
            s_data[tid] = fmaxf(s_data[tid], s_data[tid +  8]); __syncthreads();
        if(tid <  4) 
            s_data[tid] = fmaxf(s_data[tid], s_data[tid +  4]);
        if(tid <  2) 
            s_data[tid] = fmaxf(s_data[tid], s_data[tid +  2]);
        if(tid <  1) 
            s_data[tid] = fmaxf(s_data[tid], s_data[tid +  1]);

        // Accumulates final maximum result with result already in max_abs_val
        // Done atomically to ensure no interference/overwrites
        if (tid == 0) atomicMax(max_abs_val, s_data[0]);
        
        // Move to next set of blocks
        i += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    /* DONE 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 
    This kernel should be quite short.
    */

    // Compute the current thread index
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < padded_length) {
        // Divide real part in place by max_abs_val (assuming imag part =0)
        out_data[tid].x /= *max_abs_val;
        // Move to next set of blocks
        tid += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* DONE: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, 
        impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* DONE 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock, 
                        (threadsPerBlock * sizeof(float))>>>(
        out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* DONE 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, 
        max_abs_val, padded_length);
}
