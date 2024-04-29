#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * For all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */

 /* Structural Notes for Self: 
  * Each block handles a 64x64 block of the input matrix and a single block
  * has a 64x16 block of threads where each thread handles a 1x4 section of
  * the input matrix such that each warp in the block has 32 threads covering
  * a 32x4 section of the input matrix. For a single block, x iterates from 
  * 0 to 63 and y iterates from 0 to 15. Then, within a single thread, j 
  * iterates over 4 elements in one column (4 elems per y) such that 
  * input[i + n * j] accesses 64*16*4 adjacent addresses over a 
  * 16384-byte long memory block. In a single warp, x iterates over 32
  * different values (either 0-31 or 32-63) and y stays constant.
  */

__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // Comment on suboptimal accesses
    
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
    // Bank conflicts when writing to output: n*i doesn't affect the bank bc
    // n%32==0 and j is the same for all threads in a warp, so all threads in
    // a warp access addresses in output that are in the same 4 adjacent banks
    // Additionally, the writes for output[j + n * i] is uncoalesced since the
    // writes are all n floats away from each other for each adjacent thread.
    // This requires 32 GPU cache lines, which is very inefficient.
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    __shared__ float data[4160]; // 65 columns * 64 rows = 4160 elements
    // Padding avoids the bank conflicts when writing to output bc 65%32!=0.
    // Using global memory with 0 bank conflicts makes the code much faster.

    // i and j for accessing global memory
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // i and j for accessing shmem
    const int smi = threadIdx.x;
    int smj = 4 * threadIdx.y;

    // Load data from global to shared memory (shmem), then syncthreads
    for (; j < end_j; j++, smj++)
        data[smi + 65 * smj] = input[i + n * j]; 
    __syncthreads();
    // By this point, all data in the 64x64 block has been loaded into data

    // Reset shmem indices
    smj = 4 * threadIdx.y;
    const int end_smj = smj + 4;

    // i and j for writing back to global memory
    int ri = 64 * blockIdx.x + 4 * threadIdx.y;
    const int rj = 64 * blockIdx.y + threadIdx.x;

    // Store data from shmem to global memory
    for (; smj < end_smj; smj++, ri++)
        output[rj + n * ri] = data[smj + 65 * smi];
    // This makes the global memory access coalesced because each rj will be
    // dependent on threadIdx.x such that adjacent threads access adjacent
    // addresses in output. This will increase memory access efficiency.

    // This can still be made better by loop unrolling and ILP.
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.

    __shared__ float data[4160]; // 65 columns * 64 rows = 4160 elements
    // padding avoids the bank conflicts when writing to output bc 65%32!=0

    // i and j for accessing global memory
    const int i = threadIdx.x + 64 * blockIdx.x;
    const int j = 4 * threadIdx.y + 64 * blockIdx.y;

    // i and j for accessing shmem
    const int smi = threadIdx.x;
    const int smj = 4 * threadIdx.y;

    // Load data from global to shared memory with loop unrolled
    data[smi + 65 * smj] = input[i + n * j]; 
    data[smi + 65 * (smj+1)] = input[i + n * (j+1)]; 
    data[smi + 65 * (smj+2)] = input[i + n * (j+2)]; 
    data[smi + 65 * (smj+3)] = input[i + n * (j+3)]; 

    __syncthreads();
    // By this point, all data in the 64x64 block has been loaded into data

    // i and j for writing back to global memory
    const int ri = 64 * blockIdx.x + 4 * threadIdx.y;
    const int rj = 64 * blockIdx.y + threadIdx.x;

    // Store data from shmem to global memory with loop unrolled
    output[rj + n * ri] = data[smj + 65 * smi];
    output[rj + n * (ri+1)] = data[(smj+1) + 65 * smi];
    output[rj + n * (ri+2)] = data[(smj+2) + 65 * smi];
    output[rj + n * (ri+3)] = data[(smj+3) + 65 * smi];

    // Utilizing loop unrolling and ILP gets rid of the overhead computation
    // necessary when running loops and furthers the parallelism of the code.
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
