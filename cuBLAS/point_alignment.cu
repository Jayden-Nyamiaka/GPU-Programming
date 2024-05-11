/**
 * CUDA Point Alignment
 * George Stathopoulos, Jenny Lee, Mary Giambrone, 2019*/ 

#include <cstdio>
#include <stdio.h>
#include <fstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "helper_cuda.h"
#include <string>
#include <fstream>

#include "obj_structures.h"

// helper_cuda.h contains the error checking macros. note that they're called
// CUDA_CALL, CUBLAS_CALL, and CUSOLVER_CALL instead of the previous names

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // i is row, j is col


// custom helper to proint matrices (must be stored on CPU)
void printMatrix(std::string title, float *matrix, int rows, int cols) {
    std::cout << title << "\n";
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            std::cout << matrix[IDX2C(r, c, cols)] << " ";
        }
        std::cout << "\n";
    }
}


int main(int argc, char *argv[]) {

    if (argc != 4)
    {
        printf("Usage: ./point_alignment [file1.obj] [file2.obj] [output.obj]\n");
        return 1;
    }

    std::string filename, filename2, output_filename;
    filename = argv[1];
    filename2 = argv[2];
    output_filename = argv[3];

    std::cout << "Aligning " << filename << " with " << filename2 <<  std::endl;
    Object obj1 = read_obj_file(filename);
    std::cout << "Reading " << filename << ", which has " << obj1.vertices.size() << " vertices" << std::endl;
    Object obj2 = read_obj_file(filename2);

    std::cout << "Reading " << filename2 << ", which has " << obj2.vertices.size() << " vertices" << std::endl;
    if (obj1.vertices.size() != obj2.vertices.size())
    {
        printf("Error: number of vertices in the obj files do not match.\n");
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Loading in obj into vertex Array
    ///////////////////////////////////////////////////////////////////////////

    // Assume num of points match by this point (would have errored otherwise)
    int point_dim = 4; // 3 spatial + 1 homogeneous
    int num_points = obj1.vertices.size();

    // in col-major
    float * x1mat = vertex_array_from_obj(obj1);
    float * x2mat = vertex_array_from_obj(obj2);

    ///////////////////////////////////////////////////////////////////////////
    // Point Alignment
    ///////////////////////////////////////////////////////////////////////////

    // DONE: Initialize cublas handle
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    float *dev_x1mat;
    float *dev_x2mat;
    float *dev_xx4x4;
    float *dev_x1Tx2;

    // DONE: Allocate device memory and copy over the data onto the device
    // Hint: Use cublasSetMatrix() for copying
    CUDA_CALL(cudaMalloc((void **)&dev_x1mat, num_points*point_dim*sizeof(float)));
    CUBLAS_CALL(cublasSetMatrix(num_points, point_dim, sizeof(float), 
        x1mat, num_points, dev_x1mat, num_points));

    CUDA_CALL(cudaMalloc((void **)&dev_x2mat, num_points*point_dim*sizeof(float)));
    CUBLAS_CALL(cublasSetMatrix(num_points, point_dim, sizeof(float), 
        x2mat, num_points, dev_x2mat, num_points));

    // Now, proceed with the computations necessary to solve for the linear
    // transformation.

    float one = 1;
    float zero = 0;

    // Dimension Notes:
    // x1 and x2 have dim (R,C) = Nx4
    // xx4xx4 and x1Tx2 have dim (R,C) = 4xN * Nx4 = 4x4

    // DONE: First calculate xx4x4 and x1Tx2
    // Following two calls should correspond to:
    //   xx4x4 = Transpose[x1mat] . x1mat
    //   x1Tx2 = Transpose[x1mat] . x2mat
    CUDA_CALL(cudaMalloc((void **)&dev_xx4x4, point_dim*point_dim*sizeof(float)));
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        point_dim, point_dim, num_points, &one,
        dev_x1mat, num_points, dev_x1mat, num_points, &zero, 
        dev_xx4x4, point_dim));

    CUDA_CALL(cudaMalloc((void **)&dev_x1Tx2, point_dim*point_dim*sizeof(float)));
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        point_dim, point_dim, num_points, &one,
        dev_x1mat, num_points, dev_x2mat, num_points, &zero, 
        dev_x1Tx2, point_dim));

    // DONE: Finally, solve the system using LU-factorization! We're solving
    //         xx4x4 . m4x4mat.T = x1Tx2   i.e.   m4x4mat.T = Inverse[xx4x4] . x1Tx2
    //
    //       Factorize xx4x4 into an L and U matrix, ie.  xx4x4 = LU
    //
    //       Then, solve the following two systems at once using cusolver's getrs
    //           L . temp  =  P . x1Tx2
    //       And then then,
    //           U . m4x4mat = temp
    //
    //       Generally, pre-factoring a matrix is a very good strategy when
    //       it is needed for repeated solves.

    // DONE: Make handle for cuSolver
    cusolverDnHandle_t solver_handle;
    CUSOLVER_CALL(cusolverDnCreate(&solver_handle));

    // DONE: Initialize work buffer using cusolverDnSgetrf_bufferSize
    float * work;
    int Lwork;
    CUSOLVER_CALL(cusolverDnSgetrf_bufferSize(solver_handle, 
        point_dim, point_dim, dev_xx4x4, point_dim, &Lwork));

    // DONE: Compute buffer size and prepare memory
    CUDA_CALL(cudaMalloc((void **)&work, Lwork * sizeof(float)));

    // DONE: Initialize memory for pivot array, with a size of point_dim
    int * pivots;
    CUDA_CALL(cudaMalloc((void **)&pivots, point_dim * sizeof(int)));

    int *info;
    CUDA_CALL(cudaMalloc((void **)&info, sizeof(int)));

    // DONE: Now, call the factorizer cusolverDnSgetrf, using the above initialized data
    CUSOLVER_CALL(cusolverDnSgetrf(solver_handle, 
        point_dim, point_dim, dev_xx4x4, point_dim, work, pivots, info));

    // DONE: Finally, solve the factorized version using a direct call to cusolverDnSgetrs
    CUSOLVER_CALL(cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, 
        point_dim, point_dim, dev_xx4x4, point_dim, pivots, dev_x1Tx2, point_dim, info));
    
    // DONE: Destroy the cuSolver handle
    CUSOLVER_CALL(cusolverDnDestroy(solver_handle));

    // DONE: Copy final transformation back to host. Note that at this point
    // the transformation matrix is transposed
    float * out_transformation = (float *)malloc(point_dim*point_dim*sizeof(float));
    CUBLAS_CALL(cublasGetMatrix(point_dim, point_dim, sizeof(float), 
        dev_x1Tx2, point_dim, out_transformation, point_dim));

    // DONE: Don't forget to set the bottom row of the final transformation
    //       to [0,0,0,1] (right-most columns of the transposed matrix)
    out_transformation[IDX2C(0,3,point_dim)] = 0;
    out_transformation[IDX2C(1,3,point_dim)] = 0;
    out_transformation[IDX2C(2,3,point_dim)] = 0;
    out_transformation[IDX2C(3,3,point_dim)] = 1;

    // Print transformation in row order.
    printMatrix("Transformation Matrix", out_transformation, point_dim, point_dim);


    ///////////////////////////////////////////////////////////////////////////
    // Transform point and print output object file
    ///////////////////////////////////////////////////////////////////////////

    // DONE: Allocate and Initialize data matrix
    float * dev_pt;
    CUDA_CALL(cudaMalloc((void **)&dev_pt, num_points*point_dim*sizeof(float)));
    CUBLAS_CALL(cublasSetMatrix(num_points, point_dim, sizeof(float), 
        x1mat, num_points, dev_pt, num_points));

    // DONE: Allocate and Initialize transformation matrix
    float * dev_trans_mat;
    CUDA_CALL(cudaMalloc((void **)&dev_trans_mat, point_dim*point_dim*sizeof(float)));
    CUBLAS_CALL(cublasSetMatrix(point_dim, point_dim, sizeof(float), 
        out_transformation, point_dim, dev_trans_mat, point_dim));

    // DONE: Allocate and Initialize transformed points
    float * dev_trans_pt;
    CUDA_CALL(cudaMalloc((void **)&dev_trans_pt, num_points*point_dim*sizeof(float)));

    float one_d = 1;
    float zero_d = 0;

    // DONE: Transform point matrix
    //          (4x4 trans_mat) . (nx4 pointzx matrix)^T = (4xn transformed points)
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        point_dim, num_points, point_dim, &one_d,
        dev_trans_mat, point_dim, dev_pt, num_points, &zero_d, 
        dev_trans_pt, point_dim));

    // So now dev_trans_pt has shape (4 x n)
    // DONE: Copy final transformed vertices back over to the host
    float * trans_pt = (float *)malloc(num_points*point_dim*sizeof(float));
    CUBLAS_CALL(cublasGetMatrix(point_dim, num_points, sizeof(float), 
        dev_trans_pt, point_dim, trans_pt, point_dim));

    // get Object from transformed vertex matrix
    Object trans_obj = obj_from_vertex_array(trans_pt, num_points, point_dim, obj1);

    // print Object to output file
    std::ofstream obj_file (output_filename);
    print_obj_data(trans_obj, obj_file);

    // free CPU memory
    free(trans_pt);

    ///////////////////////////////////////////////////////////////////////////
    // Free Memory
    ///////////////////////////////////////////////////////////////////////////

    // DONE: Destory cublas handle
    CUBLAS_CALL(cublasDestroy(handle));

    // DONE: Free GPU memory
    CUDA_CALL(cudaFree(dev_x1mat));
    CUDA_CALL(cudaFree(dev_x2mat));
    CUDA_CALL(cudaFree(dev_xx4x4));
    CUDA_CALL(cudaFree(dev_x1Tx2));
    CUDA_CALL(cudaFree(work));
    CUDA_CALL(cudaFree(pivots));
    CUDA_CALL(cudaFree(info));
    CUDA_CALL(cudaFree(dev_trans_mat));
    CUDA_CALL(cudaFree(dev_pt));
    CUDA_CALL(cudaFree(dev_trans_pt));

    // DONE: Free CPU memory
    free(out_transformation);
    free(x1mat);
    free(x2mat);

}

