#include "../ctx_common.hpp"
#include "opencl_reduction.hpp"

void TeaOpenCLChunk::tea_leaf_jacobi_solve_kernel
(double* error)
{
    ENQUEUE(tea_leaf_jacobi_copy_u_device);
    ENQUEUE(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

