#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_jacobi_solve_kernel_ocl_
(double * error)
{
    chunk.tea_leaf_jacobi_solve_kernel(error);
}

void CloverChunk::tea_leaf_jacobi_solve_kernel
(double* error)
{
    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}
