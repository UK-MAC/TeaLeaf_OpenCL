#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_jacobi_solve_kernel_ocl_
(double * error)
{
    tea_context.tea_leaf_jacobi_solve_kernel(error);
}

void TeaCLContext::tea_leaf_jacobi_solve_kernel
(double* error)
{
    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_jacobi_copy_u_device);
        ENQUEUE(tea_leaf_jacobi_solve_device);

        *error = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_1);
    }
}
