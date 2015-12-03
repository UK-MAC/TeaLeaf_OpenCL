#include "ctx_common.hpp"

extern "C" void tea_leaf_jacobi_solve_kernel_ocl_
(double * error)
{
    tea_context.tea_leaf_jacobi_solve_kernel(error);
}

/********************/

void TeaCLContext::tea_leaf_jacobi_solve_kernel
(double* error)
{
    chunks.at(fine_chunk)->tea_leaf_jacobi_solve_kernel(error);
}
