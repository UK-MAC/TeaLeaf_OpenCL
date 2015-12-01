#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_jacobi_solve_kernel_ocl_
(double * error)
{
    tea_context.tea_leaf_jacobi_solve_kernel(error);
}

/********************/

void TeaCLContext::tea_leaf_jacobi_solve_kernel
(double* error)
{
    tiles.at(fine_tile)->tea_leaf_jacobi_solve_kernel(error);
}
