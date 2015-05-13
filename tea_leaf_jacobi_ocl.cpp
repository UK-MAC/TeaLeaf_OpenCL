#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_kernel_solve_ocl_
(const double * rx, const double * ry, double * error)
{
#if 0 // RTAG

    tea_context.tea_leaf_kernel_jacobi(*rx, *ry, error);

#endif //RTAG
}

void TeaCLContext::tea_leaf_kernel_jacobi
(double rx, double ry, double* error)
{
#if 0 // RTAG

    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);

#endif //RTAG
}

