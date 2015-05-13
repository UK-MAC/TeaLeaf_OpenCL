#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

// CG solver functions
extern "C" void tea_leaf_kernel_init_cg_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, double * rro)
{
#if 0 // RTAG

    tea_context.tea_leaf_init_cg(*coefficient, *dt, rx, ry, rro);

#endif //RTAG
}

extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_w_
(const double * rx, const double * ry, double * pw)
{
#if 0 // RTAG

    tea_context.tea_leaf_kernel_cg_calc_w(*rx, *ry, pw);

#endif //RTAG
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_ur_
(double * alpha, double * rrn)
{
#if 0 // RTAG

    tea_context.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);

#endif //RTAG
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_p_
(double * beta)
{
#if 0 // RTAG

    tea_context.tea_leaf_kernel_cg_calc_p(*beta);

#endif //RTAG
}

void TeaCLContext::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
#if 0 // RTAG
    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV || tea_solver == TEA_ENUM_PPCG);

    // Assume calc_residual has been called before this (to calculate initial_residual)

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        ENQUEUE_OFFSET(tea_leaf_block_init_device);
        ENQUEUE_OFFSET(tea_leaf_block_solve_device);
    }
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
    {
        ENQUEUE_OFFSET(tea_leaf_init_jac_diag_device);
    }

    ENQUEUE_OFFSET(tea_leaf_cg_solve_init_p_device);

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
#endif // RTAG
}

void TeaCLContext::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
#if 0 // RTAG

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);

#endif //RTAG
}

void TeaCLContext::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
#if 0 // RTAG

    tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);

    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);

#endif //RTAG
}

void TeaCLContext::tea_leaf_kernel_cg_calc_p
(double beta)
{
#if 0 // RTAG

    tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);

#endif //RTAG
}


