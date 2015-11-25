#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

#include <cassert>

// CG solver functions
extern "C" void tea_leaf_cg_init_kernel_ocl_
(double * rro)
{
    chunk.tea_leaf_cg_init_kernel(rro);
}

extern "C" void tea_leaf_cg_calc_w_kernel_ocl_
(double * pw)
{
    chunk.tea_leaf_cg_calc_w_kernel(pw);
}
extern "C" void tea_leaf_cg_calc_ur_kernel_ocl_
(double * alpha, double * rrn)
{
    chunk.tea_leaf_cg_calc_ur_kernel(*alpha, rrn);
}
extern "C" void tea_leaf_cg_calc_p_kernel_ocl_
(double * beta)
{
    chunk.tea_leaf_cg_calc_p_kernel(*beta);
}

/********************/

void CloverChunk::tea_leaf_cg_init_kernel
(double * rro)
{
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
}

void CloverChunk::tea_leaf_cg_calc_w_kernel
(double* pw)
{
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

void CloverChunk::tea_leaf_cg_calc_ur_kernel
(double alpha, double* rrn)
{
    tea_leaf_cg_solve_calc_ur_device.setArg(1, alpha);

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);

    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void CloverChunk::tea_leaf_cg_calc_p_kernel
(double beta)
{
    tea_leaf_cg_solve_calc_p_device.setArg(1, beta);

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);
}

