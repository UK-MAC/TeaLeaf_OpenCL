#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

#include <cassert>

// CG solver functions
extern "C" void tea_leaf_cg_init_kernel_ocl_
(double * rro)
{
    tea_context.tea_leaf_cg_init_kernel(rro);
}

extern "C" void tea_leaf_cg_calc_w_kernel_ocl_
(double * pw)
{
    tea_context.tea_leaf_cg_calc_w_kernel(pw);
}
extern "C" void tea_leaf_cg_calc_ur_kernel_ocl_
(double * alpha, double * rrn)
{
    tea_context.tea_leaf_cg_calc_ur_kernel(*alpha, rrn);
}
extern "C" void tea_leaf_cg_calc_p_kernel_ocl_
(double * beta)
{
    tea_context.tea_leaf_cg_calc_p_kernel(*beta);
}

/********************/

void TeaCLContext::tea_leaf_cg_init_kernel
(double * rro)
{
    assert(run_params.tea_solver == TEA_ENUM_CG || run_params.tea_solver == TEA_ENUM_CHEBYSHEV || run_params.tea_solver == TEA_ENUM_PPCG);

    // Assume calc_residual has been called before this (to calculate initial_residual)

    if (run_params.preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        FOR_EACH_TILE
        {
            ENQUEUE(tea_leaf_block_init_device);
            ENQUEUE(tea_leaf_block_solve_device);
        }
    }
    else if (run_params.preconditioner_type == TL_PREC_JAC_DIAG)
    {
        FOR_EACH_TILE
        {
            ENQUEUE(tea_leaf_init_jac_diag_device);
        }
    }

    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_cg_solve_init_p_device);

        *rro = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_2);
    }
}

void TeaCLContext::tea_leaf_cg_calc_w_kernel
(double* pw)
{
    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_cg_solve_calc_w_device);

        *pw = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_3);
    }
}

void TeaCLContext::tea_leaf_cg_calc_ur_kernel
(double alpha, double* rrn)
{
    FOR_EACH_TILE
    {
        tile->tea_leaf_cg_solve_calc_ur_device.setArg(1, alpha);

        ENQUEUE(tea_leaf_cg_solve_calc_ur_device);

        *rrn = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_5);
    }
}

void TeaCLContext::tea_leaf_cg_calc_p_kernel
(double beta)
{
    FOR_EACH_TILE
    {
        tile->tea_leaf_cg_solve_calc_p_device.setArg(1, beta);

        ENQUEUE(tea_leaf_cg_solve_calc_p_device);
    }
}

