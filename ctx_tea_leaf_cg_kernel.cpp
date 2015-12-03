#include "ctx_common.hpp"

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

    chunks.at(fine_chunk)->tea_leaf_cg_init_kernel(rro);
}

void TeaCLContext::tea_leaf_cg_calc_w_kernel
(double* pw)
{
    chunks.at(fine_chunk)->tea_leaf_cg_calc_w_kernel(pw);
}

void TeaCLContext::tea_leaf_cg_calc_ur_kernel
(double alpha, double* rrn)
{
    chunks.at(fine_chunk)->tea_leaf_cg_calc_ur_kernel(alpha, rrn);
}

void TeaCLContext::tea_leaf_cg_calc_p_kernel
(double beta)
{
    chunks.at(fine_chunk)->tea_leaf_cg_calc_p_kernel(beta);
}

