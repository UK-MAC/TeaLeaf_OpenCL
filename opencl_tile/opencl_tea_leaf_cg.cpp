#include "../ctx_common.hpp"
#include "opencl_reduction.hpp"

void TeaOpenCLChunk::tea_leaf_cg_init_kernel
(double * rro)
{
    if (run_params.preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        ENQUEUE(tea_leaf_block_init_device);
        ENQUEUE(tea_leaf_block_solve_device);
    }
    else if (run_params.preconditioner_type == TL_PREC_JAC_DIAG)
    {
        ENQUEUE(tea_leaf_init_jac_diag_device);
    }

    ENQUEUE(tea_leaf_cg_solve_init_p_device);

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
}

void TeaOpenCLChunk::tea_leaf_cg_calc_w_kernel
(double* pw)
{
    ENQUEUE(tea_leaf_cg_solve_calc_w_device);

    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

void TeaOpenCLChunk::tea_leaf_cg_calc_ur_kernel
(double alpha, double* rrn)
{
    tea_leaf_cg_solve_calc_ur_device.setArg(1, alpha);

    ENQUEUE(tea_leaf_cg_solve_calc_ur_device);

    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void TeaOpenCLChunk::tea_leaf_cg_calc_p_kernel
(double beta)
{
    tea_leaf_cg_solve_calc_p_device.setArg(1, beta);

    ENQUEUE(tea_leaf_cg_solve_calc_p_device);
}

