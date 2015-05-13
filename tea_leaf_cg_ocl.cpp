#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

// CG solver functions
extern "C" void tea_leaf_kernel_init_cg_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, double * rro)
{
    tea_context.tea_leaf_init_cg(*coefficient, *dt, rx, ry, rro);
}

extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_w_
(const double * rx, const double * ry, double * pw)
{
    tea_context.tea_leaf_kernel_cg_calc_w(*rx, *ry, pw);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_ur_
(double * alpha, double * rrn)
{
    tea_context.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_p_
(double * beta)
{
    tea_context.tea_leaf_kernel_cg_calc_p(*beta);
}

void TeaCLContext::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
    FOR_EACH_TILE
    {
        // Assume calc_residual has been called before this (to calculate initial_residual)

        if (preconditioner_type == TL_PREC_JAC_BLOCK)
        {
            ENQUEUE(tea_leaf_block_init_device);
            ENQUEUE(tea_leaf_block_solve_device);
        }
        else if (preconditioner_type == TL_PREC_JAC_DIAG)
        {
            ENQUEUE(tea_leaf_init_jac_diag_device);
        }

        ENQUEUE(tea_leaf_cg_solve_init_p_device);
    }

    *rro = sumReduceValues<double>(std::vector<int>(1, 2)).at(0);
}

void TeaCLContext::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_cg_solve_calc_w_device);
    }

    //*pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
    *pw = sumReduceValues<double>(std::vector<int>(1, 3)).at(0);
}

void TeaCLContext::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    FOR_EACH_TILE
    {
        tile->tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

        ENQUEUE(tea_leaf_cg_solve_calc_ur_device);
    }

    //*rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
    *rrn = sumReduceValues<double>(std::vector<int>(1, 5)).at(0);
}

void TeaCLContext::tea_leaf_kernel_cg_calc_p
(double beta)
{
    FOR_EACH_TILE
    {
        tile->tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

        ENQUEUE(tea_leaf_cg_solve_calc_p_device);
    }
}


