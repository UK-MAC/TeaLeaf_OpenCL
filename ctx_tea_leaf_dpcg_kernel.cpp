#include "ctx_common.hpp"

extern "C" void tea_leaf_dpcg_coarsen_matrix_kernel_ocl_
(double * host_Kx, double * host_Ky)
{
    tea_context.tea_leaf_dpcg_coarsen_matrix_kernel(host_Kx, host_Ky);
}

extern "C" void tea_leaf_dpcg_copy_reduced_coarse_grid_ocl_
(double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di)
{
    tea_context.tea_leaf_dpcg_copy_reduced_coarse_grid(global_coarse_Kx,
        global_coarse_Ky, global_coarse_Di);
}

extern "C" void tea_leaf_dpcg_prolong_z_kernel_ocl_
(double * host_t2)
{
    tea_context.tea_leaf_dpcg_prolong_z_kernel(host_t2);
}

extern "C" void tea_leaf_dpcg_subtract_u_kernel_ocl_
(double * host_t2)
{
    tea_context.tea_leaf_dpcg_subtract_u_kernel(host_t2);
}

extern "C" void tea_leaf_dpcg_restrict_zt_kernel_ocl_
(double * host_ztr)
{
    tea_context.tea_leaf_dpcg_restrict_zt_kernel(host_ztr);
}

extern "C" void tea_leaf_dpcg_copy_reduced_t2_ocl_
(double * global_coarse_t2)
{
    tea_context.tea_leaf_dpcg_copy_reduced_t2(global_coarse_t2);
}

extern "C" void tea_leaf_dpcg_matmul_zta_kernel_ocl_
(double * host_ztaz)
{
    tea_context.tea_leaf_dpcg_matmul_zta_kernel(host_ztaz);
}

extern "C" void tea_leaf_dpcg_init_p_kernel_ocl_
(void)
{
    tea_context.tea_leaf_dpcg_init_p_kernel();
}

extern "C" void tea_leaf_dpcg_store_r_kernel_ocl_
(void)
{
    tea_context.tea_leaf_dpcg_store_r_kernel();
}

extern "C" void tea_leaf_dpcg_calc_rrn_kernel_ocl_
(double * rrn)
{
    tea_context.tea_leaf_dpcg_calc_rrn_kernel(rrn);
}

extern "C" void tea_leaf_dpcg_calc_p_kernel_ocl_
(void)
{
    tea_context.tea_leaf_dpcg_calc_p_kernel();
}

extern "C" void tea_leaf_dpcg_coarse_solve_ocl_
(double * coarse_solve_eps,
 int    * coarse_solve_max_iters,
 int    * it_count,
 double * theta,
 int    * inner_use_ppcg,
 double * inner_cg_alphas,
 double * inner_cg_betas,
 double * inner_ch_alphas,
 double * inner_ch_betas,
 double * t2_result)
{
    tea_context.tea_leaf_dpcg_local_solve(coarse_solve_eps,
        coarse_solve_max_iters, it_count, theta, inner_use_ppcg,
        inner_cg_alphas, inner_cg_betas, inner_ch_alphas,
        inner_ch_betas, t2_result);
}

/********************/

void TeaCLContext::tea_leaf_dpcg_coarsen_matrix_kernel
(double * host_Kx, double * host_Ky)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_coarsen_matrix_kernel(host_Kx, host_Ky);
}

void TeaCLContext::tea_leaf_dpcg_copy_reduced_coarse_grid
(double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di)
{
    // COARSE tile
    chunks.at(coarse_chunk)->tea_leaf_dpcg_copy_reduced_coarse_grid(global_coarse_Kx, global_coarse_Ky, global_coarse_Di);
}

void TeaCLContext::tea_leaf_dpcg_copy_reduced_t2
(double * global_coarse_t2)
{
    // COARSE tile
    chunks.at(coarse_chunk)->tea_leaf_dpcg_copy_reduced_t2(global_coarse_t2);
}

void TeaCLContext::tea_leaf_dpcg_prolong_z_kernel
(double * host_t2)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_prolong_z_kernel(host_t2);
}

void TeaCLContext::tea_leaf_dpcg_subtract_u_kernel
(double * host_t2)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_subtract_u_kernel(host_t2);
}

void TeaCLContext::tea_leaf_dpcg_restrict_zt_kernel
(double * host_ztr)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_restrict_zt_kernel(host_ztr);
}

void TeaCLContext::tea_leaf_dpcg_matmul_zta_kernel
(double * host_ztaz)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_matmul_zta_kernel(host_ztaz);
}

void TeaCLContext::tea_leaf_dpcg_init_p_kernel
(void)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_init_p_kernel();
}

void TeaCLContext::tea_leaf_dpcg_store_r_kernel
(void)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_store_r_kernel();
}

void TeaCLContext::tea_leaf_dpcg_calc_rrn_kernel
(double * rrn)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_calc_rrn_kernel(rrn);
}

void TeaCLContext::tea_leaf_dpcg_calc_p_kernel
(void)
{
    chunks.at(fine_chunk)->tea_leaf_dpcg_calc_p_kernel();
}

void TeaCLContext::tea_leaf_dpcg_local_solve
(double * coarse_solve_eps,
 int    * coarse_solve_max_iters,
 int    * it_count,
 double * theta,
 int    * inner_use_ppcg,
 double * inner_cg_alphas,
 double * inner_cg_betas,
 double * inner_ch_alphas,
 double * inner_ch_betas,
 double * t2_result)
{
    chunks.at(coarse_chunk)->tea_leaf_dpcg_local_solve(coarse_solve_eps,
        coarse_solve_max_iters, it_count, theta, inner_use_ppcg,
        inner_cg_alphas, inner_cg_betas, inner_ch_alphas,
        inner_ch_betas, t2_result);
}

