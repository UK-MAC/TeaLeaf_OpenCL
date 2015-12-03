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
 double * inner_ch_betas)
{
}

/********************/

void TeaCLContext::tea_leaf_dpcg_coarsen_matrix_kernel
(double * host_Kx, double * host_Ky)
{
    tiles.at(fine_tile)->tea_leaf_dpcg_coarsen_matrix_kernel(host_Kx, host_Ky);
}

void TeaCLContext::tea_leaf_dpcg_copy_reduced_coarse_grid
(double * global_coarse_Kx, double * global_coarse_Ky, double * global_coarse_Di)
{
    // COARSE tile
    tiles.at(coarse_tile)->tea_leaf_dpcg_copy_reduced_coarse_grid(global_coarse_Kx, global_coarse_Ky, global_coarse_Di);
}

void TeaCLContext::tea_leaf_dpcg_prolong_z_kernel
(double * host_t2)
{
    // TODO copy host_t2 to device
    //ENQUEUE(tea_leaf_dpcg_prolong_Z_device);
}

void TeaCLContext::tea_leaf_dpcg_subtract_u_kernel
(double * host_t2)
{
    // TODO copy host_t2 to device
    //ENQUEUE(tea_leaf_dpcg_subtract_u_device);
}

void TeaCLContext::tea_leaf_dpcg_restrict_zt_kernel
(double * host_ztr)
{
    // TODO copy host_ztr from device
    //ENQUEUE(tea_leaf_dpcg_restrict_ZT_device);
}

void TeaCLContext::tea_leaf_dpcg_matmul_zta_kernel
(double * host_ztaz)
{
    // TODO copy ztaz from device
    //ENQUEUE(tea_leaf_dpcg_matmul_ZTA_device);
}

void TeaCLContext::tea_leaf_dpcg_init_p_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_init_p_device);
}

void TeaCLContext::tea_leaf_dpcg_store_r_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_store_r_device);
}

void TeaCLContext::tea_leaf_dpcg_calc_rrn_kernel
(double * rrn)
{
    //ENQUEUE(tea_leaf_dpcg_calc_rrn_device);

    //*rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void TeaCLContext::tea_leaf_dpcg_calc_p_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_calc_p_device);
}


