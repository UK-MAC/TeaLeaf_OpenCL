#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

#include <cassert>

extern "C" void tea_leaf_dpcg_coarsen_matrix_kernel_ocl_
(double * Kx_local, double * Ky_local)
{
    tea_context.tea_leaf_dpcg_coarsen_matrix_kernel(Kx_local, Ky_local);
}

extern "C" void tea_leaf_dpcg_prolong_z_kernel_ocl_
(double * t2_local)
{
    tea_context.tea_leaf_dpcg_prolong_z_kernel(t2_local);
}

extern "C" void tea_leaf_dpcg_subtract_u_kernel_ocl_
(double * t2_local)
{
    tea_context.tea_leaf_dpcg_subtract_u_kernel(t2_local);
}

extern "C" void tea_leaf_dpcg_restrict_zt_kernel_ocl_
(double * ztr_local)
{
    tea_context.tea_leaf_dpcg_restrict_zt_kernel(ztr_local);
}

extern "C" void tea_leaf_dpcg_matmul_zta_kernel_ocl_
(double * ztaz_local)
{
    tea_context.tea_leaf_dpcg_matmul_zta_kernel(ztaz_local);
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
(double * Kx_local, double * Ky_local)
{
    //ENQUEUE(tea_leaf_dpcg_coarsen_matrix_device);

    // TODO copy back memory
    Kx_local[0] = 1;
    Ky_local[1] = 1;
}

void TeaCLContext::tea_leaf_dpcg_prolong_z_kernel
(double * t2_local)
{
    // TODO copy t2_local to device
    //ENQUEUE(tea_leaf_dpcg_prolong_Z_device);
}

void TeaCLContext::tea_leaf_dpcg_subtract_u_kernel
(double * t2_local)
{
    // TODO copy t2_local to device
    //ENQUEUE(tea_leaf_dpcg_subtract_u_device);
}

void TeaCLContext::tea_leaf_dpcg_restrict_zt_kernel
(double * ztr_local)
{
    // TODO copy ztr_local from device
    //ENQUEUE(tea_leaf_dpcg_restrict_ZT_device);
}

void TeaCLContext::tea_leaf_dpcg_matmul_zta_kernel
(double * ztaz_local)
{
    // TODO copy ztr_local from device
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


