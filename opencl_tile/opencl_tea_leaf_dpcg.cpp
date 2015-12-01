#include "../ocl_common.hpp"

void TeaOpenCLTile::tea_leaf_dpcg_coarsen_matrix_kernel
(double * Kx_local, double * Ky_local)
{
    //ENQUEUE(tea_leaf_dpcg_coarsen_matrix_device);

    // TODO copy back memory
    Kx_local[0] = 1;
    Ky_local[1] = 1;
}

void TeaOpenCLTile::tea_leaf_dpcg_prolong_z_kernel
(double * t2_local)
{
    // TODO copy t2_local to device
    //ENQUEUE(tea_leaf_dpcg_prolong_Z_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_subtract_u_kernel
(double * t2_local)
{
    // TODO copy t2_local to device
    //ENQUEUE(tea_leaf_dpcg_subtract_u_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_restrict_zt_kernel
(double * ztr_local)
{
    // TODO copy ztr_local from device
    //ENQUEUE(tea_leaf_dpcg_restrict_ZT_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_matmul_zta_kernel
(double * ztaz_local)
{
    // TODO copy ztr_local from device
    //ENQUEUE(tea_leaf_dpcg_matmul_ZTA_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_init_p_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_init_p_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_store_r_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_store_r_device);
}

void TeaOpenCLTile::tea_leaf_dpcg_calc_rrn_kernel
(double * rrn)
{
    //ENQUEUE(tea_leaf_dpcg_calc_rrn_device);

    //*rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void TeaOpenCLTile::tea_leaf_dpcg_calc_p_kernel
(void)
{
    //ENQUEUE(tea_leaf_dpcg_calc_p_device);
}

