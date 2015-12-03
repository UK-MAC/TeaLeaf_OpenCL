#include "../ctx_common.hpp"
#include "opencl_reduction.hpp"

template <>
void TeaOpenCLTile::getKxKy
<cl::Buffer>
(cl::Buffer * Kx, cl::Buffer * Ky)
{
    Kx = &vector_Kx;
    Ky = &vector_Ky;
}

void TeaOpenCLTile::tea_leaf_dpcg_coarsen_matrix_kernel
(double * host_Kx, double * host_Ky, tile_ptr_t & coarse_tile)
{
    cl::Buffer *coarse_Kx=NULL, *coarse_Ky=NULL;
    coarse_tile->getKxKy(coarse_Kx, coarse_Ky);

    tea_leaf_dpcg_coarsen_matrix_device.setArg(3, *coarse_Kx);
    tea_leaf_dpcg_coarsen_matrix_device.setArg(4, *coarse_Ky);

    queue.enqueueReadBuffer(*coarse_Kx, CL_TRUE, 0,
        coarse_tile->tile_x_cells*coarse_tile->tile_y_cells*sizeof(double),
        host_Kx, NULL, NULL);

    queue.enqueueReadBuffer(*coarse_Ky, CL_TRUE, 0,
        coarse_tile->tile_x_cells*coarse_tile->tile_y_cells*sizeof(double),
        host_Ky, NULL, NULL);
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

