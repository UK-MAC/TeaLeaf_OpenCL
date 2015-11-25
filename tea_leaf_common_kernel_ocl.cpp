#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

#include <cmath>

// copy back dx/dy and calculate rx/ry
void CloverChunk::calcrxry
(double dt, double * rx, double * ry)
{
    // make sure intialise chunk has finished
    queue.finish();

    double dx, dy;

    try
    {
        // celldx/celldy never change, but done for consistency with fortran
        queue.enqueueReadBuffer(celldx, CL_TRUE,
            sizeof(double)*(1 + halo_exchange_depth), sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*(1 + halo_exchange_depth), sizeof(double), &dy);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
            e.err(), e.what());
    }

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

extern "C" void tea_leaf_calc_2norm_kernel_ocl_
(int* norm_array, double* norm)
{
    chunk.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_common_init_kernel_ocl_
(const int * coefficient, double * dt, double * rx, double * ry,
 int * zero_boundary, int * reflective_boundary)
{
    chunk.tea_leaf_common_init(*coefficient, *dt, rx, ry,
        zero_boundary, *reflective_boundary);
}

extern "C" void tea_leaf_common_finalise_kernel_ocl_
(void)
{
    chunk.tea_leaf_finalise();
}

extern "C" void tea_leaf_calc_residual_ocl_
(void)
{
    chunk.tea_leaf_calc_residual();
}

void CloverChunk::tea_leaf_common_init
(int coefficient, double dt, double * rx, double * ry,
 int * zero_boundary, int reflective_boundary)
{
    if (coefficient != COEF_CONDUCTIVITY && coefficient != COEF_RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    tea_leaf_init_common_device.setArg(7, *rx);
    tea_leaf_init_common_device.setArg(8, *ry);
    tea_leaf_init_common_device.setArg(9, coefficient);
    ENQUEUE_OFFSET(tea_leaf_init_common_device);

    if (!reflective_boundary)
    {
        int zero_left = zero_boundary[0];
        int zero_right = zero_boundary[1];
        int zero_bottom = zero_boundary[2];
        int zero_top = zero_boundary[3];

        tea_leaf_zero_boundary_device.setArg(1, vector_Kx);
        tea_leaf_zero_boundary_device.setArg(2, vector_Ky);
        tea_leaf_zero_boundary_device.setArg(3, zero_left);
        tea_leaf_zero_boundary_device.setArg(4, zero_right);
        tea_leaf_zero_boundary_device.setArg(5, zero_bottom);
        tea_leaf_zero_boundary_device.setArg(6, zero_top);

        ENQUEUE_OFFSET(tea_leaf_zero_boundary_device);
    }

    generate_chunk_init_u_device.setArg(2, energy1);
    ENQUEUE_OFFSET(generate_chunk_init_u_device);
}

void CloverChunk::tea_leaf_finalise
(void)
{
    ENQUEUE_OFFSET(tea_leaf_finalise_device);
}

void CloverChunk::tea_leaf_calc_residual
(void)
{
    ENQUEUE_OFFSET(tea_leaf_calc_residual_device);
}

/********************/

