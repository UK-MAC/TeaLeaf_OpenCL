#include "../ctx_common.hpp"

#include <cmath>

// copy back dx/dy and calculate rx/ry
void TeaOpenCLTile::calcrxry
(double dt, double * rx, double * ry)
{
    // make sure intialise tea_context has finished
    queue.finish();

    double dx, dy;

    try
    {
        // celldx/celldy never change, but done for consistency with fortran
        queue.enqueueReadBuffer(celldx, CL_TRUE,
            sizeof(double)*(1 + run_params.halo_exchange_depth), sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*(1 + run_params.halo_exchange_depth), sizeof(double), &dy);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
            e.err(), e.what());
    }

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

/********************/

void TeaOpenCLTile::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    if (norm_array == 0)
    {
        // norm of u0
        tea_leaf_calc_2norm_device.setArg(1, u0);
        tea_leaf_calc_2norm_device.setArg(2, u0);
    }
    else if (norm_array == 1)
    {
        // norm of r
        tea_leaf_calc_2norm_device.setArg(1, vector_r);
        tea_leaf_calc_2norm_device.setArg(2, vector_r);
    }
    else if (norm_array == 2)
    {
        // ddot(z, r)
        tea_leaf_calc_2norm_device.setArg(1, vector_r);

        if (run_params.preconditioner_type == TL_PREC_JAC_BLOCK)
        {
            tea_leaf_calc_2norm_device.setArg(2, vector_z);
        }
        else if (run_params.preconditioner_type == TL_PREC_JAC_DIAG)
        {
            tea_leaf_calc_2norm_device.setArg(2, vector_z);
        }
        else if (run_params.preconditioner_type == TL_PREC_NONE)
        {
            tea_leaf_calc_2norm_device.setArg(2, vector_r);
        }
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be 0 for u0, 1 for r, 2 for r*z", norm_array);
    }

    ENQUEUE(tea_leaf_calc_2norm_device);
    *norm = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

void TeaOpenCLTile::tea_leaf_common_init
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
    ENQUEUE(tea_leaf_init_common_device);

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

        ENQUEUE(tea_leaf_zero_boundary_device);
    }

    generate_chunk_init_u_device.setArg(2, energy1);
    ENQUEUE(generate_chunk_init_u_device);
}

void TeaOpenCLTile::tea_leaf_finalise
(void)
{
    ENQUEUE(tea_leaf_finalise_device);
}

void TeaOpenCLTile::tea_leaf_calc_residual
(void)
{
    ENQUEUE(tea_leaf_calc_residual_device);
}

