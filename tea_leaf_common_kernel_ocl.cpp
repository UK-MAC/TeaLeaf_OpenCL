#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

#include <cmath>

// copy back dx/dy and calculate rx/ry
void TeaCLContext::calcrxry
(double dt, double * rx, double * ry)
{
    FOR_EACH_TILE
    {
        // make sure intialise tea_context has finished
        tile->queue.finish();

        double dx, dy;

        try
        {
            // celldx/celldy never change, but done for consistency with fortran
            tile->queue.enqueueReadBuffer(tile->celldx, CL_TRUE,
                sizeof(double)*(1 + run_params.halo_exchange_depth), sizeof(double), &dx);
            tile->queue.enqueueReadBuffer(tile->celldy, CL_TRUE,
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
}

extern "C" void tea_leaf_calc_2norm_kernel_ocl_
(int* norm_array, double* norm)
{
    tea_context.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_common_init_kernel_ocl_
(const int * coefficient, double * dt, double * rx, double * ry,
 int * zero_boundary, int * reflective_boundary)
{
    tea_context.tea_leaf_common_init(*coefficient, *dt, rx, ry,
        zero_boundary, *reflective_boundary);
}

extern "C" void tea_leaf_common_finalise_kernel_ocl_
(void)
{
    tea_context.tea_leaf_finalise();
}

extern "C" void tea_leaf_calc_residual_ocl_
(void)
{
    tea_context.tea_leaf_calc_residual();
}

/********************/

void TeaCLContext::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    if (norm_array == 0)
    {
        // norm of u0
        FOR_EACH_TILE
        {
            tile->tea_leaf_calc_2norm_device.setArg(1, tile->u0);
            tile->tea_leaf_calc_2norm_device.setArg(2, tile->u0);
        }
    }
    else if (norm_array == 1)
    {
        // norm of r
        FOR_EACH_TILE
        {
            tile->tea_leaf_calc_2norm_device.setArg(1, tile->vector_r);
            tile->tea_leaf_calc_2norm_device.setArg(2, tile->vector_r);
        }
    }
    else if (norm_array == 2)
    {
        // ddot(z, r)
        FOR_EACH_TILE
        {
            tile->tea_leaf_calc_2norm_device.setArg(1, tile->vector_r);
        }

        if (run_params.preconditioner_type == TL_PREC_JAC_BLOCK)
        {
            FOR_EACH_TILE
            {
                tile->tea_leaf_calc_2norm_device.setArg(2, tile->vector_z);
            }
        }
        else if (run_params.preconditioner_type == TL_PREC_JAC_DIAG)
        {
            FOR_EACH_TILE
            {
                tile->tea_leaf_calc_2norm_device.setArg(2, tile->vector_z);
            }
        }
        else if (run_params.preconditioner_type == TL_PREC_NONE)
        {
            FOR_EACH_TILE
            {
                tile->tea_leaf_calc_2norm_device.setArg(2, tile->vector_r);
            }
        }
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be 0 for u0, 1 for r, 2 for r*z", norm_array);
    }

    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_calc_2norm_device);
        *norm = tile->reduceValue<double>(tile->sum_red_kernels_double, tile->reduce_buf_1);
    }
}

void TeaCLContext::tea_leaf_common_init
(int coefficient, double dt, double * rx, double * ry,
 int * zero_boundary, int reflective_boundary)
{
    if (coefficient != COEF_CONDUCTIVITY && coefficient != COEF_RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    FOR_EACH_TILE
    {
        tile->tea_leaf_init_common_device.setArg(7, *rx);
        tile->tea_leaf_init_common_device.setArg(8, *ry);
        tile->tea_leaf_init_common_device.setArg(9, coefficient);
        ENQUEUE(tea_leaf_init_common_device);
    }

    if (!reflective_boundary)
    {
        int zero_left = zero_boundary[0];
        int zero_right = zero_boundary[1];
        int zero_bottom = zero_boundary[2];
        int zero_top = zero_boundary[3];

        FOR_EACH_TILE
        {
            tile->tea_leaf_zero_boundary_device.setArg(1, tile->vector_Kx);
            tile->tea_leaf_zero_boundary_device.setArg(2, tile->vector_Ky);
            tile->tea_leaf_zero_boundary_device.setArg(3, zero_left);
            tile->tea_leaf_zero_boundary_device.setArg(4, zero_right);
            tile->tea_leaf_zero_boundary_device.setArg(5, zero_bottom);
            tile->tea_leaf_zero_boundary_device.setArg(6, zero_top);

            ENQUEUE(tea_leaf_zero_boundary_device);
        }
    }

    FOR_EACH_TILE
    {
        tile->generate_chunk_init_u_device.setArg(2, tile->energy1);
        ENQUEUE(generate_chunk_init_u_device);
    }
}

void TeaCLContext::tea_leaf_finalise
(void)
{
    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_finalise_device);
    }
}

void TeaCLContext::tea_leaf_calc_residual
(void)
{
    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_calc_residual_device);
    }
}

/********************/

