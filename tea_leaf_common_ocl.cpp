#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_calc_2norm_kernel_ocl_
(int* norm_array, double* norm)
{
    tea_context.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_kernel_init_common_ocl_
(const int * coefficient, double * dt, double * rx, double * ry)
{
    tea_context.tea_leaf_init_common(*coefficient, *dt, rx, ry);
}

// used by both
extern "C" void tea_leaf_kernel_finalise_ocl_
(void)
{
    tea_context.tea_leaf_finalise();
}

extern "C" void tea_leaf_calc_residual_ocl_
(void)
{
    tea_context.tea_leaf_calc_residual();
}

void TeaCLContext::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    FOR_EACH_TILE
    {
        if (norm_array == 0)
        {
            // norm of u0
            tile->tea_leaf_calc_2norm_device.setArg(0, tile->u0);
            tile->tea_leaf_calc_2norm_device.setArg(1, tile->u0);
        }
        else if (norm_array == 1)
        {
            // norm of r
            tile->tea_leaf_calc_2norm_device.setArg(0, tile->vector_r);
            tile->tea_leaf_calc_2norm_device.setArg(1, tile->vector_r);
        }
        else if (norm_array == 2)
        {
            // ddot(z, r)
            tile->tea_leaf_calc_2norm_device.setArg(0, tile->vector_r);

            if (preconditioner_type == TL_PREC_JAC_BLOCK)
            {
                tile->tea_leaf_calc_2norm_device.setArg(1, tile->vector_z);
            }
            else if (preconditioner_type == TL_PREC_JAC_DIAG)
            {
                tile->tea_leaf_calc_2norm_device.setArg(1, tile->vector_z);
            }
            else if (preconditioner_type == TL_PREC_NONE)
            {
                tile->tea_leaf_calc_2norm_device.setArg(1, tile->vector_r);
            }
        }
        else
        {
            DIE("Invalid value '%d' for norm_array passed, should be 0 for u0, 1 for r, 2 for r*z", norm_array);
        }
    }

#if 0 // RTAG
    ENQUEUE(tea_leaf_calc_2norm_device);
#endif // RTAG

    std::vector<int> indexes(1, 1);
    std::vector<double> reduced_values = sumReduceValues<double>(indexes);
    *norm = reduced_values.at(0);
}

void TeaCLContext::tea_leaf_init_common
(int coefficient, double dt, double * rx, double * ry)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    FOR_EACH_TILE
    {
        tile->tea_leaf_init_common_device.setArg(6, *rx);
        tile->tea_leaf_init_common_device.setArg(7, *ry);
        tile->tea_leaf_init_common_device.setArg(8, coefficient);

        tile->generate_chunk_init_u_device.setArg(1, tile->energy1);
    }

#if 0 // RTAG
    ENQUEUE(tea_leaf_init_common_device);
#endif // RTAG
#if 0 // RTAG
    ENQUEUE(generate_chunk_init_u_device);
#endif // RTAG
}

// both
void TeaCLContext::tea_leaf_finalise
(void)
{
#if 0 // RTAG
    ENQUEUE(tea_leaf_finalise_device);
#endif // RTAG
}

void TeaCLContext::tea_leaf_calc_residual
(void)
{
#if 0 // RTAG
    ENQUEUE(tea_leaf_calc_residual_device);
#endif // RTAG
}

// copy back dx/dy and calculate rx/ry
void TeaCLContext::calcrxry
(double dt, double * rx, double * ry)
{
    FOR_EACH_TILE
    {
        // make sure intialise chunk has finished
        tile->queue.finish();

        double dx, dy;

        try
        {
            // celldx/celldy never change, but done for consistency with fortran
            tile->queue.enqueueReadBuffer(tile->celldx, CL_TRUE,
                sizeof(double)*(1 + run_flags.halo_allocate_depth), sizeof(double), &dx);
            tile->queue.enqueueReadBuffer(tile->celldy, CL_TRUE,
                sizeof(double)*(1 + run_flags.halo_allocate_depth), sizeof(double), &dy);
        }
        catch (cl::Error e)
        {
            DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
                e.err(), e.what());
        }

        *rx = dt/(dx*dx);
        *ry = dt/(dy*dy);

        break;
    }
}

