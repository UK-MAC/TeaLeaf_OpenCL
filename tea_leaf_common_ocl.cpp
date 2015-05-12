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
    if (norm_array == 0)
    {
        // norm of u0
        tea_leaf_calc_2norm_device.setArg(0, u0);
        tea_leaf_calc_2norm_device.setArg(1, u0);
    }
    else if (norm_array == 1)
    {
        // norm of r
        tea_leaf_calc_2norm_device.setArg(0, vector_r);
        tea_leaf_calc_2norm_device.setArg(1, vector_r);
    }
    else if (norm_array == 2)
    {
        // ddot(z, r)
        tea_leaf_calc_2norm_device.setArg(0, vector_r);

        if (preconditioner_type == TL_PREC_JAC_BLOCK)
        {
            tea_leaf_calc_2norm_device.setArg(1, vector_z);
        }
        else if (preconditioner_type == TL_PREC_JAC_DIAG)
        {
            tea_leaf_calc_2norm_device.setArg(1, vector_z);
        }
        else if (preconditioner_type == TL_PREC_NONE)
        {
            tea_leaf_calc_2norm_device.setArg(1, vector_r);
        }
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be 0 for u0, 1 for r, 2 for r*z", norm_array);
    }

    ENQUEUE_OFFSET(tea_leaf_calc_2norm_device);
    *norm = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

void TeaCLContext::tea_leaf_init_common
(int coefficient, double dt, double * rx, double * ry)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    tea_leaf_init_common_device.setArg(6, *rx);
    tea_leaf_init_common_device.setArg(7, *ry);
    tea_leaf_init_common_device.setArg(8, coefficient);
    ENQUEUE_OFFSET(tea_leaf_init_common_device);

    generate_chunk_init_u_device.setArg(1, energy1);
    ENQUEUE_OFFSET(generate_chunk_init_u_device);
}

// both
void TeaCLContext::tea_leaf_finalise
(void)
{
    ENQUEUE_OFFSET(tea_leaf_finalise_device);
}

void TeaCLContext::tea_leaf_calc_residual
(void)
{
    ENQUEUE_OFFSET(tea_leaf_calc_residual_device);
}

// copy back dx/dy and calculate rx/ry
void TeaCLContext::calcrxry
(double dt, double * rx, double * ry)
{
    // make sure intialise chunk has finished
    queue.finish();

    double dx, dy;

    try
    {
        // celldx/celldy never change, but done for consistency with fortran
        queue.enqueueReadBuffer(celldx, CL_TRUE,
            sizeof(double)*(1 + halo_allocate_depth), sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*(1 + halo_allocate_depth), sizeof(double), &dy);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
            e.err(), e.what());
    }

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

