#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_calc_2norm_kernel_ocl_
(int* norm_array, double* norm)
{
    tea_context.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_kernel_init_common_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, const int * chunk_neighbours)
{
    tea_context.tea_leaf_init_common(*coefficient, *dt, rx, ry, chunk_neighbours);
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

        ENQUEUE(tea_leaf_calc_2norm_device);
    }

    *norm = sumReduceValues<double>(std::vector<int>(1, 1)).at(0);
}

void TeaCLContext::tea_leaf_init_common
(int coefficient, double dt, double * rx, double * ry, const int * chunk_neighbours)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    int depth = run_flags.halo_exchange_depth;
    std::vector<double> zeros((std::max(run_flags.x_cells, run_flags.y_cells) + 2*depth)*depth, 0);

    #define ZERO_BOUNDARY(face, dir, xy) \
    if (chunk_neighbours[CHUNK_ ## face - 1] == EXTERNAL_FACE && \
        tile->isExternal(CHUNK_ ## face - 1)) \
    {   \
        tile->queue.enqueueWriteBuffer(tile->face##_buffer, CL_TRUE, 0, \
            sizeof(double)*(tile->tile_##xy##_cells + 2*depth)*depth, &zeros.front()); \
        tile->unpack_##face##_buffer_device.setArg(0, 0);     \
        tile->unpack_##face##_buffer_device.setArg(1, 0);     \
        tile->unpack_##face##_buffer_device.setArg(2, tile->vector_K##xy);      \
        tile->unpack_##face##_buffer_device.setArg(3, tile->face##_buffer);     \
        tile->unpack_##face##_buffer_device.setArg(4, depth);       \
        tile->unpack_##face##_buffer_device.setArg(5, 0);     \
        cl::NDRange offset_plus_one(tile->update_##dir##_offset[depth][0]+1,  \
            tile->update_##dir##_offset[depth][1]+1); \
        tile->enqueueKernel(tile->unpack_##face##_buffer_device, \
                      __LINE__, __FILE__,  \
                      offset_plus_one, \
                      tile->update_##dir##_global_size[depth], \
                      tile->update_##dir##_local_size[depth]); \
    }

    FOR_EACH_TILE
    {
        tile->tea_leaf_init_common_device.setArg(6, *rx);
        tile->tea_leaf_init_common_device.setArg(7, *ry);
        tile->tea_leaf_init_common_device.setArg(8, coefficient);

        tile->generate_chunk_init_u_device.setArg(1, tile->energy1);

        ENQUEUE(tea_leaf_init_common_device);
        ENQUEUE(generate_chunk_init_u_device);

        ZERO_BOUNDARY(left, lr, x)
        ZERO_BOUNDARY(right, lr, x)
        ZERO_BOUNDARY(bottom, bt, y)
        ZERO_BOUNDARY(top, bt, y)
    }
}

// both
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

