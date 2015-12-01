#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_ppcg_init_ocl_
(const double * ch_alphas, const double * ch_betas,
 double* theta, int* n_inner_steps)
{
    tea_context.ppcg_init(ch_alphas, ch_betas, *theta, *n_inner_steps);
}

extern "C" void tea_leaf_ppcg_init_sd_kernel_ocl_
(void)
{
    tea_context.ppcg_init_sd_kernel();
}

extern "C" void tea_leaf_ppcg_inner_kernel_ocl_
(int * inner_step,
 int * bounds_extra,
 const int* chunk_neighbours)
{
    tea_context.tea_leaf_ppcg_inner_kernel(*inner_step, *bounds_extra, chunk_neighbours);
}

void TeaCLContext::ppcg_init
(const double * ch_alphas, const double * ch_betas,
 const double theta, const int n_inner_steps)
{
    // never going to do more than n_inner_steps steps? XXX
    size_t ch_buf_sz = n_inner_steps*sizeof(double);

    FOR_EACH_TILE
    {
        tile->tea_leaf_ppcg_solve_init_sd_device.setArg(11, theta);

        // upload to device
        tile->ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
        tile->ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);

        tile->queue.enqueueWriteBuffer(tile->ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
        tile->queue.enqueueWriteBuffer(tile->ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);

        tile->tea_leaf_ppcg_solve_calc_sd_device.setArg(9, tile->ch_alphas_device);
        tile->tea_leaf_ppcg_solve_calc_sd_device.setArg(10, tile->ch_betas_device);
    }
}

void TeaCLContext::ppcg_init_sd_kernel
(void)
{
    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_ppcg_solve_init_sd_device);
    }
}

void TeaCLContext::tea_leaf_ppcg_inner_kernel
(int inner_step, int bounds_extra, const int* chunk_neighbours)
{
    int step_depth = run_params.halo_exchange_depth - bounds_extra;

    int bounds_extra_x = bounds_extra;
    int bounds_extra_y = bounds_extra;

    int step_global_size[2];

    FOR_EACH_TILE
    {
        int step_offset[2] = {step_depth, step_depth};
        step_global_size[0] = tile->tile_x_cells + (run_params.halo_exchange_depth-step_depth)*2,
        step_global_size[1] = tile->tile_y_cells + (run_params.halo_exchange_depth-step_depth)*2;

        if (chunk_neighbours[CHUNK_LEFT - 1] == EXTERNAL_FACE)
        {
            step_offset[0] = run_params.halo_exchange_depth;
            step_global_size[0] -= (run_params.halo_exchange_depth-step_depth);
        }
        if (chunk_neighbours[CHUNK_RIGHT - 1] == EXTERNAL_FACE)
        {
            step_global_size[0] -= (run_params.halo_exchange_depth-step_depth);
            bounds_extra_x = 0;
        }

        if (chunk_neighbours[CHUNK_BOTTOM - 1] == EXTERNAL_FACE)
        {
            step_offset[1] = run_params.halo_exchange_depth;
            step_global_size[1] -= (run_params.halo_exchange_depth-step_depth);
        }
        if (chunk_neighbours[CHUNK_TOP - 1] == EXTERNAL_FACE)
        {
            step_global_size[1] -= (run_params.halo_exchange_depth-step_depth);
            bounds_extra_y = 0;
        }

        step_global_size[0] -= step_global_size[0] % tile->local_size[0];
        step_global_size[0] += tile->local_size[0];
        step_global_size[1] -= step_global_size[1] % tile->local_size[1];
        step_global_size[1] += tile->local_size[1];

        tile->tea_leaf_ppcg_solve_update_r_device.setArg(6, bounds_extra_x);
        tile->tea_leaf_ppcg_solve_update_r_device.setArg(7, bounds_extra_y);
        tile->tea_leaf_ppcg_solve_calc_sd_device.setArg(12, bounds_extra_x);
        tile->tea_leaf_ppcg_solve_calc_sd_device.setArg(13, bounds_extra_y);

        cl::NDRange step_offset_range(step_offset[0], step_offset[1]);
        cl::NDRange step_global_size_range(step_global_size[0], step_global_size[1]);

        tile->enqueueKernel(tile->tea_leaf_ppcg_solve_update_r_device, __LINE__, __FILE__,
                      step_offset_range,
                      step_global_size_range,
                      tile->local_size);

        tile->tea_leaf_ppcg_solve_calc_sd_device.setArg(11, inner_step - 1);

        tile->enqueueKernel(tile->tea_leaf_ppcg_solve_calc_sd_device, __LINE__, __FILE__,
                      step_offset_range,
                      step_global_size_range,
                      tile->local_size);
    }
}

