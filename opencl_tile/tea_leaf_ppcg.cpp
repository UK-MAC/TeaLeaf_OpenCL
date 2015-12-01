#include "../ocl_common.hpp"

void TeaOpenCLTile::ppcg_init
(const double * ch_alphas, const double * ch_betas,
 const double theta, const int n_inner_steps)
{
    // never going to do more than n_inner_steps steps? XXX
    size_t ch_buf_sz = n_inner_steps*sizeof(double);

    tea_leaf_ppcg_solve_init_sd_device.setArg(11, theta);

    // upload to device
    ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);

    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);

    tea_leaf_ppcg_solve_calc_sd_device.setArg(9, ch_alphas_device);
    tea_leaf_ppcg_solve_calc_sd_device.setArg(10, ch_betas_device);
}

void TeaOpenCLTile::ppcg_init_sd_kernel
(void)
{
    ENQUEUE(tea_leaf_ppcg_solve_init_sd_device);
}

void TeaOpenCLTile::tea_leaf_ppcg_inner_kernel
(int inner_step, int bounds_extra, const int* chunk_neighbours)
{
    int step_depth = run_params.halo_exchange_depth - bounds_extra;

    int bounds_extra_x = bounds_extra;
    int bounds_extra_y = bounds_extra;

    int step_global_size[2];

    int step_offset[2] = {step_depth, step_depth};
    step_global_size[0] = tile_x_cells + (run_params.halo_exchange_depth-step_depth)*2,
    step_global_size[1] = tile_y_cells + (run_params.halo_exchange_depth-step_depth)*2;

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

    step_global_size[0] -= step_global_size[0] % local_size[0];
    step_global_size[0] += local_size[0];
    step_global_size[1] -= step_global_size[1] % local_size[1];
    step_global_size[1] += local_size[1];

    tea_leaf_ppcg_solve_update_r_device.setArg(6, bounds_extra_x);
    tea_leaf_ppcg_solve_update_r_device.setArg(7, bounds_extra_y);
    tea_leaf_ppcg_solve_calc_sd_device.setArg(12, bounds_extra_x);
    tea_leaf_ppcg_solve_calc_sd_device.setArg(13, bounds_extra_y);

    cl::NDRange step_offset_range(step_offset[0], step_offset[1]);
    cl::NDRange step_global_size_range(step_global_size[0], step_global_size[1]);

    enqueueKernel(tea_leaf_ppcg_solve_update_r_device, __LINE__, __FILE__,
                  step_offset_range,
                  step_global_size_range,
                  local_size);

    tea_leaf_ppcg_solve_calc_sd_device.setArg(11, inner_step - 1);

    enqueueKernel(tea_leaf_ppcg_solve_calc_sd_device, __LINE__, __FILE__,
                  step_offset_range,
                  step_global_size_range,
                  local_size);
}

