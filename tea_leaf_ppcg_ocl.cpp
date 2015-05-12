#include "ocl_common.hpp"

extern "C" void tea_leaf_kernel_ppcg_init_ocl_
(const double * ch_alphas, const double * ch_betas,
 double* theta, int* n_inner_steps)
{
#if 0 // RTAG

    tea_context.ppcg_init(ch_alphas, ch_betas, *theta, *n_inner_steps);

#endif //RTAG
}

extern "C" void tea_leaf_kernel_ppcg_init_sd_ocl_
(void)
{
#if 0 // RTAG

    tea_context.ppcg_init_sd();

#endif //RTAG
}

extern "C" void tea_leaf_kernel_ppcg_inner_ocl_
(int * ppcg_cur_step,
 int * max_steps,
 const int* chunk_neighbours)
{
#if 0 // RTAG

    tea_context.ppcg_inner(*ppcg_cur_step, *max_steps, chunk_neighbours);

#endif //RTAG
}

void TeaCLContext::ppcg_init
(const double * ch_alphas, const double * ch_betas,
 const double theta, const int n_inner_steps)
{
#if 0 // RTAG

    tea_leaf_ppcg_solve_init_sd_device.setArg(10, theta);

    // never going to do more than n_inner_steps steps? XXX
    size_t ch_buf_sz = n_inner_steps*sizeof(double);

    // upload to device
    ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);

    tea_leaf_ppcg_solve_calc_sd_device.setArg(8, ch_alphas_device);
    tea_leaf_ppcg_solve_calc_sd_device.setArg(9, ch_betas_device);

#endif //RTAG
}

void TeaCLContext::ppcg_init_sd
(void)
{
#if 0 // RTAG

    ENQUEUE_OFFSET(tea_leaf_ppcg_solve_init_sd_device);

#endif //RTAG
}

void TeaCLContext::ppcg_inner
(int ppcg_cur_step, int max_steps, const int* chunk_neighbours)
{
#if 0 // RTAG
    for (int step_depth = 1 + (halo_allocate_depth - halo_exchange_depth);
        step_depth <= halo_allocate_depth; step_depth++)
    {
        size_t step_offset[2] = {step_depth, step_depth};
        size_t step_global_size[2] = {
            x_max + (halo_allocate_depth-step_depth)*2,
            y_max + (halo_allocate_depth-step_depth)*2};

        if (chunk_neighbours[CHUNK_LEFT - 1] == EXTERNAL_FACE)
        {
            step_offset[0] = halo_allocate_depth;
            step_global_size[0] -= (halo_exchange_depth-step_depth);
        }
        if (chunk_neighbours[CHUNK_RIGHT - 1] == EXTERNAL_FACE)
        {
            step_global_size[0] -= (halo_exchange_depth-step_depth);
        }
        if (chunk_neighbours[CHUNK_BOTTOM - 1] == EXTERNAL_FACE)
        {
            step_offset[1] = halo_allocate_depth;
            step_global_size[1] -= (halo_exchange_depth-step_depth);
        }
        if (chunk_neighbours[CHUNK_TOP - 1] == EXTERNAL_FACE)
        {
            step_global_size[1] -= (halo_exchange_depth-step_depth);
        }

        cl::NDRange step_offset_range(step_offset[0], step_offset[1]);
        cl::NDRange step_global_size_range(step_global_size[0], step_global_size[1]);

        enqueueKernel(tea_leaf_ppcg_solve_update_r_device, __LINE__, __FILE__,
                      step_offset_range,
                      step_global_size_range,
                      cl::NullRange);

        tea_leaf_ppcg_solve_calc_sd_device.setArg(10, ppcg_cur_step - 1 + (step_depth - 1));

        enqueueKernel(tea_leaf_ppcg_solve_calc_sd_device, __LINE__, __FILE__,
                      step_offset_range,
                      step_global_size_range,
                      cl::NullRange);

        if (ppcg_cur_step + step_depth >= max_steps)
        {
            break;
        }
    }
#endif // RTAG
}

