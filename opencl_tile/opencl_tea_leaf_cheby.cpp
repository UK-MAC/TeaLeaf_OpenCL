#include "../ctx_common.hpp"

void TeaOpenCLTile::tea_leaf_cheby_init_kernel
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta)
{
    size_t ch_buf_sz = n_coefs*sizeof(double);

    cl_ulong max_constant;

    device.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &max_constant);

    if (ch_buf_sz > max_constant)
    {
        DIE("Size to store requested number of chebyshev coefs (%d coefs -> %zu bytes) bigger than device max (%lu bytes). Set tl_max_iters to a smaller value\n", n_coefs, ch_buf_sz, max_constant);
    }

    // upload to device
    ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);

    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);

    tea_leaf_cheby_solve_init_p_device.setArg(11, theta);
    tea_leaf_cheby_solve_init_p_device.setArg(12, rx);
    tea_leaf_cheby_solve_init_p_device.setArg(13, ry);

    tea_leaf_cheby_solve_calc_p_device.setArg(11, ch_alphas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(12, ch_betas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(13, rx);
    tea_leaf_cheby_solve_calc_p_device.setArg(14, ry);

    ENQUEUE(tea_leaf_cheby_solve_init_p_device);

    ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
}

void TeaOpenCLTile::tea_leaf_cheby_iterate_kernel
(const int cheby_calc_step)
{
    tea_leaf_cheby_solve_calc_p_device.setArg(15, cheby_calc_step-1);

    ENQUEUE(tea_leaf_cheby_solve_calc_p_device);
    ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
}


