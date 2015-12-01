#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_cheby_init_kernel_ocl_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * theta)
{
    tea_context.tea_leaf_cheby_init_kernel(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta);
}

extern "C" void tea_leaf_cheby_iterate_kernel_ocl_
(const int * cheby_calc_step)
{
    tea_context.tea_leaf_cheby_iterate_kernel(*cheby_calc_step);
}

void TeaCLContext::tea_leaf_cheby_init_kernel
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta)
{
    size_t ch_buf_sz = n_coefs*sizeof(double);

    FOR_EACH_TILE
    {
        cl_ulong max_constant;

        tile->device.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &max_constant);

        if (ch_buf_sz > max_constant)
        {
            DIE("Size to store requested number of chebyshev coefs (%d coefs -> %zu bytes) bigger than device max (%lu bytes). Set tl_max_iters to a smaller value\n", n_coefs, ch_buf_sz, max_constant);
        }
    }

    FOR_EACH_TILE
    {
        // upload to device
        tile->ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
        tile->ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);

        tile->queue.enqueueWriteBuffer(tile->ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
        tile->queue.enqueueWriteBuffer(tile->ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);

        tile->tea_leaf_cheby_solve_init_p_device.setArg(11, theta);
        tile->tea_leaf_cheby_solve_init_p_device.setArg(12, rx);
        tile->tea_leaf_cheby_solve_init_p_device.setArg(13, ry);

        tile->tea_leaf_cheby_solve_calc_p_device.setArg(11, tile->ch_alphas_device);
        tile->tea_leaf_cheby_solve_calc_p_device.setArg(12, tile->ch_betas_device);
        tile->tea_leaf_cheby_solve_calc_p_device.setArg(13, rx);
        tile->tea_leaf_cheby_solve_calc_p_device.setArg(14, ry);
    }

    FOR_EACH_TILE
    {
        ENQUEUE(tea_leaf_cheby_solve_init_p_device);

        ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
    }
}

void TeaCLContext::tea_leaf_cheby_iterate_kernel
(const int cheby_calc_step)
{
    FOR_EACH_TILE
    {
        tile->tea_leaf_cheby_solve_calc_p_device.setArg(15, cheby_calc_step-1);

        ENQUEUE(tea_leaf_cheby_solve_calc_p_device);
        ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
    }
}

