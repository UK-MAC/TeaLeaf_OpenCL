#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

extern "C" void tea_leaf_cheby_init_kernel_ocl_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * theta)
{
    chunk.tea_leaf_cheby_init_kernel(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta);
}

extern "C" void tea_leaf_cheby_iterate_kernel_ocl_
(const int * cheby_calc_step)
{
    chunk.tea_leaf_cheby_iterate_kernel(*cheby_calc_step);
}

void CloverChunk::tea_leaf_calc_2norm_kernel
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

        if (preconditioner_type == TL_PREC_JAC_BLOCK)
        {
            tea_leaf_calc_2norm_device.setArg(2, vector_z);
        }
        else if (preconditioner_type == TL_PREC_JAC_DIAG)
        {
            tea_leaf_calc_2norm_device.setArg(2, vector_z);
        }
        else if (preconditioner_type == TL_PREC_NONE)
        {
            tea_leaf_calc_2norm_device.setArg(2, vector_r);
        }
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be 0 for u0, 1 for r, 2 for r*z", norm_array);
    }

    ENQUEUE_OFFSET(tea_leaf_calc_2norm_device);
    *norm = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

void CloverChunk::tea_leaf_cheby_init_kernel
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
    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);

    tea_leaf_cheby_solve_init_p_device.setArg(11, theta);
    tea_leaf_cheby_solve_init_p_device.setArg(12, rx);
    tea_leaf_cheby_solve_init_p_device.setArg(13, ry);

    tea_leaf_cheby_solve_calc_p_device.setArg(11, ch_alphas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(12, ch_betas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(13, rx);
    tea_leaf_cheby_solve_calc_p_device.setArg(14, ry);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_init_p_device);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

void CloverChunk::tea_leaf_cheby_iterate_kernel
(const int cheby_calc_step)
{
    tea_leaf_cheby_solve_calc_p_device.setArg(15, cheby_calc_step-1);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

