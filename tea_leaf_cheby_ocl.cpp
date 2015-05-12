#include "ocl_reduction.hpp"

extern "C" void tea_leaf_kernel_cheby_init_ocl_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * theta, double* error)
{
#if 0 // RTAG

    tea_context.tea_leaf_kernel_cheby_init(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta, error);

#endif //RTAG
}

extern "C" void tea_leaf_kernel_cheby_iterate_ocl_
(const double * ch_alphas, const double * ch_betas, int *n_coefs,
 const double * rx, const double * ry, const int * cheby_calc_step)
{
#if 0 // RTAG

    tea_context.tea_leaf_kernel_cheby_iterate(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *cheby_calc_step);

#endif //RTAG
}

void TeaCLContext::tea_leaf_kernel_cheby_init
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta, double* error)
{
#if 0 // RTAG
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
    tea_leaf_cheby_solve_calc_p_device.setArg(10, ch_alphas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(11, ch_betas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(12, rx);
    tea_leaf_cheby_solve_calc_p_device.setArg(13, ry);

    tea_leaf_cheby_solve_init_p_device.setArg(10, theta);
    tea_leaf_cheby_solve_init_p_device.setArg(11, rx);
    tea_leaf_cheby_solve_init_p_device.setArg(12, ry);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_init_p_device);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
#endif //RTAG
}

void TeaCLContext::tea_leaf_kernel_cheby_iterate
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const int cheby_calc_step)
{
#if 0 // RTAG

    tea_leaf_cheby_solve_calc_p_device.setArg(14, cheby_calc_step-1);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);

#endif //RTAG
}

/********************/

extern "C" void tea_leaf_kernel_solve_ocl_
(const double * rx, const double * ry, double * error)
{
#if 0 // RTAG

    tea_context.tea_leaf_kernel_jacobi(*rx, *ry, error);

#endif //RTAG
}

void TeaCLContext::tea_leaf_kernel_jacobi
(double rx, double ry, double* error)
{
#if 0 // RTAG

    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);

#endif //RTAG
}

/********************/
