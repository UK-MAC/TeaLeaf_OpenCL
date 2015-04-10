#include "ocl_common.hpp"

#include <cassert>
#include <cmath>

extern CloverChunk chunk;

// same as in fortran
#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

extern "C" void tea_leaf_calc_2norm_kernel_ocl_
(int* norm_array, double* norm)
{
    chunk.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_kernel_cheby_init_ocl_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * theta, double* error)
{
    chunk.tea_leaf_kernel_cheby_init(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta, error);
}

extern "C" void tea_leaf_kernel_cheby_iterate_ocl_
(const double * ch_alphas, const double * ch_betas, int *n_coefs,
 const double * rx, const double * ry, const int * cheby_calc_step)
{
    chunk.tea_leaf_kernel_cheby_iterate(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *cheby_calc_step);
}

void CloverChunk::tea_leaf_calc_2norm_kernel
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
        else
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

void CloverChunk::tea_leaf_kernel_cheby_init
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta, double* error)
{
    size_t ch_buf_sz = n_coefs*sizeof(double);

    // upload to device
    ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);
    tea_leaf_cheby_solve_calc_p_device.setArg(8, ch_alphas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(9, ch_betas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(10, rx);
    tea_leaf_cheby_solve_calc_p_device.setArg(11, ry);

    tea_leaf_cheby_solve_init_p_device.setArg(8, theta);
    tea_leaf_cheby_solve_init_p_device.setArg(9, rx);
    tea_leaf_cheby_solve_init_p_device.setArg(10, ry);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_init_p_device);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

void CloverChunk::tea_leaf_kernel_cheby_iterate
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const int cheby_calc_step)
{
    tea_leaf_cheby_solve_calc_p_device.setArg(12, cheby_calc_step-1);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

/********************/

// CG solver functions
extern "C" void tea_leaf_kernel_init_cg_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, double * rro)
{
    chunk.tea_leaf_init_cg(*coefficient, *dt, rx, ry, rro);
}

extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_w_
(const double * rx, const double * ry, double * pw)
{
    chunk.tea_leaf_kernel_cg_calc_w(*rx, *ry, pw);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_ur_
(double * alpha, double * rrn)
{
    chunk.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);
}
extern "C" void tea_leaf_kernel_solve_cg_ocl_calc_p_
(double * beta)
{
    chunk.tea_leaf_kernel_cg_calc_p(*beta);
}

// copy back dx/dy and calculate rx/ry
void CloverChunk::calcrxry
(double dt, double * rx, double * ry)
{
    // make sure intialise chunk has finished
    queue.finish();

    double dx, dy;

    try
    {
        // celldx/celldy never change, but done for consistency with fortran
        queue.enqueueReadBuffer(celldx, CL_TRUE,
            sizeof(double)*x_min, sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*y_min, sizeof(double), &dy);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
            e.err(), e.what());
    }

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

/********************/

void CloverChunk::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV || tea_solver == TEA_ENUM_PPCG);

    // Assume calc_residual has been called before this (to calculate initial_residual)

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        block_jacobi_offset = cl::NDRange(2, 0);

        block_jacobi_local = cl::NDRange(32, 4);

        size_t xl = x_max;
        xl += block_jacobi_local[0] - (xl % block_jacobi_local[0]);
        size_t yl = y_max/JACOBI_BLOCK_SIZE;
        yl += block_jacobi_local[1] - (yl % block_jacobi_local[1]);

        block_jacobi_global = cl::NDRange(xl, yl);

        enqueueKernel(tea_leaf_block_init_device, __LINE__, __FILE__,
                      block_jacobi_offset,
                      block_jacobi_global,
                      block_jacobi_local);

        enqueueKernel(tea_leaf_block_solve_device, __LINE__, __FILE__,
                      block_jacobi_offset,
                      block_jacobi_global,
                      block_jacobi_local);
    }
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
    {
        ENQUEUE_OFFSET(tea_leaf_init_jac_diag_device);
    }

    ENQUEUE_OFFSET(tea_leaf_cg_solve_init_p_device);

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
}

void CloverChunk::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

void CloverChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        enqueueKernel(tea_leaf_cg_solve_calc_ur_device, __LINE__, __FILE__,
                      block_jacobi_offset,
                      block_jacobi_global,
                      block_jacobi_local);
    }
    else
    {
        ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);

    }

    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
}

void CloverChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);
}

/********************/

extern "C" void tea_leaf_kernel_solve_ocl_
(const double * rx, const double * ry, double * error)
{
    chunk.tea_leaf_kernel_jacobi(*rx, *ry, error);
}

void CloverChunk::tea_leaf_kernel_jacobi
(double rx, double ry, double* error)
{
    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

/********************/

extern "C" void tea_leaf_kernel_init_common_ocl_
(const int * coefficient, double * dt, double * rx, double * ry)
{
    chunk.tea_leaf_init_common(*coefficient, *dt, rx, ry);
}

// used by both
extern "C" void tea_leaf_kernel_finalise_ocl_
(void)
{
    chunk.tea_leaf_finalise();
}

extern "C" void tea_leaf_calc_residual_ocl_
(void)
{
    chunk.tea_leaf_calc_residual();
}

void CloverChunk::tea_leaf_init_common
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
}

// both
void CloverChunk::tea_leaf_finalise
(void)
{
    ENQUEUE_OFFSET(tea_leaf_finalise_device);
}

void CloverChunk::tea_leaf_calc_residual
(void)
{
    ENQUEUE_OFFSET(tea_leaf_calc_residual_device);
}

/********************/

extern "C" void tea_leaf_kernel_ppcg_init_ocl_
(const double * ch_alphas, const double * ch_betas,
 double* theta, int* n_inner_steps)
{
    chunk.ppcg_init(ch_alphas, ch_betas, *theta, *n_inner_steps);
}

extern "C" void tea_leaf_kernel_ppcg_init_sd_ocl_
(void)
{
    chunk.ppcg_init_sd();
}

extern "C" void tea_leaf_kernel_ppcg_inner_ocl_
(int * ppcg_cur_step)
{
    chunk.ppcg_inner(*ppcg_cur_step);
}

void CloverChunk::ppcg_init
(const double * ch_alphas, const double * ch_betas,
 const double theta, const int n_inner_steps)
{
    tea_leaf_ppcg_solve_init_sd_device.setArg(8, theta);

    // never going to do more than n_inner_steps steps? XXX
    size_t ch_buf_sz = n_inner_steps*sizeof(double);

    // upload to device
    ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);

    tea_leaf_ppcg_solve_calc_sd_device.setArg(4, ch_alphas_device);
    tea_leaf_ppcg_solve_calc_sd_device.setArg(5, ch_betas_device);
}

void CloverChunk::ppcg_init_sd
(void)
{
    if (preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        enqueueKernel(tea_leaf_ppcg_solve_init_sd_device, __LINE__, __FILE__,
                      block_jacobi_offset,
                      block_jacobi_global,
                      block_jacobi_local);
    }
    else
    {
        ENQUEUE_OFFSET(tea_leaf_ppcg_solve_init_sd_device);
    }
}

void CloverChunk::ppcg_inner
(int ppcg_cur_step)
{
    ENQUEUE_OFFSET(tea_leaf_ppcg_solve_update_r_device);

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        enqueueKernel(tea_leaf_block_solve_device, __LINE__, __FILE__,
                      block_jacobi_offset,
                      block_jacobi_global,
                      block_jacobi_local);
    }

    tea_leaf_ppcg_solve_calc_sd_device.setArg(6, ppcg_cur_step - 1);
    ENQUEUE_OFFSET(tea_leaf_ppcg_solve_calc_sd_device);
}

