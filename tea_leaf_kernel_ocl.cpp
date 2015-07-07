#include "ocl_common.hpp"
#include "ocl_reduction.hpp"

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

    tea_leaf_cheby_solve_init_p_device.setArg(10, theta);
    tea_leaf_cheby_solve_init_p_device.setArg(11, rx);
    tea_leaf_cheby_solve_init_p_device.setArg(12, ry);

    tea_leaf_cheby_solve_calc_p_device.setArg(10, ch_alphas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(11, ch_betas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(12, rx);
    tea_leaf_cheby_solve_calc_p_device.setArg(13, ry);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_init_p_device);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

void CloverChunk::tea_leaf_cheby_iterate_kernel
(const int cheby_calc_step)
{
    tea_leaf_cheby_solve_calc_p_device.setArg(14, cheby_calc_step-1);

    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
}

/********************/

// CG solver functions
extern "C" void tea_leaf_cg_init_kernel_ocl_
(double * rro)
{
    chunk.tea_leaf_cg_init_kernel(rro);
}

extern "C" void tea_leaf_cg_calc_w_kernel_ocl_
(double * pw)
{
    chunk.tea_leaf_cg_calc_w_kernel(pw);
}
extern "C" void tea_leaf_cg_calc_ur_kernel_ocl_
(double * alpha, double * rrn)
{
    chunk.tea_leaf_cg_calc_ur_kernel(*alpha, rrn);
}
extern "C" void tea_leaf_cg_calc_p_kernel_ocl_
(double * beta)
{
    chunk.tea_leaf_cg_calc_p_kernel(*beta);
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
            sizeof(double)*(1 + halo_exchange_depth), sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*(1 + halo_exchange_depth), sizeof(double), &dy);
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

void CloverChunk::tea_leaf_cg_init_kernel
(double * rro)
{
    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV || tea_solver == TEA_ENUM_PPCG);

    // Assume calc_residual has been called before this (to calculate initial_residual)

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        ENQUEUE_OFFSET(tea_leaf_block_init_device);
        ENQUEUE_OFFSET(tea_leaf_block_solve_device);
    }
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
    {
        ENQUEUE_OFFSET(tea_leaf_init_jac_diag_device);
    }

    ENQUEUE_OFFSET(tea_leaf_cg_solve_init_p_device);

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
}

void CloverChunk::tea_leaf_cg_calc_w_kernel
(double* pw)
{
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

void CloverChunk::tea_leaf_cg_calc_ur_kernel
(double alpha, double* rrn)
{
    tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);

    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_5);
}

void CloverChunk::tea_leaf_cg_calc_p_kernel
(double beta)
{
    tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);
}

/********************/

extern "C" void tea_leaf_jacobi_solve_kernel_ocl_
(double * error)
{
    chunk.tea_leaf_jacobi_solve_kernel(error);
}

void CloverChunk::tea_leaf_jacobi_solve_kernel
(double* error)
{
    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

/********************/

extern "C" void tea_leaf_common_init_kernel_ocl_
(const int * coefficient, double * dt, double * rx, double * ry, int * chunk_neighbours)
{
    chunk.tea_leaf_init_common(*coefficient, *dt, rx, ry, chunk_neighbours);
}

// used by both
extern "C" void tea_leaf_common_finalise_kernel_ocl_
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
(int coefficient, double dt, double * rx, double * ry, int * chunk_neighbours)
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

    int depth = halo_exchange_depth;
    std::vector<double> zeros((std::max(x_max, y_max) + 2*depth)*depth, 0);

    #define ZERO_BOUNDARY(face, dir, xy) \
    if (chunk_neighbours[CHUNK_ ## face - 1] == EXTERNAL_FACE)\
    {   \
        queue.enqueueWriteBuffer(face##_buffer, CL_TRUE, 0, \
            sizeof(double)*(xy##_max + 2*depth)*depth, &zeros.front()); \
        unpack_##face##_buffer_device.setArg(0, 0);     \
        unpack_##face##_buffer_device.setArg(1, 0);     \
        unpack_##face##_buffer_device.setArg(2, vector_K##xy);      \
        unpack_##face##_buffer_device.setArg(3, face##_buffer);     \
        unpack_##face##_buffer_device.setArg(4, depth);       \
        unpack_##face##_buffer_device.setArg(5, 0);     \
        cl::NDRange offset_plus_one(update_##dir##_offset[depth][0]+1,  \
            update_##dir##_offset[depth][1]+1); \
        enqueueKernel(unpack_##face##_buffer_device, \
                      __LINE__, __FILE__,  \
                      offset_plus_one, \
                      update_##dir##_global_size[depth], \
                      update_##dir##_local_size[depth]); \
    }

    ZERO_BOUNDARY(left, lr, x)
    ZERO_BOUNDARY(right, lr, x)
    ZERO_BOUNDARY(bottom, bt, y)
    ZERO_BOUNDARY(top, bt, y)

    generate_chunk_init_u_device.setArg(1, energy1);
    ENQUEUE_OFFSET(generate_chunk_init_u_device);
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

extern "C" void tea_leaf_ppcg_init_ocl_
(const double * ch_alphas, const double * ch_betas,
 double* theta, int* n_inner_steps)
{
    chunk.ppcg_init(ch_alphas, ch_betas, *theta, *n_inner_steps);
}

extern "C" void tea_leaf_ppcg_init_sd_kernel_ocl_
(void)
{
    chunk.ppcg_init_sd_kernel();
}

extern "C" void tea_leaf_ppcg_inner_kernel_ocl_
(int * ppcg_cur_step,
 int * max_steps,
 const int* chunk_neighbours)
{
    chunk.tea_leaf_ppcg_inner_kernel(*ppcg_cur_step, *max_steps, chunk_neighbours);
}

void CloverChunk::ppcg_init
(const double * ch_alphas, const double * ch_betas,
 const double theta, const int n_inner_steps)
{
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
}

void CloverChunk::ppcg_init_sd_kernel
(void)
{
    ENQUEUE_OFFSET(tea_leaf_ppcg_solve_init_sd_device);
}

void CloverChunk::tea_leaf_ppcg_inner_kernel
(int ppcg_cur_step, int max_steps, const int* chunk_neighbours)
{
    for (int step_depth = 1 + (halo_exchange_depth - halo_exchange_depth);
        step_depth <= halo_exchange_depth; step_depth++)
    {
        int step_offset[2] = {step_depth, step_depth};
        int step_global_size[2] = {
            x_max + (halo_exchange_depth-step_depth)*2,
            y_max + (halo_exchange_depth-step_depth)*2};

        if (chunk_neighbours[CHUNK_LEFT - 1] == EXTERNAL_FACE)
        {
            step_offset[0] = halo_exchange_depth;
            step_global_size[0] -= (halo_exchange_depth-step_depth);
        }
        if (chunk_neighbours[CHUNK_RIGHT - 1] == EXTERNAL_FACE)
        {
            step_global_size[0] -= (halo_exchange_depth-step_depth);
        }

        if (chunk_neighbours[CHUNK_BOTTOM - 1] == EXTERNAL_FACE)
        {
            step_offset[1] = halo_exchange_depth;
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
}

