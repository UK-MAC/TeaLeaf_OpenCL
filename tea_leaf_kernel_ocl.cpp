#include "ocl_common.hpp"

extern CloverChunk chunk;

// same as in fortran
#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

// jacobi solver functions
extern "C" void tea_leaf_kernel_init_ocl_
(const int * coefficient, double * dt, double * rx, double * ry)
{
    chunk.tea_leaf_init_jacobi(*coefficient, *dt, rx, ry);
}

extern "C" void tea_leaf_kernel_solve_ocl_
(const double * rx, const double * ry, double * error)
{
    chunk.tea_leaf_kernel_jacobi(*rx, *ry, error);
}

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

// used by both
extern "C" void tea_leaf_kernel_finalise_ocl_
(void)
{
    chunk.tea_leaf_finalise();
}

// copy back dx/dy and calculate rx/ry
void CloverChunk::calcrxry
(double dt, double * rx, double * ry)
{
    static int initd = 0;
    if (!initd)
    {
        // make sure intialise chunk has finished
        queue.finish();
        // celldx doesnt change after that so check once
        initd = 1;
    }

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

void CloverChunk::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    tea_leaf_cg_init_others_device.setArg(8, *rx);
    tea_leaf_cg_init_others_device.setArg(9, *ry);

    // only needs to be set once
    tea_leaf_cg_solve_calc_w_device.setArg(5, *rx);
    tea_leaf_cg_solve_calc_w_device.setArg(6, *ry);

    // copy u, get density value modified by coefficient
    tea_leaf_cg_init_u_device.setArg(6, coefficient);
    //ENQUEUE(tea_leaf_cg_init_u_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_u_device);

    // init Kx, Ky
    //ENQUEUE(tea_leaf_cg_init_directions_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_directions_device);

    // get initial guess in w, r, etc
    //ENQUEUE(tea_leaf_cg_init_others_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_others_device);

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
}

void CloverChunk::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
    //ENQUEUE(tea_leaf_cg_solve_calc_w_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

void CloverChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

    //ENQUEUE(tea_leaf_cg_solve_calc_ur_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);
    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
}

void CloverChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

    //ENQUEUE(tea_leaf_cg_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);
}

// jacobi
void CloverChunk::tea_leaf_init_jacobi
(int coefficient, double dt, double * rx, double * ry)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    tea_leaf_jacobi_init_device.setArg(6, coefficient);
    ENQUEUE(tea_leaf_jacobi_init_device);

    tea_leaf_jacobi_solve_device.setArg(0, *rx);
    tea_leaf_jacobi_solve_device.setArg(1, *ry);
}

void CloverChunk::tea_leaf_kernel_jacobi
(double rx, double ry, double* error)
{
    ENQUEUE(tea_leaf_jacobi_copy_u_device);
    ENQUEUE(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(max_red_kernels_double, reduce_buf_1);
}

// both
void CloverChunk::tea_leaf_finalise
(void)
{
    ENQUEUE(tea_leaf_finalise_device);
}

